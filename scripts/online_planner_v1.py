"""Causal MIDAS planner: picks each operator's point when it is dispatched, knowing only requests that have arrived.

Offline, per operator: the lower convex hull of (latency, energy) over the points, from peak V/f. The free point is
simply the first hull vertices, and its real latency cost (transitions, calibrated overheads) is charged like any step.
Online:
- A shadow clock tracks when a nominal machine would be at the same operator, using observed HBM contention. Lag is
  now minus shadow, so the loop is closed on measured time and needs no repair afterwards. Observed contention belongs to
  the slowed run, so the shadow drifts high; with nom the clock is a nominal twin instead, a model of the machine at peak
  V/f advanced over the arrivals seen so far. It is causal because the nominal machine at time x depends only on arrivals before x.
- Prefill: at admission and at each layer the request's remaining steps are taken by energy saved per delay until the
  budget left is spent. A request queued behind it tightens that budget as soon as it has arrived.
- Decode: each step's budget is the smallest TPOT credit of its members, which needs no output length.
"""
import math, time
import bisect, os
import qwen_first_order_v1 as q
import insight_planner_v1 as IP
ALLDOM = os.environ.get('ON_ALLDOM') == '1'; MEMSTEP = os.environ.get('ON_MEMPLAN') == '1'; RETAWARE = os.environ.get('ON_RETAWARE') == '1'
SLOHEDGE = os.environ.get('ON_SLOHEDGE') == '1'


_F = {}
RETBG = lambda pw, k: BGRET.get('w', 0.)
BGRET = {}
def _domain_factor(r):
    """Array energy ratio at each MAC point, with CTRL/DEC/OUT at the lowest point their clock headroom allows (array_domains_v1)."""
    k = tuple(r)
    if k not in _F:
        import array_domains_v1 as AD
        st = q.J.MAC_STRETCH; rest = 1-sum(AD.SHARE[d] for d in AD.HEADROOM); o = []
        for pi in range(len(r)):
            x = rest*r[pi]
            for d, h in AD.HEADROOM.items(): x += AD.SHARE[d]*r[max(j for j in range(len(r)) if st[j] <= h*st[pi]+1e-12)]
            o.append(x)
        _F[k] = o
    return _F[k]


def energy(k, supply, A, sc):
    """Request energy of a plan, with every domain priced when ON_ALLDOM is set."""
    e = q.price(k, supply, A, sc, pg=True)[0]
    if ALLDOM:
        import array_domains_v1 as AD
        e -= AD.saving(A, supply)
    return e


class Online:
    def __init__(s, k, supply, bt, bd, floor, guard, bg, cf='C_true', bf='B_true', reserve=True, lag_q=None, hedge=False, warm=1, nom=None, open_loop=False, slo=None, own=None):
        s.k, s.bt, s.bd, s.floor, s.g, s.bg, s.cf, s.bf, s.reserve = k, bt, bd, floor, guard, bg, cf, bf, reserve; s.bgv = bg; s.lag_q = lag_q; s.hist = []; s.hedge = hedge; s.warm = warm; s.nom = nom  # nom: op -> dispatch time on the nominal twin
        s.tlow = {}; s.slo = slo; s.own = own; s.cur = None; s.tc = {}; s.Sc = {}; s.tj = {}; s.tok = {}  # slo: ttft and tpot per request, as given; own: this costing's nominal twin (rec: op -> dispatch, end: prefill ends)
        s.open_loop = open_loop; s.olag = dict(P=0., D=0.); s.dt = {}  # open_loop: budget spent = sum of the costing's own step delays, for the attribution study only
        t0 = time.perf_counter(); r = q.RATIO(supply); n = len(q.points); s.r = r
        s.key = {}; s.items = {}; s.S = {}; s.ops = {}; s.t0 = {}; s.m = {}; s.extra = {}; s.c0 = {}; s.nsteps = 0
        for i, o in enumerate(q.OPS):
            oid = o['id'].split('/'); key = oid[0] if oid[0][0] == 'g' and not oid[1].startswith('t') else oid[0]+'/'+oid[1]
            emit = oid[-1] == 'emit' and o['engine'] == 'D'
            s.key[i] = None if emit else key
            # The shadow clock and budgets use true times for every costing: the objectives are facts, and only the step
            # ranking below follows the costing, so a costing is judged on what it picks and not on a wrong clock.
            Ct = o['C_true']/k; Mt = o['B_true']/q.BW_R*1e12; s.m[i] = Mt; s.c0[i] = Ct
            cal = q.CAL.get(f"{o['engine']}/{o['family']}") if q.CAL else None
            ext = (cal.get('a', 0.)*min(Ct, Mt)+cal.get('y', 0.)*Ct) if cal else 0.
            s.extra[i] = (ext, cal)
            t = max(Ct, Mt)+ext; s.t0[i] = t
            C = o[cf]/k; M = o[bf]/q.BW_R*1e12; ex0 = (cal.get('a', 0.)*min(C, M)+cal.get('y', 0.)*C) if cal else 0.
            s.tc[i] = max(C, M)+ex0
            if key and not emit: s.Sc[key] = s.Sc.get(key, 0.)+s.tc[i]
            if key and not emit:
                s.S[key] = s.S.get(key, 0.)+t; s.ops.setdefault(key, []).append(i)
            if not o['unit'] or emit: continue
            st = q.STRETCH[o['unit']]; E = IP._dyn(o, k)
            def lat(pi):
                c = C*st[pi]
                if cal and 'g' in cal: c = C+cal['g']*(c-C)
                tr = (.7-q.points[pi]['v'])*q.J.IVR_PS_PER_V*2 if pi else 0.
                return max(c+tr, M)+ex0+(cal.get('b', 0.)*max(0., c-C) if cal else 0.)
            f = _domain_factor(r) if ALLDOM and o['unit'] == 'MATRIX' else r  # a MAC point also sets how low CTRL, DEC and OUT may go
            pts = [(lat(pi), E*f[pi]+bg*lat(pi)) for pi in range(n)]  # background power charged with the latency
            if RETAWARE:  # retained logic is not time-proportional: slowing an op lowers its utilisation, so its MEM rail follows and energy falls with voltage
                pw = q.J.RET_W_PER_ENGINE*k**q.R.alpha_for(k)*1e-12; bn = max(0., bg-RETBG(pw, k)); sv = q.J.MEM_STRETCH; mb = o['sram_bytes']/k/q.CAP if o['sram_bytes'] else 0.
                def ret(pi):
                    L = lat(pi); u = min(1., mb/L) if mb else min(1., C/L) if L > 0 else 1.
                    j = max(0, bisect.bisect_right(sv, 1/max(u, 1e-9)+1e-12)-1)
                    return pw*r[j]*u*L
                pts = [(lat(pi), E*f[pi]+ret(pi)+bn*lat(pi)) for pi in range(n)]
            hull = [0]
            for pi in range(1, n):
                if pts[pi][1] >= pts[hull[-1]][1]: continue
                while len(hull) >= 2:
                    a, b = hull[-2], hull[-1]
                    if (pts[a][1]-pts[b][1])/max(1e-9, pts[b][0]-pts[a][0]) <= (pts[b][1]-pts[pi][1])/max(1e-9, pts[pi][0]-pts[b][0]): hull.pop()
                    else: break
                hull.append(pi)
            s.tlow[i] = pts[hull[-1]][0]
            for a, b in zip(hull, hull[1:]):
                dT = max(0., pts[b][0]-pts[a][0]); dE = pts[a][1]-pts[b][1]
                s.items.setdefault(key, []).append((dE/max(dT, 1e-9), i, a, b, dT)); s.nsteps += 1; s.dt[(i, b)] = dT
        for v in s.items.values(): v.sort(key=lambda x: -x[0])
        s.table_s = time.perf_counter()-t0
        s.assign = {}; s.lv = {}; s.VP = 0.; s.VPend = {}; s.curP = None; s.VD = 0.; s.curD = None; s.acc = 0.; s.join = {}; s.Vn0 = {}
        s.decide_s = 0.; s.decisions = 0; s.done = set(); s.arr = [q.ARRIVAL_END.get(f'arr{j}', 0.) for j in range(s._nreq())]

    def _nreq(s): return 1+max((int(x[1:].split('/')[0]) for x in s.S if x[0] == 'r'), default=0)
    def _dti(s, i): return {(i, b): s.dt[(i, b)] for pi in range(1, len(q.points)) for b in [pi] if (i, b) in s.dt}
    def _cap(s, T0): return s.g*max(s.bt*T0, s.floor)

    def _take(s, key, budget):
        lv = {}
        for ratio, i, a, b, dT in s.items.get(key, ()):
            if i in s.done or lv.get(i, 0) != a: continue
            if dT > budget: continue
            budget -= dT; lv[i] = b
        return lv

    def _prefill(s, key, j, t):
        t0 = time.perf_counter(); lag = s.olag['P'] if s.open_loop else t-s.VP; B = s.capj-lag
        if s.slo:  # fair view: the SLO is given, elapsed time is measured, and how long the rest takes comes from this costing's own nominal twin
            own = s.own; rem = t-own['rec'][s.cur]; T = s.slo['ttft'][j]; s.capj = s._cap(T); lag = rem+own['end'][j]-(s.arr[j]+T); B = s.capj-lag
            if s.reserve:
                for k in range(j+1, len(s.arr)):
                    if s.arr[k] > t: break
                    Tk = s.slo['ttft'][k]; B = min(B, s._cap(Tk)-(rem+own['end'][k]-(s.arr[k]+Tk)))
            if SLOHEDGE and len(s.hist) >= s.warm:  # lag outlives this request: an arrival that would have found the machine idle queues behind it. Keep lag within the slack a typical later arrival has, weighed by the chance one comes before this backlog drains
                h = sorted(s.hist); cq = h[int(s.lag_q*(len(h)-1))]; n = sum(1 for a in s.arr[:j+1] if a <= t); rate = (n-1)/max(t-s.arr[0], 1e9) if n > 1 else 0.
                last = max((k for k in range(j, len(s.arr)) if s.arr[k] <= t), default=j); p = 1-math.exp(-rate*max(own['end'][last]-own['rec'][s.cur], 0.)); sj = T-(own['end'][j]-s.arr[j]); B = min(B, p*cq+(1-p)*sj-rem)
        elif s.reserve:  # requests that have arrived and wait behind this one
            V = s.Vend
            for k in range(j+1, len(s.arr)):
                if s.arr[k] > t: break
                sk = s.S[f'r{k}/t0']; B = min(B, s._cap(V+sk-s.arr[k])-lag); V += sk
        if s.lag_q is not None and s.hist and (s.floor > 0 or len(s.hist) >= s.warm):  # no floor and no history means no evidence of later arrivals: reserve nothing. Lag outlives this request: keep it within what a typical later arrival could absorb
            h = sorted(s.hist); cq = s._cap(h[int(s.lag_q*(len(h)-1))]) if len(h) >= s.warm else s._cap(0.)
            if s.hedge:  # weigh by the chance that someone arrives before this prefill drains, from the arrival rate seen so far
                n = sum(1 for a in s.arr[:j+1] if a <= t); rate = (n-1)/max(t-s.arr[0], 1e9) if n > 1 else 0.
                p = 1-math.exp(-rate*max(s.Vend-t, 0.)-rate*s.capj); cq = p*cq+(1-p)*s.capj
            B = min(B, cq-lag)
        lv = s._take(key, B); s.lv.update(lv)
        if MEMSTEP: s._mem(key, B, lv)
        s.decide_s += time.perf_counter()-t0; s.decisions += 1

    def _mem(s, key, B, lv):
        """Budget left once every compute step of this prefill is taken buys MEM voltage, when the energy saved beats the background power of the delay."""
        ops = [i for i in s.ops.get(key, ()) if i not in s.done and q.OPS[i]['sram_bytes'] and i in s.tlow]
        for ratio, i, a, b, dT in s.items.get(key, ()):
            if i not in s.done and lv.get(i, 0) < b: return  # compute steps left: they are worth far more per unit of delay
        B -= sum(dT for ratio, i, a, b, dT in s.items.get(key, ()) if i not in s.done)
        if B <= 0 or not ops: return
        st = q.J.MEM_STRETCH; pw = q.J.RET_W_PER_ENGINE*s.k**q.R.alpha_for(s.k)*1e-12; mb = {i: q.OPS[i]['sram_bytes']/s.k/q.CAP for i in ops}; best = 0; prev = 0.
        for j in range(1, len(st)):
            dt = sum(max(0., mb[i]*st[j]-s.tlow[i]) for i in ops)
            if dt > B: break
            if pw*(s.r[j-1]-s.r[j])*sum(mb.values()) <= s.bgv*(dt-prev): break
            best = j; prev = dt
        for i in ops: q.MEMPLAN[i] = best

    def _decode(s, key, mem, t):
        t0 = time.perf_counter(); lag = s.olag['D'] if s.open_loop else t-s.VD; n = s.S[key]; B = 1e30
        if s.slo:  # TPOT is an absolute target: token credit minus measured elapsed, minus this costing's estimate of the step
            for r in mem:
                s.tj.setdefault(r, t); s.tok[r] = s.tok.get(r, 0)+1
                B = min(B, (1+s.g*s.bd)*s.slo['tpot'][r]*s.tok[r]-(t-s.tj[r])-s.Sc.get(key, n))
            s.lv.update(s._take(key, B)); s.decide_s += time.perf_counter()-t0; s.decisions += 1; return
        for r in mem:
            N = s.VD-s.Vn0[r]
            B = min(B, s.g*s.bd*(N+n)-(lag-s.join[r]))
        s.lv.update(s._take(key, B)); s.decide_s += time.perf_counter()-t0; s.decisions += 1

    def live(s, i, t, hbm_free, queues, free):
        o = q.OPS[i]; key = s.key[i]; M = s.m[i]; C = s.c0[i]; ex, cal = s.extra[i]; s.cur = i
        oe = 'D' if o['engine'] == 'P' else 'P'; hs = max(t, hbm_free) if M else t
        dn = max(C, hs+M-t)+ex
        if cal and cal.get('pc') and M and queues[oe] and free[oe] > t: dn += cal['pc']*M
        if o['engine'] == 'P':
            if s.nom is not None: s.VP = s.nom[i]
            if key != s.curP:
                j = int(key[1:].split('/')[0]); s.curP = key; s.VP = max(s.VP, s.arr[j]); s.Vend = s.VP+s.S[key]
                s.capj = s._cap(s.Vend-s.arr[j]); s.hist.append((s.slo['ttft'][j]-(s.own['end'][j]-s.arr[j]) if SLOHEDGE else s.slo['ttft'][j]) if s.slo else s.Vend-s.arr[j]); s._prefill(key, j, t)
            elif o['id'].endswith('input_norm'):
                s._prefill(key, int(key[1:].split('/')[0]), t)
            s.VP += dn
            s.VPend[int(key[1:].split('/')[0])] = s.VP
        else:
            if key is not None and key != s.curD:
                mem = o.get('members') or [int(key[1:].split('/')[0]) if key[0] == 'r' else 0]
                s.VD = s.nom[i] if s.nom is not None else max([s.VD+s.acc]+[s.VPend.get(r, t) for r in mem if r not in s.join]); s.acc = 0.
                for r in mem:
                    if r not in s.join: s.join[r] = None; s.Vn0[r] = s.VPend.get(r, s.VD)
                lag = s.olag['D'] if s.open_loop else t-s.VD
                for r in mem:
                    if s.join[r] is None: s.join[r] = lag
                s.curD = key; s._decode(key, mem, t)
            s.acc += dn
        s.done.add(i); p = s.lv.pop(i, 0)
        if p:
            s.assign[i] = p
            if s.open_loop: s.olag[o['engine']] += sum(d for (j, b), d in s._dti(i).items() if b <= p)
        return p


def nominal_dispatch(k, supply):
    """Dispatch time of every op on the nominal twin. Same run as the reference, so call it in the same memory mode."""
    rec = {}
    def live(i, t, *_): rec[i] = t; return 0
    q.LIVE = live
    try: q.run(k, {}, supply, False)
    finally: q.LIVE = None
    return rec


def own_twin(k, supply, cf, bf, per, nreq, arr):
    """Nominal twin on a costing's own times: what this costing believes the machine does at peak V/f."""
    rec = {}
    def live(i, t, *_): rec[i] = t; return 0
    q.LIVE = live
    try: sc = q.run(k, {}, supply, False, cf, bf)
    finally: q.LIVE = None
    return dict(rec=rec, end=[arr[j]+x[0] for j, x in enumerate(per(sc[3]))])


def run_online(k, supply, bt, bd, floor, guard, bg, cf='C_true', bf='B_true', **o):
    c = Online(k, supply, bt, bd, floor, guard, bg, cf, bf, **o); q.LIVE = c.live; q.MEMPLAN.clear()
    try: sc = q.run(k, {}, supply, False)
    finally: q.LIVE = None
    return c.assign, sc, dict(table_s=c.table_s, decide_s=c.decide_s, decisions=c.decisions, steps=c.nsteps, ops=len(q.OPS))
