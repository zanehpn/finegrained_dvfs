"""Search-free V/f planner from three insights (no trial schedules while planning).

Insight 1 (free points): an operator's compute time C follows its tile precisions (94+20(p-4) cycles
per tile) and its HBM time M its weight bytes, both known at compile time. Any point whose stretch s
satisfies C*s + regulator transitions <= M costs no latency, so the free point is a lookup on C/M.
Insight 2 (where slack goes): for a compute-bound operator a one-level step costs
dT = max(C*s', M) - max(C*s, M) and saves dE = E_dyn*(r_s - r_s') - P_bg*dT. Operators just past the
balance point (small dT) and narrow-precision ones (higher energy per cycle) come first, so one sort
of all steps by dE/dT gives the allocation order.
Insight 3 (which budget): TTFT slack pays for prefill slowdowns and TPOT slack for decode slowdowns,
each with its own budget. A decode operator gains from TPOT slack only past its free point, i.e. when
its stretched compute at a lower point would exceed its HBM time; with one request and a long context
C << M even at 0.45 V, so TPOT slack buys nothing, while decode batches bring C close to M and make
the steps past the free point the cheapest ones (small dT, full MAC energy).
Latency model while planning: a request's TTFT grows by the summed dT of its own and earlier prefills
on the P core (prefix constraints); decode TPOT grows by the summed dT of decode steps over the
nominal decode-core time. One schedule check follows; a violation drops the lowest-ratio accepted
steps of the violating phase (20% per round) and re-checks.
Also returns a Lagrangian lower bound on energy for the planning latency model (TTFT of the last
request and total decode time only; the other constraints are relaxed), used to report the gap.
"""
import os
import collections, math, time
import qwen_first_order_v1 as q


def _bg_power(comp, T):
    """Time-proportional chip power (J/ps) charged for added latency.

    HBM controller and DRAM-die background, leakage and link/frontend; the memory subsystem's retained logic
    is billed by utilisation, so stretching a compute-bound operator adds little of it (measured marginal
    background on Qwen decode batch 4: 3.6 W against 3.8 W from these terms).
    """
    keys = ('dram_die_background', 'controller_background', 'frontend_controller', 'pd_link_fifo', 'array_leakage')
    return sum(comp.get(k, 0.) for k in keys)/T


def model_delta(assign, k, supply, cfield, bfield, bg):
    """Energy change of an assignment in the planning model (dynamic energy plus background for added time)."""
    r = q.RATIO(supply); out = 0.
    for i, p in assign.items():
        o = q.OPS[i]; C = o[cfield]/k; M = o[bfield]/q.BW_R*1e12; st = q.STRETCH[o['unit']]; E = _dyn(o, k)
        out += E*(r[p]-1)+bg*(max(C*st[p], M)-max(C, M))
    return out


def _dyn(o, k):
    if o['unit'] == 'MATRIX': return q.S.mac_in/q.S.macs_total*o['macs']*o.get('mac_e', 1.)
    if o['unit'] == 'VU': return q.J.vu_in/q.J.vu_C*o['C']
    if o['unit'] == 'ATT': return q.J.att_in/q.J.att_C*o['C']
    return 0.


def _cs(o, C, st, pi):
    """Compute time at point pi, with the schedule model's calibrated response to a stretch."""
    cal = q.CAL.get(f"{o['engine']}/{o['family']}") if q.CAL else None
    return C+cal['g']*(C*st[pi]-C) if cal and 'g' in cal else C*st[pi]


def plan(k, supply, b_ttft, b_tpot, cfield, bfield, per_req, outs, coarse=False, lat_ref=None):
    """coarse: one point for all domains of a core per operator (MEM stretched with it); lat_ref: objective latencies (e.g. uniform-P8) instead of this plan's nominal run."""
    t0 = time.monotonic(); eta = q.L.SUPPLY[supply]; r = q.RATIO(supply); npts = len(q.points)
    q.MEM_MODE = 'cg'; sc0 = q.run(k, {}, supply, False); E_ref, comp, _ = q.price(k, supply, {}, sc0, pg=True); q.MEM_MODE = 'dvfs'
    if os.environ.get('PLAN_SELF_REF') == '1': lat_ref = q.run(k, {}, supply, coarse, cfield, bfield)[3]  # judge the plan against the costing's own nominal
    R0 = per_req(lat_ref if lat_ref is not None else q.run(k, {}, supply, coarse)[3]); bg = _bg_power(comp, sc0[0])
    # Insight 1: free points from C/M with the planner's costing
    assign = {}; dur = {}
    for i, o in enumerate(q.OPS):
        u = o['unit']
        if not u: continue
        C = o[cfield]/k; M = o[bfield]/q.BW_R*1e12; st = q.STRETCH[u]
        best = 0
        for pi in range(1, npts):
            mb = o['sram_bytes']/k/q.CAP*q.J.MEM_STRETCH[pi] if coarse and o['sram_bytes'] else 0.
            if max(_cs(o, C, st, pi), mb)+(.7-q.points[pi]['v'])*q.J.IVR_PS_PER_V*2 <= M and r[pi] < r[best]: best = pi
        if best: assign[i] = best
        dur[i] = (C, M, st)
    # Insight 2/3: candidate steps beyond the free point, with their latency owner
    base_free = dict(assign); accepted_first = {}
    nreq = len(R0); steps = []
    d_nominal_total = 0.
    for i, o in enumerate(q.OPS):
        if not o['unit']: continue
        C, M, st = dur[i]; ph = 'P' if o['engine'] == 'P' else 'D'
        if ph == 'D': d_nominal_total += max(C, M)
        m = o['id'].split('/')[0]; owner = int(m[1:]) if m.startswith('r') else None
        E = _dyn(o, k); prev = assign.get(i, 0)
        pts = [(max(_cs(o, C, st, pi), M), E*r[pi]) for pi in range(npts)]
        # lower convex hull from the free point (keeps marginal ratios decreasing)
        hull = [prev]
        for pi in range(prev+1, npts):
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                s1 = (pts[a][1]-pts[b][1])/max(1e-9, pts[b][0]-pts[a][0]); s2 = (pts[b][1]-pts[pi][1])/max(1e-9, pts[pi][0]-pts[b][0])
                if s2 >= s1: hull.pop()
                else: break
            hull.append(pi)
        for a, b in zip(hull, hull[1:]):
            dT = pts[b][0]-pts[a][0]; dE = pts[a][1]-pts[b][1]-bg*dT
            if dE > 0: steps.append((dE/max(dT, 1e-9), i, a, b, dT, dE, ph, owner))
    steps.sort(key=lambda s: -s[0])
    # budgets: prefix TTFT constraints (P-core order = request index) and total decode time
    FIX2 = os.environ.get('PLAN_FIX2') == '1'; floor = float(os.environ.get('TTFT_FLOOR_PS', 0))
    ttft_cap = [max(b_ttft*R0[j][0], floor) for j in range(nreq)]; used = [0.]*nreq
    d_cap = b_tpot*d_nominal_total; d_used = 0.; accepted = []
    ns = list(outs) if isinstance(outs, (list, tuple)) else [outs]*nreq; dreq = [0.]*nreq
    dreq_cap = [b_tpot*R0[j][1]*(ns[j]-1) for j in range(nreq)]
    level = {}
    for st_ in steps:
        ratio, i, a, b, dT, dE, ph, owner = st_
        if level.get(i, assign.get(i, 0)) != a: continue  # steps of one op are taken in order
        if ph == 'P':
            if owner is None: continue
            if any(sum(used[:j+1])+dT > ttft_cap[j] for j in range(owner, nreq)): continue
            used[owner] += dT
        else:
            mem = q.OPS[i].get('members') if FIX2 else None
            if mem is None:
                if d_used+dT > d_cap: continue
            elif any(dreq[j]+dT > dreq_cap[j] for j in mem): continue
            else:
                for j in mem: dreq[j] += dT
            d_used += dT
        level[i] = b; accepted.append(st_)
    for i, b in level.items(): assign[i] = b
    plan_wall = time.monotonic()-t0
    # one schedule check; if it fails, find the water level: the longest prefix of each phase's sorted
    # steps that still meets the objectives (bisection, ~log2(steps) schedule checks per phase)
    tgt = [(t0_+max(t0_*b_ttft, floor), d0*(1+b_tpot)) for t0_, d0 in R0]
    def ok(lat, ph=None): return all((ph == 'D' or t <= a+1) and (ph == 'P' or d <= b+1) for (a, b), (t, d) in zip(tgt, per_req(lat)))
    def build(nP, nD):
        a = {i: p for i, p in base_free.items()}
        for s_ in accP[:nP]+accD[:nD]: a[s_[1]] = s_[3]
        return {i: p for i, p in a.items() if p}
    accP = [s_ for s_ in accepted if s_[6] == 'P']; accD = [s_ for s_ in accepted if s_[6] == 'D']
    nP, nD = len(accP), len(accD); rounds = 0
    FIX = FIX2 or os.environ.get('PLAN_PHASE_FIX') == '1'
    if os.environ.get('PLAN_DIAG'):
        def viol(a):
            R = per_req(q.run(k, a, supply, coarse, cfield, bfield)[3])
            return [sum(t > x*(1+b_ttft)+1 for (x, _), (t, _) in zip(R0, R)), sum(d > y*(1+b_tpot)+1 for (_, y), (_, d) in zip(R0, R)),
                    max(t/x-1 for (x, _), (t, _) in zip(R0, R)), max(d/y-1 for (_, y), (_, d) in zip(R0, R))]
        print('DIAG full', viol(build(nP, nD)), 'free', viol(build(0, 0)), 'P only', viol(build(nP, 0)), 'D only', viol(build(0, nD)), 'nP nD', nP, nD, flush=True)
    if os.environ.get('PLAN_DUMP'):
        import json
        json.dump(dict(R0=R0, free=per_req(q.run(k, build(0, 0), supply, coarse, cfield, bfield)[3]), full=per_req(q.run(k, build(nP, nD), supply, coarse, cfield, bfield)[3])), open(os.environ['PLAN_DUMP'], 'w')); os._exit(0)
    if not ok(q.run(k, build(nP, nD), supply, coarse, cfield, bfield)[3]):
        if FIX2:  # a request the free points already push over cannot be helped by dropping steps: hold it at its free-point latency
            Rf = per_req(q.run(k, build(0, 0), supply, coarse, cfield, bfield)[3]); tgt = [(max(a, f[0]), max(b, f[1])) for (a, b), f in zip(tgt, Rf)]
        for which in ('D', 'P'):
            if FIX and ok(q.run(k, build(nP, nD), supply, coarse, cfield, bfield)[3], which): continue
            lo, hi = 0, (nD if which == 'D' else nP)
            while lo < hi:
                mid = (lo+hi+1)//2; rounds += 1
                trial = build(nP, mid) if which == 'D' else build(mid, nD)
                if ok(q.run(k, trial, supply, coarse, cfield, bfield)[3], which if FIX else None): lo = mid
                else: hi = mid-1
            if which == 'D': nD = lo
            else: nP = lo
            if ok(q.run(k, build(nP, nD), supply, coarse, cfield, bfield)[3]): break
        if os.environ.get('PLAN_DIAG'): print('DIAG after nP nD', nP, nD, 'of', len(accP), len(accD), viol(build(nP, nD)), flush=True)
    assign = build(nP, nD); accepted = accP[:nP]+accD[:nD]
    return assign, dict(E_ref=E_ref, steps=len(steps), accepted=len(accepted), backoff_rounds=rounds, plan_wall_s=plan_wall,
                        model_delta_j=model_delta(assign, k, supply, cfield, bfield, bg),
                        bg_w=bg*1e12, ttft_used=used, d_used=d_used, d_cap=d_cap)


def lagrange_bound(k, supply, b_ttft, b_tpot, cfield, bfield, per_req):
    """Lower bound on energy of the planning latency model: min sum_i E_i(p_i)+bg*T_i s.t. two relaxed budgets."""
    r = q.RATIO(supply); q.MEM_MODE = 'cg'; sc0 = q.run(k, {}, supply, False); E_ref, comp, _ = q.price(k, supply, {}, sc0, pg=True)
    q.MEM_MODE = 'dvfs'; R0 = per_req(sc0[3]); bg = _bg_power(comp, sc0[0])
    items = {'P': [], 'D': []}; base = 0.
    for i, o in enumerate(q.OPS):
        if not o['unit']: continue
        C = o[cfield]/k; M = o[bfield]/q.BW_R*1e12; st = q.STRETCH[o['unit']]; E = _dyn(o, k)
        opts = [(max(_cs(o, C, st, pi), M)-max(C, M), E*r[pi]-E+bg*(max(_cs(o, C, st, pi), M)-max(C, M))) for pi in range(len(q.points))]
        items['P' if o['engine'] == 'P' else 'D'].append(opts)
    caps = {'P': b_ttft*R0[-1][0], 'D': b_tpot*sum(max(o[cfield]/k, o[bfield]/q.BW_R*1e12) for o in q.OPS if o['unit'] and o['engine'] == 'D')}
    total = 0.
    for ph, its in items.items():
        def dual(lam): return sum(min(e+lam*t for t, e in opts) for opts in its)-lam*caps[ph]
        lo, hi = 0., 1.
        while dual(hi) > dual(hi/2) and hi < 1e6: hi *= 2
        for _ in range(60):  # concave in lambda: ternary search for the maximum
            m1, m2 = lo+(hi-lo)/3, hi-(hi-lo)/3
            if dual(m1) < dual(m2): lo = m1
            else: hi = m2
        total += dual((lo+hi)/2)
    return total  # lower bound on the planning-model energy change (J); compare with model_delta_j
