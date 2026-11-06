"""Finite-SRAM compiler for actual per-output-channel P4/P5 weight precision.

Precision can change only at N=32 array-tile boundaries. Activation residency
survives precision boundaries; output layout remains the original full matrix.
This module describes traffic/work, not calibrated switching or timing.
"""
from collections import Counter
from flexposit_matrix_memory_plan import align,compile_matrix


def compile_channel_matrix(m,n,k,channel_precision,policy,*,a_capacity=512*1024,w_capacity=512*1024):
    if len(channel_precision)!=n or any(type(p) is not int or p not in (4,5) for p in channel_precision):
        raise ValueError('Explicit P4/P5 precision for every output channel required')
    anchor=compile_matrix(m,n,k,max(channel_precision),policy,a_capacity=a_capacity,w_capacity=w_capacity)
    tiles=[];offset=0
    for col in range(0,n,32):
        bits=channel_precision[col:col+32]
        if len(set(bits))!=1:raise ValueError('Mixed precision within one physical N=32 tile requires another hardware mode')
        precision=bits[0];payload=k//4*16*precision;slot=align(payload)+64
        if slot>w_capacity:raise ValueError('Full-K precision tile exceeds finite SRAM')
        tiles.append(dict(column=col,precision=precision,weight_payload_bytes=payload,
            weight_slot_bytes=slot,weight_hbm_offset=offset,scale_address=align(payload)))
        offset+=slot
    regions=dict(activation=dict(anchor['hbm_regions']['activation']),
        weights=dict(base=anchor['hbm_regions']['weights']['base'],bytes=offset),
        output=dict(base=anchor['hbm_regions']['weights']['base']+offset,bytes=m*n*2))
    return dict(status='COMPILED_CHANNEL_PRECISION_TRAFFIC_NOT_HARDWARE_COST',m=m,n=n,k=k,
        policy=policy,block_rows=anchor['block_rows'],column_tiles=tiles,array_rows=16,array_columns=32,
        activation_sram_bytes=a_capacity,weight_sram_bytes=w_capacity,
        output_sram_offset=anchor['output_sram_offset'],hbm_regions=regions,
        maximum_weight_slot_bytes=max(t['weight_slot_bytes'] for t in tiles),
        scale_format='32 signed 8-bit scale exponents in a dedicated 64-byte aligned line',
        overlap='Serialized reference events; no free precision switch or prefetch assumed',
        hardware_controller_admitted=False,timing_ps=None,energy_j=None)


def events(plan):
    m,n,k=plan['m'],plan['n'],plan['k'];block=plan['block_rows'];regions=plan['hbm_regions']
    def load_a(row,rows):
        return dict(op='load_activation',row=row,rows=rows,bytes=rows*k*2,
            hbm_address=regions['activation']['base']+row*k*2,sram_address=0,
            mapping='row_major_to_group_major_8byte_words')
    def load_w(tile):
        return dict(op='load_weight',column=tile['column'],bytes=tile['weight_slot_bytes'],
            hbm_address=regions['weights']['base']+tile['weight_hbm_offset'],sram_address=0)
    def compute(row,rows,tile,resident_row,resident_rows):
        yield dict(op='compute',row=row,rows=rows,column=tile['column'],resident_row=resident_row,
            resident_rows=resident_rows,groups=k//4,scale_address=tile['scale_address'],
            scale_read_bytes=32,precision=tile['precision'])
        yield dict(op='store_output',row=row,rows=rows,column=tile['column'],bytes=rows*64,
            hbm_address=regions['output']['base']+(row*n+tile['column'])*2,
            hbm_row_stride=n*2,sram_address=plan['output_sram_offset'])
    if plan['policy']=='activation_resident':
        for start in range(0,m,block):
            rows=min(block,m-start);yield load_a(start,rows)
            for tile in plan['column_tiles']:
                yield load_w(tile)
                for row in range(start,start+rows,16):
                    yield from compute(row,min(16,start+rows-row),tile,start,rows)
    elif plan['policy']=='weight_resident':
        for tile in plan['column_tiles']:
            yield load_w(tile)
            for start in range(0,m,block):
                rows=min(block,m-start);yield load_a(start,rows)
                yield from compute(start,rows,tile,start,rows)
    else:raise ValueError('Unknown loop order')


def group_words(plan,compute,group):
    if not 0<=group<plan['k']//4:raise ValueError('Group outside full K')
    tile=plan['column_tiles'][compute['column']//32]
    if tile['precision']!=compute['precision']:raise ValueError('Compute precision differs from stored codes')
    offset=compute['row']-compute['resident_row'];words=2*compute['precision']
    return dict(SRAM_A=[group*compute['resident_rows']+offset+r for r in range(compute['rows'])],
        SRAM_W=[group*words+i for i in range(words)])


def summarize(plan):
    counts=Counter();by_precision=Counter();serial=Counter();peak_a=peak_w=current_a=0
    precision_changes=0;previous=None
    for e in events(plan):
        op=e['op'];counts[op+'_events']+=1
        if op=='load_activation':
            current_a=e['bytes'];peak_a=max(peak_a,current_a)
            counts['hbm_activation_read_bytes']+=e['bytes'];counts['sram_activation_write_bytes']+=e['bytes']
        elif op=='load_weight':
            peak_w=max(peak_w,e['bytes']);counts['hbm_weight_read_bytes']+=e['bytes'];counts['sram_weight_write_bytes']+=e['bytes']
        elif op=='compute':
            peak_a=max(peak_a,current_a+e['rows']*64);precision=e['precision']
            counts['array_mac_operations']+=e['rows']*32*plan['k']
            counts['sram_activation_read_bytes']+=e['rows']*plan['k']*2
            counts['sram_weight_read_bytes']+=plan['column_tiles'][e['column']//32]['weight_payload_bytes']+32
            counts['sram_output_write_bytes']+=e['rows']*64
            by_precision[precision]+=e['rows']*32*plan['k'];serial[precision]+=e['groups']*precision
            if previous is not None and previous!=precision:precision_changes+=1
            previous=precision
        elif op=='store_output':
            counts['sram_output_read_bytes']+=e['bytes'];counts['hbm_output_write_bytes']+=e['bytes']
    if peak_a>plan['activation_sram_bytes'] or peak_w>plan['weight_sram_bytes']:raise ValueError('SRAM capacity exceeded')
    if counts['array_mac_operations']!=plan['m']*plan['n']*plan['k']:raise ValueError('Lost/repeated arithmetic')
    counts['hbm_total_bytes']=sum(counts[k] for k in ('hbm_activation_read_bytes','hbm_weight_read_bytes','hbm_output_write_bytes'))
    if counts['hbm_total_bytes']%64:raise ValueError('Unaligned traffic')
    counts['hbm_64byte_transactions']=counts['hbm_total_bytes']//64
    return dict(counts),dict(SRAM_A=peak_a,SRAM_W=peak_w),dict(mac_operations_by_precision=dict(by_precision),
        serial_dp4_group_steps_by_precision=dict(serial),precision_changes=precision_changes,
        precision_switch_latency_ps=None,precision_switch_energy_j=None)
