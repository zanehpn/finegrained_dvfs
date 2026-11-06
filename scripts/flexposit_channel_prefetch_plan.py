"""Two bounded weight buffers for variable P4/P5 column-tile payloads.

SRAM uses a fixed slot stride equal to the largest tile, while HBM retains
compact per-tile offsets. Unused SRAM bytes are neither transferred nor read.
"""
from flexposit_channel_matrix_memory_plan import compile_channel_matrix as base_compile
from flexposit_channel_matrix_memory_plan import events as base_events, group_words as base_words, summarize as base_summary
from flexposit_channel_scale_encoding import storage_contract


def compile_channel_matrix(*args, **kwargs):
    plan = base_compile(*args, **kwargs)
    if plan['policy'] != 'activation_resident':
        raise ValueError('Channel prefetch requires activation-resident order')
    stride = plan['maximum_weight_slot_bytes']
    if 2*stride > plan['weight_sram_bytes']:
        raise ValueError('Two maximum-size weight slots exceed finite SRAM')
    plan.update(weight_buffer_slots=2, weight_buffer_stride_bytes=stride,
                scale_format=storage_contract(plan['n'])['format'], scale_encoding=storage_contract(plan['n']),
                overlap='Next-column weight DMA shares the finite descriptor interface with compute')
    return plan


def events(plan):
    for event in base_events(plan):
        event = dict(event)
        if event['op'] in ('load_weight', 'compute'):
            offset = (event['column']//32 % 2)*plan['weight_buffer_stride_bytes']
            if event['op'] == 'load_weight':
                event['sram_address'] = offset
            else:
                event['weight_sram_offset'] = offset
                event['scale_address'] += offset
        yield event


def group_words(plan, compute, group):
    words = base_words(plan, compute, group)
    words['SRAM_W'] = [word+compute['weight_sram_offset']//8 for word in words['SRAM_W']]
    return words


def summarize(plan):
    counts, peak, work = base_summary(plan)
    peak['SRAM_W'] = min(2, plan['n']//32)*plan['weight_buffer_stride_bytes']
    if peak['SRAM_W'] > plan['weight_sram_bytes']:
        raise ValueError('Prefetch buffer exceeds physical capacity')
    return counts, peak, work
