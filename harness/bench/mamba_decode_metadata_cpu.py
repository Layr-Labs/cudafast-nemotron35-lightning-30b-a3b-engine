"""CPU-only four-group proxy using the final candidate's guarded call path."""
import ast
import json
from pathlib import Path
import statistics
import sys
import time

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'harness/vllm_resident'))
import test_mamba_decode_reuse as actual
import torch

torch.set_num_threads(1)
source = repo / 'vllm/vllm/v1/worker/gpu_model_runner.py'
function = next(node for node in ast.walk(ast.parse(source.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name == '_can_reuse_mamba_metadata')
namespace = {'prefill_fast': actual.fast, 'Mamba2AttentionMetadataBuilder': actual.Builder}
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), 'exec'), namespace)
can_reuse = namespace['_can_reuse_mamba_metadata']
builders = [actual.make_builder(depth=3, full_graph=True, cap=8) for _ in range(4)]
tables = [torch.tensor([[1000*g+1, 1000*g+2, 1000*g+3, 1000*g+4]], dtype=torch.int32) for g in range(4)]
commons = [actual.common([4], [800], table) for table in tables]
accepted = torch.tensor([2], dtype=torch.int32)

def step(instrument=False):
    cache = {}
    output = []
    built = reused = 0
    for builder, common in zip(builders, commons):
        # Identical frozen spec and exact builder type, as in the runner.
        key = (3, type(builder))
        if key in cache and can_reuse(builder, cache[key]):
            metadata = builder.update_block_table(cache[key], common.block_table_tensor, common.slot_mapping)
            reused += 1
        else:
            metadata = actual.build(builder, common, accepted)
            built += 1
            if can_reuse(builder, metadata):
                cache[key] = metadata
        output.append(metadata)
    return (output, built, reused) if instrument else output

actual.fast.MAMBA_DECODE_META_REUSE = False
baseline, before_built, before_reused = step(True)
snapshots = [actual.fast.snapshot_metadata(metadata) for metadata in baseline]
actual.fast.MAMBA_DECODE_META_REUSE = True
candidate, after_built, after_reused = step(True)
for before, after in zip(snapshots, candidate):
    actual.fast.check_same(before, after, 'integrated-proxy')
assert (before_built, before_reused, after_built, after_reused) == (4, 0, 1, 3)
samples = {'fresh_builds': [], 'guarded_reuse': []}
for i in range(9):
    order = [('fresh_builds', False), ('guarded_reuse', True)]
    if i % 2:
        order.reverse()
    for name, enabled in order:
        actual.fast.MAMBA_DECODE_META_REUSE = enabled
        for _ in range(100):
            step()
        started = time.perf_counter_ns()
        for _ in range(2000):
            step()
        samples[name].append((time.perf_counter_ns() - started) / 2000 / 1000)
actual.fast.MAMBA_DECODE_META_REUSE = True
medians = {name: statistics.median(values) for name, values in samples.items()}
report = {'environment': {'device': 'CPU', 'torch': torch.__version__, 'threads': 1},
          'method': 'actual final builder/helper and runner predicate, four-group depth3 batch1 CPU proxy, 9 alternating reps x2000 steps; no GPU or whole-model timing',
          'build_update_counts': {'baseline': [before_built, before_reused], 'candidate': [after_built, after_reused]},
          'sample_us_per_step': samples, 'median_us_per_step': medians,
          'saving_us_per_step': medians['fresh_builds']-medians['guarded_reuse']}
print(json.dumps(report, indent=2))
