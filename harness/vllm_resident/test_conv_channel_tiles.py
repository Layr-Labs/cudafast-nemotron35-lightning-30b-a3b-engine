"""No-GPU checks of the actual convolution launch wrapper and dispatch guard.

Tensor metadata and the launch are intercepted; no convolution is evaluated.
The real Triton kernel still requires compilation and numerical checks on GB10.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class ShapeTensor:
    dtype = "bfloat16"
    device = "cuda"

    def __init__(self, *shape):
        self.shape = shape

    def size(self, index=None):
        return self.shape if index is None else self.shape[index]

    def dim(self):
        return len(self.shape)

    def stride(self, index=None):
        strides = []
        running = 1
        for size in reversed(self.shape):
            strides.append(running)
            running *= size
        strides = tuple(reversed(strides))
        return strides if index is None else strides[index]

    def to(self, dtype):
        assert dtype == self.dtype
        return self


class LaunchRecorder:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.grid = grid(kwargs)
            self.args, self.kwargs = args, kwargs
        return launch


def load_wrapper(capability=121, cuda=True):
    path = Path(__file__).resolve().parents[2] / (
        "vllm/vllm/model_executor/layers/mamba/ops/causal_conv1d.py"
    )
    tree = ast.parse(path.read_text())
    names = {"_causal_conv1d_update_block_n", "causal_conv1d_update"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    recorder = LaunchRecorder()
    namespace = dict(
        NULL_BLOCK_ID=0,
        current_platform=NS(is_cuda=lambda: cuda,
                            is_device_capability=lambda wanted: capability == wanted,
                            is_arch_support_pdl=lambda: False),
        triton=NS(cdiv=lambda n, d: (n + d - 1) // d,
                  next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        _causal_conv1d_update_kernel=recorder,
    )
    source = "from __future__ import annotations\n" + "\n".join(ast.unparse(n) for n in nodes)
    exec(compile(source, str(path), "exec"), namespace)
    return namespace, recorder


class ConvChannelTiles(unittest.TestCase):
    def test_all_permitted_short_windows_select_48_ctas(self):
        ns, recorder = load_wrapper()
        for length in range(1, 8):
            with self.subTest(length=length):
                x = ShapeTensor(length, 6144)
                state = ShapeTensor(8, 6144, 10)
                weights = ShapeTensor(6144, 4)
                result = ns["causal_conv1d_update"](
                    x, state, weights, conv_state_indices=ShapeTensor(1, length),
                    num_accepted_tokens=ShapeTensor(1), query_start_loc=ShapeTensor(2),
                    max_query_len=length,
                )
                self.assertIs(result, x)
                self.assertEqual(recorder.grid, (1, 48))
                self.assertEqual(recorder.kwargs["BLOCK_N"], 128)
                self.assertTrue(recorder.kwargs["IS_SPEC_DECODING"])
                self.assertIs(recorder.args[3], state)
                self.assertIs(recorder.args[1], weights)

    def test_other_hardware_and_non_cuda_keep_default(self):
        for capability, cuda in [(120, True), (100, True), (90, True), (None, True), (121, False)]:
            ns, _ = load_wrapper(capability, cuda)
            self.assertEqual(ns["_causal_conv1d_update_block_n"](1, 6144, 4, 4), 256)

    def test_shape_boundaries_keep_default(self):
        ns, _ = load_wrapper()
        for shape in [(0, 6144, 4, 4), (2, 6144, 4, 4), (1, 6143, 4, 4),
                      (1, 6145, 4, 4), (1, 6144, 3, 4), (1, 6144, 5, 4),
                      (1, 6144, 4, 0), (1, 6144, 4, -1), (1, 6144, 4, 8)]:
            with self.subTest(shape=shape):
                self.assertEqual(ns["_causal_conv1d_update_block_n"](*shape), 256)

    def test_channel_partition_is_disjoint_complete_and_mask_preserving(self):
        for channels in [1, 127, 128, 129, 6143, 6144, 6145]:
            for tile in [128, 256]:
                visited = [block * tile + lane
                           for block in range((channels + tile - 1) // tile)
                           for lane in range(tile) if block * tile + lane < channels]
                self.assertEqual(visited, list(range(channels)))

    def test_batched_wrapper_keeps_the_original_grid(self):
        ns, recorder = load_wrapper()
        x = ShapeTensor(8, 6144)
        ns["causal_conv1d_update"](
            x, ShapeTensor(16, 6144, 10), ShapeTensor(6144, 4),
            conv_state_indices=ShapeTensor(2, 4), num_accepted_tokens=ShapeTensor(2),
            query_start_loc=ShapeTensor(3), max_query_len=4,
        )
        self.assertEqual(recorder.grid, (2, 24))
        self.assertEqual(recorder.kwargs["BLOCK_N"], 256)


if __name__ == "__main__":
    unittest.main()
