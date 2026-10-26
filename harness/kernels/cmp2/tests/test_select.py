"""Host-side selection contract for cmp2. No CUDA, no libcmp2.

The scoring path turns expandable segments on. Legacy families stay ordinary
then; only a target routed layer's w13_weight is eligible. w2_weight and the
block scales stay out so the zskip repack and the nvsc alias are not split.
"""
import importlib.util
import os
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "py" / "__init__.py"


def _load():
    spec = importlib.util.spec_from_file_location("cmp2_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Tensor:
    def __init__(self, n=4, cuda=True):
        self._n = n
        self.is_cuda = cuda

    def numel(self):
        return self._n


class _Torch:
    Tensor = _Tensor


class _Method:
    pass


class _NvFp4(_Method):
    pass


class _Layer:
    def __init__(self, method=None, **attrs):
        self.quant_method = method
        self._params = []
        for key, value in attrs.items():
            setattr(self, key, value)

    def named_parameters(self, recurse=False):
        return list(self._params)


class _Model:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        return list(self._modules)


def _names(selected):
    return [name for name, _ in selected]


def test_scoring_path_selects_only_target_w13():
    os.environ.pop("CMP2_ROUTED", None)
    mod = _load()
    w13, w2, scale = _Tensor(), _Tensor(), _Tensor()
    fp8 = _Tensor()
    shared_w = _Tensor()
    draft_w13, draft_w2 = _Tensor(), _Tensor()
    routed = _Layer(_NvFp4(), w13_weight=w13, w2_weight=w2, w13_weight_scale=scale)
    shared = _Layer(_NvFp4(), _fp8c_weight=fp8)
    shared._params = [("weight", shared_w), ("workspace", _Tensor())]
    draft = _Layer(_NvFp4(), w13_weight=draft_w13, w2_weight=draft_w2,
                    _mtp_pk_sm=_Tensor())
    head = _Layer(_NvFp4())
    head_w = _Tensor()
    head._params = [("weight", head_w)]
    target = _Model([
        ("layers.1.mlp.experts", routed),
        ("layers.1.mlp.shared_experts", shared),
        ("lm_head", head),
    ])
    drafter = _Model([("layers.0", draft)])
    selected = mod._select((target, drafter), _Torch(), legacy=False)
    assert _names(selected) == ["target.layers.1.mlp.experts.w13_weight"]
    assert selected[0][1] is w13


def test_legacy_keeps_families_and_still_skips_w2_and_scales():
    os.environ.pop("CMP2_ROUTED", None)
    mod = _load()
    w13, w2 = _Tensor(), _Tensor()
    scale = _Tensor()
    zs, su = _Tensor(), _Tensor()
    plain = _Tensor()
    routed = _Layer(_NvFp4(), w13_weight=w13, w2_weight=w2, w13_weight_scale=scale)
    shared = _Layer(_NvFp4(), _marlinx_zs=zs, _marlinx_su=su)
    shared._params = [("weight", plain), ("weight_scale", _Tensor()), ("scratch_pad", _Tensor())]
    head = _Layer(_NvFp4())
    head_w = _Tensor()
    head._params = [("weight", head_w)]
    plane = _Tensor()
    attn = _Layer(None, _apk_planes=(plane, _Tensor(n=0)))
    target = _Model([
        ("layers.0.self_attn", attn),
        ("layers.1.mlp.experts", routed),
        ("layers.1.mlp.shared_experts", shared),
        ("lm_head", head),
    ])
    selected = mod._select((target, None), _Torch(), legacy=True)
    names = _names(selected)
    assert "target.layers.1.mlp.experts.w13_weight" in names
    assert "target.layers.1.mlp.experts.w2_weight" not in names
    assert "target.layers.1.mlp.experts.w13_weight_scale" not in names
    assert "target.layers.0.self_attn._apk_planes.0" in names
    assert "target.layers.0.self_attn._apk_planes.1" not in names
    assert "target.layers.1.mlp.shared_experts._marlinx_zs" in names
    assert "target.layers.1.mlp.shared_experts._marlinx_su" in names
    assert "target.layers.1.mlp.shared_experts.weight" not in names
    assert "target.layers.1.mlp.shared_experts.weight_scale" not in names
    assert "target.lm_head.weight" in names


def test_routed_switch_off_selects_nothing_new():
    os.environ["CMP2_ROUTED"] = "0"
    try:
        mod = _load()
        routed = _Layer(_NvFp4(), w13_weight=_Tensor(), w2_weight=_Tensor())
        target = _Model([("layers.1.mlp.experts", routed)])
        assert mod._select((target, None), _Torch(), legacy=False) == []
        assert mod._select((target, None), _Torch(), legacy=True) == []
    finally:
        os.environ.pop("CMP2_ROUTED", None)


def test_missing_partner_weight_is_not_routed():
    os.environ.pop("CMP2_ROUTED", None)
    mod = _load()
    only_w13 = _Layer(_NvFp4(), w13_weight=_Tensor())
    target = _Model([("layers.1.mlp.experts", only_w13)])
    assert mod._select((target, None), _Torch(), legacy=False) == []


def test_prepare_early_return_matches_the_worker_flag():
    os.environ["CMP2_ROUTED"] = "0"
    os.environ["VLLM_EXPANDABLE_SEGMENTS_ACTIVE"] = "1"
    os.environ.pop("CMP2", None)
    try:
        mod = _load()

        def _boom():
            raise AssertionError("library loaded after the early return")

        mod._library = _boom
        mod.prepare((None, None))
        # A second call is a no-op even if the flag is later cleared.
        os.environ["CMP2_ROUTED"] = "1"
        mod.prepare((None, None))
    finally:
        os.environ.pop("CMP2_ROUTED", None)
        os.environ.pop("VLLM_EXPANDABLE_SEGMENTS_ACTIVE", None)


def test_prepare_disabled_switch_does_not_read_the_flag():
    os.environ["CMP2"] = "0"
    os.environ["VLLM_EXPANDABLE_SEGMENTS_ACTIVE"] = "1"
    os.environ["CMP2_ROUTED"] = "1"
    try:
        mod = _load()
        mod._library = lambda: (_ for _ in ()).throw(AssertionError("library loaded"))
        mod.prepare((None, None))
    finally:
        os.environ.pop("CMP2", None)
        os.environ.pop("VLLM_EXPANDABLE_SEGMENTS_ACTIVE", None)
        os.environ.pop("CMP2_ROUTED", None)


if __name__ == "__main__":
    test_scoring_path_selects_only_target_w13()
    test_legacy_keeps_families_and_still_skips_w2_and_scales()
    test_routed_switch_off_selects_nothing_new()
    test_missing_partner_weight_is_not_routed()
    test_prepare_early_return_matches_the_worker_flag()
    test_prepare_disabled_switch_does_not_read_the_flag()
    print("ok")
