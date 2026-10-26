"""Input-independent transparent VMM weight storage, before engine capture.

CMP2=0 keeps the ordinary tensors. No codec, format, kernel or arithmetic
change. All selected storages are byte-checked before any tensor is rebound;
failed support, allocation or verification leaves every ordinary tensor intact.
Original storages stay alive for boot-time private graphs and any retained view.
"""
import ctypes
import logging
import os
from pathlib import Path

_LOG = logging.getLogger("vllm.cmp2")
_LIB = None
_COMPRESSED = []
_ORIGINALS = []
_PREPARED = False


def _library():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libcmp2.so")))
        lib.cmp_supported.restype = ctypes.c_int
        lib.cmp_last_error.restype = ctypes.c_char_p
        lib.cmp_allocate.argtypes = [ctypes.c_size_t, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_int),
                                    ctypes.POINTER(ctypes.c_size_t)]
        lib.cmp_allocate.restype = ctypes.c_void_p
        _LIB = lib
    return _LIB


def _select(models, torch):
    """Resolved positive frozen-weight families; no mutable cache or scratch."""
    selected = []
    seen = set()

    def add(name, tensor):
        if (isinstance(tensor, torch.Tensor) and tensor.is_cuda
                and tensor.numel() and id(tensor) not in seen):
            seen.add(id(tensor))
            selected.append((name, tensor))

    for owner, model in zip(("target", "drafter"), models):
        if model is None:
            continue
        for name, layer in model.named_modules():
            prefix = owner + "." + name
            # Both module buffers and FP8 kernel objects refer to these SAME
            # Tensor objects. Preserve their identity, shape, stride and offset.
            add(prefix + "._fp8c_weight", getattr(layer, "_fp8c_weight", None))
            # The attention registry and projection hold the same plane objects.
            for i, plane in enumerate(getattr(layer, "_apk_planes", ())):
                add(prefix + f"._apk_planes.{i}", plane)
            method = type(getattr(layer, "quant_method", None)).__name__
            if "NvFp4" in method and ("shared_experts" in name or name == "lm_head"):
                zs = getattr(layer, "_marlinx_zs", None)
                add(prefix + "._marlinx_zs", zs)
                su = getattr(layer, "_marlinx_su", None)
                add(prefix + "._marlinx_su", su)
                for field, tensor in layer.named_parameters(recurse=False):
                    # Decode reads the existing zero-skip repack. The ordinary
                    # fallback matrix remains ordinary for its prefill path.
                    if field == "weight" and zs is not None:
                        continue
                    if field == "weight_scale" and su is not None:
                        continue
                    if any(tag in field.lower() for tag in ("workspace", "scratch", "lock")):
                        continue
                    add(prefix + "." + field, tensor)
            if owner == "drafter":
                # Drafter packed BF16 planes read by every draft pass: the
                # attention projections' and the eh hidden half's registered
                # planes, and the shared-expert up planes; the routed-up packs
                # contribute only their small base/index/raw planes.
                for field in ("_mtp_pk_", "_mtp_eh_pk_"):
                    for part in ("sm", "ex", "base", "ri", "raw"):
                        add(prefix + "." + field + part, getattr(layer, field + part, None))
                fast = getattr(layer, "_fast_moe_weights", None)
                if isinstance(fast, tuple) and len(fast) == 6:
                    for idx, first in ((0, 2), (2, 0)):
                        for j, plane in enumerate(getattr(fast[idx], "planes", ())[first:]):
                            add(prefix + f"._fast_moe_weights.{idx}.planes.{first + j}", plane)
    if models and models[-1] is not None:
        # The drafter's packed down projections live in the down-pack registry,
        # keyed by the BF16 sources (which stay ordinary); move their planes:
        # the shared slice whole, the routed one's small planes.
        try:
            from vllm.model_executor.models import nemotron_h_mtp_down_packed as dp
            for key, entry in list(dp._PACKS.items()):
                if len(entry) < 4 or entry[2] is None or entry[3] is None:
                    continue
                for idx, first in ((2, 2), (3, 0)):
                    for j, plane in enumerate(getattr(entry[idx], "planes", ())[first:]):
                        add(f"drafter.down_pack.{idx}.planes.{first + j}", plane)
        except Exception:  # noqa: BLE001 - registry absent: nothing more selected
            pass
    return selected


def _rekey_packed_mm():
    """The packed-GEMV admission registry is keyed by its first plane's address;
    re-key every entry after the planes moved (same objects, same flags)."""
    try:
        from vllm.model_executor.models import nemotron_h_mtp_pkmm as pk
    except Exception:  # noqa: BLE001
        return
    entries = list(pk._ADMITTED.values())
    pk._ADMITTED.clear()
    for packed, flags in entries:
        pk._ADMITTED[pk._identity(packed)] = (packed, flags)


def prepare(models):
    """One load-time transaction; every fallback precedes any pointer change."""
    global _PREPARED
    if _PREPARED:
        return
    _PREPARED = True
    if os.environ.get("CMP2", "1") == "0":
        _LOG.info("CMP2 compressed=none reason=disabled")
        return
    if os.environ.get("VLLM_EXPANDABLE_SEGMENTS_ACTIVE") == "1":
        # The weights already sit in VMM-backed expandable segments.
        _LOG.info("CMP2 compressed=none reason=expandable_segments")
        return
    import torch

    changes, originals, compressed = [], [], []
    labels = []
    checked_bytes = 0
    try:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("load-time allocation requested during capture")
        lib = _library()
        if not lib.cmp_supported():
            raise RuntimeError(lib.cmp_last_error().decode())
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        capsule_new.restype = ctypes.py_object
        # Gather all aliases before rebinding even one of them.
        storages = {}
        for name, tensor in _select(models, torch):
            storage = tensor.untyped_storage()
            key = (tensor.device.index, storage.data_ptr(), storage.nbytes())
            storages.setdefault(key, []).append((name, tensor))
        for (_, _, nbytes), aliases in storages.items():
            tensor = aliases[0][1]
            source = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                tensor.untyped_storage(), 0, (nbytes,), (1,))
            host = source.cpu()
            actual, allocated = ctypes.c_int(-1), ctypes.c_size_t(0)
            ptr = lib.cmp_allocate(nbytes, 1, ctypes.byref(actual), ctypes.byref(allocated))
            if not ptr:
                raise RuntimeError(lib.cmp_last_error().decode())
            raw = torch.from_dlpack(capsule_new(ptr, b"dltensor", None))
            if actual.value != 1 or raw.data_ptr() % 2097152:
                raise RuntimeError("compression type or 2 MiB alignment check failed")
            raw.copy_(host)
            # Full storage comparison as bytes, including padding/NaN payloads.
            if not torch.equal(host, raw.cpu()):
                raise RuntimeError("full storage byte roundtrip failed")
            checked_bytes += nbytes
            originals.append(source)
            compressed.append(raw)
            for name, value in aliases:
                view = raw.view(value.dtype).as_strided(
                    value.shape, value.stride(), value.storage_offset())
                if (view.shape != value.shape or view.stride() != value.stride()
                        or view.storage_offset() != value.storage_offset()):
                    raise RuntimeError("tensor metadata check failed")
                changes.append((value, view, value.data))
                labels.append(name)
        torch.cuda.synchronize()
        # Host-only pointer rebinding. Retain all old storages, including any
        # aliases used by a private graph that packing/admission captured at boot.
        with torch.inference_mode():
            try:
                for value, view, _ in changes:
                    value.data = view
            except Exception:
                for value, _, old in changes:
                    value.data = old
                raise
        _ORIGINALS.extend(originals)
        _COMPRESSED.extend(compressed)
        _rekey_packed_mm()
    except Exception as exc:
        _LOG.warning("CMP2 compressed=none reason=%r ordinary_tensors_retained=True", exc)
        return
    _LOG.info(
        "CMP2 compressed=[%s] storages=%d full_byte_checks=%d checked_bytes=%d "
        "original_storages_retained=True",
        ",".join(labels), len(compressed), len(compressed), checked_bytes,
    )
