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
    return selected


def _select_mtp(models, torch):
    selected, seen = [], set()

    def add(name, t):
        if isinstance(t, torch.Tensor) and t.is_cuda and t.numel() and id(t) not in seen:
            seen.add(id(t))
            selected.append((name, t))

    for owner, model in zip(("target", "drafter"), models):
        if model is None:
            continue
        for name, layer in model.named_modules():
            prefix = owner + "." + name
            # Preserve SM addresses: packed-matrix admission keys name that
            # first plane. Other planes keep the same objects and exact bytes.
            for field, tensor in layer.named_buffers(recurse=False):
                if (field.startswith(("_mtp_pk_", "_mtp_eh_pk_"))
                        and field.rsplit("_", 1)[-1] in ("ex", "base", "ri", "raw")):
                    add(prefix + "." + field, tensor)
            weights = getattr(layer, "_fast_moe_weights", None)
            if isinstance(weights, tuple):
                for slot in (0, 2):
                    if slot < len(weights):
                        packed = weights[slot]
                        planes = getattr(packed, "planes", ())
                        if type(packed).__name__ == "PackedBF16" and len(planes) == 5:
                            for i in (1, 2, 3, 4):
                                add(prefix + f".packed{slot}.plane{i}", planes[i])
    return selected


def prepare(models):
    """One load-time transaction; every fallback precedes any pointer change."""
    global _PREPARED
    if _PREPARED:
        return
    _PREPARED = True
    if os.environ.get("CMP2", "1") == "0":
        _LOG.info("CMP2 compressed=none reason=disabled")
        return
    import torch

    try:
        selected = _select(models, torch)
    except Exception as exc:
        _LOG.warning("CMP2 original selection declined: %r", exc)
        return
    _relocate(selected, "model")
    if os.environ.get("CMP2_MTP_PLANES", "1") != "0":
        # An added-family failure cannot roll back the parent's successful
        # model-family transaction. Bound extra physical backing independently.
        try:
            selected_mtp = _select_mtp(models, torch)
        except Exception as exc:
            _LOG.warning("CMP2 mtp-metadata ordinary: %r", exc)
        else:
            _relocate(selected_mtp, "mtp-metadata", 512 * (1 << 20))


def prepare_screen(screen):
    """Checkpoint-only screen matrices, before its fold check/graph capture."""
    if getattr(screen, "_cmp_prepared", False):
        return
    screen._cmp_prepared = True
    if os.environ.get("CMP2", "1") == "0" or os.environ.get("CMP2_MTP_SCREEN", "1") == "0":
        return
    import torch

    # Positive immutable matrix list; never scores, candidate IDs, inputs,
    # outputs, gates or any other request-written operational buffer.
    names = ("w_full", "s_full", "w_scr", "s_scr", "w_coarse", "s_coarse",
             "w_rest", "s_rest", "w_pre", "s_pre", "w_mid", "s_mid")
    selected = [("mtp_screen." + n, getattr(screen, n)) for n in names
                if isinstance(getattr(screen, n, None), torch.Tensor)
                and getattr(screen, n).is_cuda and getattr(screen, n).numel()]
    _relocate(selected, "mtp-screen", 512 * (1 << 20))


def _relocate(selected, label, budget_bytes=None):
    """Original full-storage transaction, shared by independently bound owners."""
    import torch

    changes, originals, compressed = [], [], []
    labels = []
    checked_bytes = 0
    allocated_bytes = 0
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
        for name, tensor in selected:
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
            allocated_bytes += allocated.value
            if budget_bytes is not None and allocated_bytes > budget_bytes:
                raise RuntimeError("additional compressed backing budget exceeded")
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
    except Exception as exc:
        _LOG.warning("CMP2 owner=%s compressed=none reason=%r ordinary_tensors_retained=True", label, exc)
        return
    _LOG.info(
        "CMP2 owner=%s compressed=[%s] storages=%d full_byte_checks=%d checked_bytes=%d "
        "original_storages_retained=True",
        label, ",".join(labels), len(compressed), len(compressed), checked_bytes,
    )
