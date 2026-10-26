"""Input-independent transparent VMM weight storage, before engine capture.

CMP2=0 keeps the ordinary tensors. No codec, format, kernel or arithmetic
change. Each storage family is fully byte-checked before any tensor in that family is
rebound; an extension failure leaves the measured record family active.
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
    """Frozen families split into the record arm and the routed-pair extension."""
    base = []
    routed_pair = []
    seen = set()

    def add(bucket, name, tensor):
        if (isinstance(tensor, torch.Tensor) and tensor.is_cuda
                and tensor.numel() and id(tensor) not in seen):
            seen.add(id(tensor))
            bucket.append((name, tensor))

    for owner, model in zip(("target", "drafter"), models):
        if model is None:
            continue
        for name, layer in model.named_modules():
            prefix = owner + "." + name
            # Both module buffers and FP8 kernel objects refer to these SAME
            # Tensor objects. Preserve their identity, shape, stride and offset.
            add(base, prefix + "._fp8c_weight", getattr(layer, "_fp8c_weight", None))
            # The attention registry and projection hold the same plane objects.
            for i, plane in enumerate(getattr(layer, "_apk_planes", ())):
                add(base, prefix + f"._apk_planes.{i}", plane)
            method = type(getattr(layer, "quant_method", None)).__name__
            if "NvFp4" in method and ("shared_experts" in name or name == "lm_head"):
                zs = getattr(layer, "_marlinx_zs", None)
                add(base, prefix + "._marlinx_zs", zs)
                su = getattr(layer, "_marlinx_su", None)
                add(base, prefix + "._marlinx_su", su)
                for field, tensor in layer.named_parameters(recurse=False):
                    # Decode reads the existing zero-skip repack. The ordinary
                    # fallback matrix remains ordinary for its prefill path.
                    if field == "weight" and zs is not None:
                        continue
                    if field == "weight_scale" and su is not None:
                        continue
                    if any(tag in field.lower() for tag in ("workspace", "scratch", "lock")):
                        continue
                    add(base, prefix + "." + field, tensor)

            # Own extension: the target M=5 routed-down kernel reads the swap2
            # pair, not Humming's ordinary store nor the legacy zskip repack.
            # Select only its immutable FP4 codes and FP8 scales. Keeping this a
            # separate transaction means an allocation failure cannot disable
            # the already measured record arm above.
            pair = getattr(layer, "zskip_pair_w2", None)
            if owner == "target" and getattr(pair, "layout", None) == "swap2":
                add(routed_pair, prefix + ".zskip_pair_w2.w", getattr(pair, "w", None))
                add(routed_pair, prefix + ".zskip_pair_w2.s", getattr(pair, "s", None))
    return base, routed_pair


def _compress(selected, torch, family, reserve_bytes=0):
    """Atomically rebind one family; a failed extension leaves the base active."""
    if not selected:
        return True
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
        for name, tensor in selected:
            storage = tensor.untyped_storage()
            key = (tensor.device.index, storage.data_ptr(), storage.nbytes())
            storages.setdefault(key, []).append((name, tensor))
        requested = sum(key[2] for key in storages)
        if reserve_bytes:
            device = next(iter(storages.values()))[0][1].device
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if free_bytes < requested + reserve_bytes:
                _LOG.warning(
                    "CMP2 family=%s compressed=none reason=memory_reserve "
                    "requested_bytes=%d free_bytes=%d reserve_bytes=%d "
                    "ordinary_tensors_retained=True",
                    family, requested, free_bytes, reserve_bytes,
                )
                return False
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
    except Exception as exc:
        _LOG.warning(
            "CMP2 family=%s compressed=none reason=%r ordinary_tensors_retained=True",
            family, exc,
        )
        return False
    _LOG.info(
        "CMP2 family=%s compressed=[%s] storages=%d full_byte_checks=%d "
        "checked_bytes=%d original_storages_retained=True",
        family, ",".join(labels), len(compressed), len(compressed), checked_bytes,
    )
    return True


def prepare(models):
    """Run the measured base transaction, then the optional routed pair."""
    global _PREPARED
    if _PREPARED:
        return
    _PREPARED = True
    if os.environ.get("CMP2", "1") == "0":
        _LOG.info("CMP2 compressed=none reason=disabled")
        return
    import torch

    base, routed_pair = _select(models, torch)
    if not _compress(base, torch, "record"):
        return
    # Worst-case (incompressible) backing must leave room for graph capture,
    # scratch and allocator fragmentation. A skipped extension is the record.
    _compress(routed_pair, torch, "routed_pair", reserve_bytes=8 << 30)
