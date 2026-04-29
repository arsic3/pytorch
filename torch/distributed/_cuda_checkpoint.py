"""
Self-checkpoint/restore for the current process's CUDA context.

Wraps cuCheckpointProcess{Lock,Checkpoint,Restore,Unlock} via ctypes so a
process can release its entire CUDA context (VRAM returned to the driver)
and later restore it at the same virtual addresses.

Usage::

    from torch.distributed._cuda_checkpoint import checkpoint_self, restore_self

    torch.cuda.synchronize()
    checkpoint_self()      # VRAM freed, CUDA ops will fault
    # ... wait for scheduling ...
    restore_self()         # VRAM restored, CUDA ops resume
"""

import ctypes
import logging
import os
import threading

log = logging.getLogger(__name__)

_cuda = None
_cuda_lock = threading.Lock()


def _get_cuda():
    global _cuda
    if _cuda is not None:
        return _cuda
    with _cuda_lock:
        if _cuda is not None:
            return _cuda
        try:
            lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise RuntimeError(
                "torchmux requires the CUDA driver library (libcuda.so.1). "
                "Ensure CUDA drivers are installed and libcuda.so.1 is on "
                "the library search path."
            ) from e

        for name in (
            "cuCheckpointProcessLock",
            "cuCheckpointProcessCheckpoint",
            "cuCheckpointProcessRestore",
            "cuCheckpointProcessUnlock",
            "cuCheckpointProcessGetState",
        ):
            fn = getattr(lib, name)
            fn.restype = ctypes.c_int
            if name == "cuCheckpointProcessGetState":
                fn.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
            else:
                fn.argtypes = [ctypes.c_uint, ctypes.c_void_p]

        _cuda = lib
    return _cuda


class CudaCheckpointError(RuntimeError):
    pass


def _call(fn_name: str) -> None:
    pid = os.getpid()
    fn = getattr(_get_cuda(), fn_name)
    rc = fn(pid, None)
    if rc != 0:
        raise CudaCheckpointError(
            f"{fn_name}(pid={pid}) failed with CUresult={rc}"
        )


def checkpoint_self() -> None:
    """Lock + checkpoint the current process. Releases VRAM."""
    _call("cuCheckpointProcessLock")
    _call("cuCheckpointProcessCheckpoint")


def restore_self() -> None:
    """Restore + unlock the current process. Re-allocates VRAM."""
    _call("cuCheckpointProcessRestore")
    _call("cuCheckpointProcessUnlock")
