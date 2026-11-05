#!/bin/bash
# Body of a reservation job (reserve_<machine>_<idx>, bin/reservenode). Makes its GPU look occupied
# outside SLURM (nvidia-smi, ffg): opens a CUDA context on it with a small allocation, and holds
# them until killed (cancelled or preempted). Its GPU is CUDA device 0 (task_prolog sets
# CUDA_VISIBLE_DEVICES). Uses the driver's libcuda directly, so it needs only the system python.
exec /usr/bin/python3 -I -c '
import ctypes, signal, sys
MIB = 16
cuda = ctypes.CDLL("libcuda.so.1")
def check(rc, call):
    if rc:
        sys.exit(f"reserve.sh: {call} failed: CUDA error {rc}")
ctx, dev, ptr = ctypes.c_void_p(), ctypes.c_int(), ctypes.c_uint64()
check(cuda.cuInit(0), "cuInit")
check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(MIB << 20)), "cuMemAlloc")
signal.pause()
'
