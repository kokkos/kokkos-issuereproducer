Issue: https://github.com/kokkos/kokkos/issues/4224
Reporter: pvelesko and rgayatri
Compiler version - llvm/13

# Build APP
The reproducer is dependent on the develop branch of kokkos-source code
https://github.com/kokkos/kokkos
The reproducer was tried last with the following commit: 8d0f40256e2b25209afc3b8204adcd635d48401c
The reproducer fails with the OpenMPTarget backend of Kokkos on NVIDIA V100 GPU.

## Makefile
Edit the path to `KOKKOS_PATH` in the Makefile.

# Error generated

```console
CUDA error: misaligned address
Libomptarget error: Call to targetDataEnd failed, abort target.
Libomptarget error: Failed to process data after launching the kernel.
Libomptarget error: Run with LIBOMPTARGET_INFO=4 to dump host-target pointer mappings.
Kokkos_OpenMPTarget_Parallel.hpp:85:1: Libomptarget fatal error 1: failure of target construct while offloading is mandatory
Aborted
```

The error arises with llvm/13 but works with llvm/12.
Using `-fopenmp-cuda-mode` avoids the error.
