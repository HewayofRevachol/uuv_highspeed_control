"""Optional CuSADi kernel for the parameterized 22-state GS-12 predictor."""
from __future__ import annotations

import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import torch

from simulator.rti_symbolic import actuator_aware_function


class CuSADi22:
    def __init__(self, capacity, device="cuda", root=None, build_dir=None):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CuSADi22 requires CUDA")
        root = Path(root or os.getenv("CUSADI_ROOT", ".third_party/cusadi")).resolve()
        if not (root/"src"/"generateCUDACode.py").is_file():
            raise RuntimeError(f"CuSADi source missing at {root}; set CUSADI_ROOT")
        sys.path.insert(0, str(root))
        try:
            from src.generateCUDACode import generateCUDACodeDouble
            from src.CusadiOperations import OP_CUDA_DICT
            from casadi import OP_EXPM1
            # Pinned CuSADi predates CasADi's expm1 opcode, used by the
            # actuator recurrence. Extend only this process's code generator.
            OP_CUDA_DICT[OP_EXPM1] = "\n        work[env_idx + %d] = expm1(work[env_idx + %d]);"
        finally:
            sys.path.remove(str(root))
        self.function = actuator_aware_function(.1)
        digest = hashlib.sha256(self.function.serialize().encode()).hexdigest()[:16]
        output = Path(build_dir or os.getenv("UUV_RTI_BUILD", ".rti_build")).resolve()
        output.mkdir(parents=True, exist_ok=True)
        source = output/f"gs12_aug_{digest}.cu"
        library = output/f"libgs12_aug_{digest}.so"
        if not library.exists():
            generateCUDACodeDouble(self.function, str(source), benchmarking=False,
                                    debug_mode=False)
            content = source.read_text()
            marker = "float evaluate(const double *inputs[]"
            if marker not in content:
                raise RuntimeError("unexpected CuSADi generated interface")
            content = content.split(marker)[0] + """void evaluate(const double *inputs[],
                    double *work, double *outputs[], int batch_size,
                    cudaStream_t stream) {
                int blocks = (batch_size + 255) / 256;
                evaluate_kernel<<<blocks, 256, 0, stream>>>(inputs, work, outputs,
                                                             batch_size);
            }
            }\n"""
            source.write_text(content)
            compiler = os.getenv("NVCC", "/usr/local/cuda/bin/nvcc")
            subprocess.run([compiler, "-O3", "-std=c++11", "-shared", "-Xcompiler",
                            "-fPIC", "-o", str(library), str(source)], check=True)
        self.lib = ct.CDLL(str(library))
        self.lib.evaluate.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                      ct.c_int, ct.c_void_p]
        self.lib.evaluate.restype = None
        self.capacity = capacity
        self.inputs = [torch.empty(capacity, 22, device=self.device, dtype=torch.float64),
                       torch.empty(capacity, 3, device=self.device, dtype=torch.float64),
                       torch.empty(capacity, 2, device=self.device, dtype=torch.float64),
                       torch.zeros(capacity, 5, device=self.device, dtype=torch.float64)]
        self.outputs = [torch.empty(capacity, self.function.nnz_out(i),
                                    device=self.device, dtype=torch.float64)
                        for i in range(3)]
        self.work = torch.empty(capacity, self.function.sz_w(), device=self.device,
                                dtype=torch.float64)
        self.input_ptrs = torch.tensor([v.data_ptr() for v in self.inputs],
                                       device=self.device, dtype=torch.int64)
        self.output_ptrs = torch.tensor([v.data_ptr() for v in self.outputs],
                                        device=self.device, dtype=torch.int64)
        self.indices = []
        for i in range(3):
            rows, cols = self.function.sparsity_out(i).get_triplet()[:2]
            self.indices.append((torch.tensor(rows, device=self.device, dtype=torch.long),
                                 torch.tensor(cols, device=self.device, dtype=torch.long)))

    def evaluate(self, state, command, beta):
        count = state.shape[0]
        if count > self.capacity or state.shape != (count, 22) or command.shape != (count, 3):
            raise ValueError("invalid CuSADi predictor batch")
        self.inputs[0][:count].copy_(state)
        self.inputs[1][:count].copy_(command)
        self.inputs[2][:count, 0].copy_(1.-.15*beta)
        self.inputs[2][:count, 1].copy_(1.+beta)
        self.lib.evaluate(ct.c_void_p(self.input_ptrs.data_ptr()),
                          ct.c_void_p(self.work.data_ptr()),
                          ct.c_void_p(self.output_ptrs.data_ptr()), count,
                          ct.c_void_p(torch.cuda.current_stream(self.device).cuda_stream))
        dense = []
        for i, shape in enumerate(((22, 1), (22, 22), (22, 3))):
            rows, cols = self.indices[i]
            value = torch.zeros(count, *shape, device=self.device, dtype=torch.float64)
            value[:, rows, cols] = self.outputs[i][:count]
            dense.append(value)
        return dense[0].squeeze(-1), dense[1], dense[2]
