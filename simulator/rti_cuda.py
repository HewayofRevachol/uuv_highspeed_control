"""CuSADi CUDA dynamics and cuDSS batched sparse Cholesky for RTI QPs.

CuSADi is used only at build time.  The generated kernel is launched on the
current PyTorch stream, so its outputs and cuDSS consume the same stream.
"""

import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from simulator.rti_symbolic import NX, NU, discrete_function


class CudaCasadiDynamics:
    def __init__(self, capacity, dt=.1, cusadi_root=None, build_dir=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CuSADi dynamics require CUDA")
        root = Path(cusadi_root or os.getenv("CUSADI_ROOT", ".third_party/cusadi")).resolve()
        if not (root/"src"/"generateCUDACode.py").is_file():
            raise RuntimeError(f"CuSADi source missing at {root}; set CUSADI_ROOT")
        output = Path(build_dir or os.getenv("UUV_RTI_BUILD", ".rti_build")).resolve()
        output.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(root))
        try:
            from src.generateCUDACode import generateCUDACodeDouble
        finally:
            sys.path.remove(str(root))
        self.function = discrete_function(dt)
        digest = hashlib.sha256(self.function.serialize().encode()).hexdigest()[:12]
        stem = f"uuv_fd_jac_{digest}"
        source, library = output/f"{stem}.cu", output/f"lib{stem}.so"
        if not library.exists():
            generateCUDACodeDouble(self.function, str(source), benchmarking=False,
                                    debug_mode=False)
            content = source.read_text()
            marker = "float evaluate(const double *inputs[]"
            if marker not in content:
                raise RuntimeError("CuSADi generated an unexpected C interface")
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
            subprocess.run([compiler, "-O3", "-std=c++11", "-shared",
                            "-Xcompiler", "-fPIC", "-o", str(library), str(source)],
                           check=True)
        self.lib = ct.CDLL(str(library))
        self.lib.evaluate.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p,
                                      ct.c_int, ct.c_void_p]
        self.lib.evaluate.restype = None
        self.capacity = capacity
        self.device = torch.device("cuda")
        self.work = torch.empty(capacity, self.function.sz_w(), device="cuda",
                                dtype=torch.float64)
        self.output = [torch.empty(capacity, self.function.nnz_out(i),
                                   device="cuda", dtype=torch.float64)
                       for i in range(self.function.n_out())]
        self.input_ptrs = torch.empty(2, device="cuda", dtype=torch.int64)
        self.output_ptrs = torch.tensor([o.data_ptr() for o in self.output],
                                        device="cuda", dtype=torch.int64)
        self.indices = []
        for i in range(self.function.n_out()):
            rows, cols = self.function.sparsity_out(i).get_triplet()[:2]
            self.indices.append((torch.tensor(rows, device="cuda", dtype=torch.long),
                                 torch.tensor(cols, device="cuda", dtype=torch.long)))

    def evaluate(self, states, controls):
        if states.shape[:-1] != controls.shape[:-1] or states.shape[-1] != NX or controls.shape[-1] != NU:
            raise ValueError("invalid CuSADi input shapes")
        shape = states.shape[:-1]
        count = states.numel()//NX
        if count > self.capacity:
            raise ValueError("CuSADi batch exceeds compiled capacity")
        x = states.reshape(count, NX).contiguous()
        u = controls.reshape(count, NU).contiguous()
        self.input_ptrs.copy_(torch.tensor([x.data_ptr(), u.data_ptr()],
                                           device="cuda", dtype=torch.int64))
        stream = torch.cuda.current_stream().cuda_stream
        self.lib.evaluate(ct.c_void_p(self.input_ptrs.data_ptr()),
                          ct.c_void_p(self.work.data_ptr()),
                          ct.c_void_p(self.output_ptrs.data_ptr()), count,
                          ct.c_void_p(stream))
        dense = []
        for i in range(3):
            rows, cols = self.indices[i]
            matrix = torch.zeros(count, self.function.size1_out(i),
                                 self.function.size2_out(i), device="cuda",
                                 dtype=torch.float64)
            matrix[:, rows, cols] = self.output[i][:count]
            dense.append(matrix)
        return (dense[0].reshape(*shape, NX),
                dense[1].reshape(*shape, NX, NX),
                dense[2].reshape(*shape, NX, NU))


def _configure(lib, name, args):
    func = getattr(lib, name)
    func.argtypes = args
    func.restype = ct.c_int
    return func


class CudssBoxQP:
    """Box QP ADMM with one cuDSS factorization per SQP step.

    The SPD matrix P+rho A' A+sigma I is stored as upper CSR.  Its sparsity
    and symbolic analysis are reused across control steps and all batch items.
    """

    def __init__(self, variables, constraints, batch_size, iterations=80,
                 tolerance=2e-3, rho=2., sigma=1e-6, equalities=0,
                 matrix_type=3, pattern=None):
        if not torch.cuda.is_available():
            raise RuntimeError("cuDSS solver requires CUDA")
        self.n, self.m, self.batch_size = variables, constraints, batch_size
        self.iterations, self.tolerance = iterations, tolerance
        self.rho, self.sigma = rho, sigma
        self.rho_vector = torch.full((constraints,), rho, device="cuda",
                                      dtype=torch.float64)
        self.rho_vector[:equalities] = 100.
        try:
            from importlib.metadata import distribution
            path = distribution("nvidia-cudss-cu13").locate_file(
                "nvidia/cu13/lib/libcudss.so.0")
        except Exception as error:
            raise RuntimeError("install nvidia-cudss-cu13 in the CUDA interpreter") from error
        self.lib = ct.CDLL(str(path))
        V, I, L = ct.c_void_p, ct.c_int, ct.c_int64
        self.create = _configure(self.lib, "cudssCreate", [ct.POINTER(V)])
        self.config_create = _configure(self.lib, "cudssConfigCreate", [ct.POINTER(V)])
        self.data_create = _configure(self.lib, "cudssDataCreate", [V, ct.POINTER(V)])
        self.config_set = _configure(self.lib, "cudssConfigSet", [V, I, V, ct.c_size_t])
        self.set_stream = _configure(self.lib, "cudssSetStream", [V, V])
        self.make_csr = _configure(self.lib, "cudssMatrixCreateCsr",
                                    [ct.POINTER(V), L, L, L, V, V, V, V,
                                     I, I, I, I, I, I])
        self.make_dn = _configure(self.lib, "cudssMatrixCreateDn",
                                   [ct.POINTER(V), L, L, L, V, I, I])
        self.execute = _configure(self.lib, "cudssExecute", [V, I, V, V, V, V, V])
        self.destroy_matrix = _configure(self.lib, "cudssMatrixDestroy", [V])
        self.destroy_data = _configure(self.lib, "cudssDataDestroy", [V, V])
        self.destroy_config = _configure(self.lib, "cudssConfigDestroy", [V])
        self.destroy_handle = _configure(self.lib, "cudssDestroy", [V])
        self.handle, self.config, self.data = V(), V(), V()
        self._check(self.create(ct.byref(self.handle)), "create")
        self._check(self.config_create(ct.byref(self.config)), "config create")
        uniform = I(batch_size)
        self._check(self.config_set(self.config, 18, ct.cast(ct.byref(uniform), V),
                                    ct.sizeof(uniform)), "uniform batch")
        self._check(self.data_create(self.handle, ct.byref(self.data)), "data create")
        if pattern is None:
            row, col = torch.triu_indices(variables, variables, device="cuda")
        else:
            r, c = np.nonzero(np.triu(pattern))
            row = torch.as_tensor(r, device="cuda", dtype=torch.long)
            col = torch.as_tensor(c, device="cuda", dtype=torch.long)
        self.row, self.col = row, col
        counts = torch.bincount(row, minlength=variables).to(torch.int32)
        self.rowptr = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32),
                                 counts.cumsum(0).to(torch.int32)))
        self.values = torch.empty(batch_size, col.numel(), device="cuda", dtype=torch.float64)
        self.col32 = self.col.to(torch.int32)
        self.rhs = torch.empty(batch_size, variables, device="cuda", dtype=torch.float64)
        self.solution = torch.empty_like(self.rhs)
        self.matrix, self.b_matrix, self.x_matrix = V(), V(), V()
        self._check(self.make_csr(ct.byref(self.matrix), variables, variables,
                                  col.numel(), V(self.rowptr.data_ptr()),
                                  V(), V(self.col32.data_ptr()),
                                  V(self.values.data_ptr()), 10, 10, 1,
                                  matrix_type, 2, 0),
                    "CSR create")
        self._check(self.make_dn(ct.byref(self.b_matrix), variables, 1, variables,
                                 V(self.rhs.data_ptr()), 1, 0), "RHS create")
        self._check(self.make_dn(ct.byref(self.x_matrix), variables, 1, variables,
                                 V(self.solution.data_ptr()), 1, 0), "solution create")
        self._stream()
        self._check(self.execute(self.handle, 3, self.config, self.data,
                                 self.matrix, self.x_matrix, self.b_matrix),
                    "analysis")

    def _check(self, code, operation):
        if code:
            raise RuntimeError(f"cuDSS {operation} failed (status {code})")

    def _stream(self):
        self._check(self.set_stream(self.handle,
                                    ct.c_void_p(torch.cuda.current_stream().cuda_stream)),
                    "set stream")

    def solve(self, P, q, A, lower, upper):
        self._stream()
        matrix = P+A.transpose(-1, -2)@(self.rho_vector[:, None]*A)
        matrix.diagonal(dim1=-2, dim2=-1).add_(self.sigma)
        self.values.copy_(matrix[:, self.row, self.col])
        self._check(self.execute(self.handle, 4, self.config, self.data,
                                 self.matrix, self.x_matrix, self.b_matrix),
                    "factorization")
        x = torch.zeros_like(q)
        z = torch.minimum(torch.maximum(torch.zeros_like(lower), lower), upper)
        y = torch.zeros_like(lower)
        at = A.transpose(-1, -2)
        for _ in range(self.iterations):
            self.rhs.copy_(-q+self.sigma*x+(at @ (self.rho_vector*z-y)[..., None]).squeeze(-1))
            self._check(self.execute(self.handle, 1008, self.config, self.data,
                                     self.matrix, self.x_matrix, self.b_matrix),
                        "solve")
            x = self.solution.clone()
            ax = (A@x[..., None]).squeeze(-1)
            old_z = z
            z = torch.minimum(torch.maximum(ax+y/self.rho_vector, lower), upper)
            y = y+self.rho_vector*(ax-z)
        primal = (ax-z).abs().amax(-1)
        dual = (at@((self.rho_vector*(z-old_z))[..., None])).squeeze(-1).abs().amax(-1)
        self.last_primal, self.last_dual = primal, dual
        valid = torch.isfinite(x).all(-1) & (primal < self.tolerance) & (dual < self.tolerance)
        return x, valid

    def close(self):
        for matrix in (self.matrix, self.b_matrix, self.x_matrix):
            if matrix:
                self.destroy_matrix(matrix)
        if self.data:
            self.destroy_data(self.handle, self.data)
        if self.config:
            self.destroy_config(self.config)
        if self.handle:
            self.destroy_handle(self.handle)
        self.matrix = self.b_matrix = self.x_matrix = None
        self.data = self.config = self.handle = None


class CudssEqualityBoxQP(CudssBoxQP):
    """RTI QP with dynamics equalities solved exactly in an indefinite KKT."""

    def __init__(self, variables, constraints, equalities, batch_size,
                 iterations=200, tolerance=2e-3, rho=2., sigma=1e-6):
        self.variables, self.equalities = variables, equalities
        if equalities % NX or variables != equalities + (equalities//NX)*NU:
            raise ValueError("invalid multiple-shooting KKT dimensions")
        horizon, total = equalities//NX, variables+equalities
        pattern = np.eye(total, dtype=bool)
        for k in range(horizon):
            sx = slice(k*NX, (k+1)*NX)
            su = slice(equalities+k*NU, equalities+(k+1)*NU)
            erow = slice(variables+k*NX, variables+(k+1)*NX)
            pattern[sx, sx] = True
            pattern[su, su] = True
            pattern[sx, erow] = True
            pattern[su, erow] = True
            if k:
                prev_x = slice((k-1)*NX, k*NX)
                prev_u = slice(equalities+(k-1)*NU, equalities+k*NU)
                pattern[prev_x, erow] = True
                pattern[prev_u, su] = True
        super().__init__(variables+equalities, constraints, batch_size,
                         iterations=iterations, tolerance=tolerance, rho=rho,
                         sigma=sigma, matrix_type=1, pattern=pattern)

    def solve(self, P, q, A, lower, upper):
        self._stream()
        n, e = self.variables, self.equalities
        equality = A[:, :e, :]
        box = A[:, e:, :]
        lower_box, upper_box = lower[:, e:], upper[:, e:]
        kkt = torch.zeros(self.batch_size, n+e, n+e, device="cuda",
                          dtype=torch.float64)
        kkt[:, :n, :n] = P+self.rho*(box.transpose(-1, -2)@box)
        kkt[:, :n, :n].diagonal(dim1=-2, dim2=-1).add_(self.sigma)
        kkt[:, :n, n:] = equality.transpose(-1, -2)
        kkt[:, n:, :n] = equality
        self.values.copy_(kkt[:, self.row, self.col])
        self._check(self.execute(self.handle, 4, self.config, self.data,
                                 self.matrix, self.x_matrix, self.b_matrix),
                    "KKT factorization")
        x = torch.zeros_like(q)
        z = torch.minimum(torch.maximum(torch.zeros_like(lower_box),
                                         lower_box), upper_box)
        y = torch.zeros_like(z)
        bt = box.transpose(-1, -2)
        for _ in range(self.iterations):
            self.rhs[:, :n] = -q+self.sigma*x+(bt@(self.rho*z-y)[..., None]).squeeze(-1)
            self.rhs[:, n:] = lower[:, :e]
            self._check(self.execute(self.handle, 1008, self.config, self.data,
                                     self.matrix, self.x_matrix, self.b_matrix),
                        "KKT solve")
            x = self.solution[:, :n].clone()
            bx = (box@x[..., None]).squeeze(-1)
            old_z = z
            z = torch.minimum(torch.maximum(bx+y/self.rho, lower_box), upper_box)
            y = y+self.rho*(bx-z)
        equality_residual = ((equality@x[..., None]).squeeze(-1)-lower[:, :e]).abs().amax(-1)
        primal = (bx-z).abs().amax(-1)
        dual = (self.rho*(bt@(z-old_z)[..., None]).squeeze(-1)).abs().amax(-1)
        self.last_primal = torch.maximum(equality_residual, primal)
        self.last_dual = dual
        valid = (torch.isfinite(x).all(-1) &
                 (self.last_primal < self.tolerance) & (dual < self.tolerance))
        return x, valid
