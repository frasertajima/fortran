# Fortran vs Rust engine — known design differences

Status: **observations, not verdicts.** Recorded 2026-09-18 from
`benchmark_v5_1_RXT4060_fortran_vs_rust.ipynb` on the RTX 4060. The Fortran
engine remains the reference implementation; nothing here is a reason to change
that. Each item says what would settle it.

The notebook's premise — "both engines use cuBLAS/cuBLASLt for the GEMMs at the
same math modes, so they should land within run-to-run noise" — holds for about
half the rows. These are the places it does not.

## D1. `batched_vector` is not a like-for-like comparison

| | call | measured max_diff |
|---|---|---|
| Fortran `batched_vector_multiply` (`cuda_matlib.cuf:965`) | `cublasSgemmStridedBatched`, `n x 1 x n`, stride_a=0 broadcast | ~3e-8 (FP32) |
| Rust `rs_batched_vector_matmul` | `cublasGemmEx`, `COMPUTE_32F` + `GEMM_DEFAULT_TENSOR_OP` | ~1e-5 (TF32) |

Rust's 1.3-1.67x on this row is therefore partly bought with precision. Two
separate effects are tangled together and should be measured apart:

- **Formulation (real, keep):** Fortran issues a strided batch of `n x 1 x n`
  GEMVs; Rust does one `n x batch x n` GEMM via `Y^T = A^T V^T`. The single-GEMM
  framing is a genuine improvement and is independent of precision.
- **Math mode:** ~~see D3~~ — **withdrawn**, tested and refuted in v5.2. The
  handle setting is not why this path is FP32; the `n x 1 x n` GEMM shape is.
  See `V52_CHANGES.md`.

*To settle:* force both to the same math mode and re-run. Until then this row
supports no claim in either direction.

## D2. RESOLVED in v5.2 — `strided_batch_multiply` was not a strided-batch GEMM

The v5/v5.1 body looped over the batch: one `cublasGemmEx` + two device-wide
syncs per slice, so 8 batches cost 8 GEMMs and 16 syncs. Not a memory-footprint
design choice — review finding **P3**, which v5.1 fixed for
`tensor_4d_matmul`/`tensor_5d_matmul` but never applied here.

**Fixed 2026-09-18 (v5.2):** one chunked `cublasGemmStridedBatchedEx` per chunk.
~4x at (8,16^3), 2.99x at (8,64^3), decaying to parity by (8,1024^3) — the
expected shape for a fixed-cost fix (median of 3 runs; variance at the top two
shapes is +-10%, so single runs mislead there). Peak workspace is capped at
96 MiB independent of `batch_size`. Measurements, the workspace trade-off and the correctness matrix:
**`V52_CHANGES.md`**.

The Rust side is unchanged, so its large-size deficit — the F-order relayout
pass described above — remains open. The 3x Rust advantage at small sizes is
gone.

## D8. RESOLVED — Rust's F-order deficit was a discarded copy in the wrapper, not the kernel

`strided_batch` was the last row where Rust trailed Fortran, by 2.5-3.3x on the
F-order inputs the benchmark feeds. The obvious suspect was
`relayout_forder_f64`, the tile-transpose kernel. **It was not.**

The measurement that settled it: `rs.batched_matmul` on the *same* F-order data
took 11.3 ms at (8,2048^3) against 10.2 ms for C-order — so the relayout kernel
costs ~1 ms and is fine. Only `rs.strided_batch_matmul` was slow (36.5 ms).

Cause, in `rust_matrix_ops.py`:

```python
def strided_batch_matmul(self, m, k, n, batch_size, a, b):
    a_g, b_g = self._dev(a), self._dev(b)          # ascontiguousarray -> full FP64 copy
    if a_g.shape != (batch_size, m, k) or ...:     # ...used only for .shape
        raise ValueError(...)
    return self._batched(a, b, _BATCH_TF32)        # ...then thrown away
```

`_dev()` calls `ascontiguousarray`, which for an F-order input is a full FP64
relayout — 25.4 ms for the pair at (8,2048^3) — and the result was discarded,
because `_batched()` is handed the *original* arrays and does its own fused
in-kernel layout handling. A shape check needs no materialisation.

**Fixed:** check `.shape` directly. F-order `strided_batch`, GFLOPS:

| shape | Fortran | Rust before | Rust after | after vs Fortran |
|---|---|---|---|---|
| (8, 64^3) | 58.7 | ~35 | 51.6 | 0.88x |
| (8, 256^3) | 1603.9 | ~1393 | 1960.9 | **1.22x** |
| (8, 512^3) | 3550.4 | ~2204 | 4216.1 | **1.19x** |
| (8, 1024^3) | 7222.0 | ~2181 | 7420.5 | **1.03x** |
| (8, 2048^3) | 9616.2 | 3769 | **12174.3** | **1.27x** |

3.2x at (8,2048^3), and Rust now leads at every size from 256^3 up. Note this is
the same class of bug as D5 on the Fortran side — a wrapper doing FP64 relayout
work the kernel did not need — and in both engines it was worth more than any
kernel tuning.

*Measurement note:* an earlier draft quoted Rust at 13.5 TFLOPS here. That was
the C-order path: `cp.asfortranarray(x) / scalar` is not in-place and CuPy
returns C-order, so the benchmark script had silently switched layouts. The
notebook does `a /= ...` in place. Check `.flags.f_contiguous` after any scaling
step.

## D5. RESOLVED in v5.2 — the (8,256^3) gap was host-side relayout, not the kernel

After D2, Rust still led 1.82x at (8,256^3). Cause: `strided_batch_matmul`'s
wrapper performed three full FP64 relayouts (35-47% of total call time) because
the kernel demanded column-major slices. `batched_matmul` never paid this, and
neither does Rust — both fuse the relayout into their conversion pass.

Fixed by having the kernel consume **arbitrary element strides**, so no memory
order needs a host-side copy. 1.55-2.14x on top of D2 against the F-order inputs
the benchmark uses, and (8,2048^3) reaches 8015 GFLOPS vs v5.1's 5165.

Two plausible causes (64-bit index arithmetic, `GemmEx` vs `SgemmStridedBatched`)
were built and measured first, and both refuted. A first fix that merely flipped
the contract to row-major was **validated against the wrong memory order** and
turned out to cost up to 50% on F-order inputs; see `V52_CHANGES.md`.

## D3. REVERTED — TF32 bought no accuracy, only range, at 40% cost

**Original hypothesis (wrong):** that `CUBLAS_TENSOR_OP_MATH` at
`cuda_matlib.cuf:138` was a no-op leaving the `cublasSgemm*` paths on FP32.

**What it actually is:** the deprecated alias for `CUBLAS_COMPUTE_32F_FAST_16F`
— FP16 tensor cores. FP16 and TF32 both carry a 10-bit mantissa.

**Tried, measured, reverted.** Switching to `CUBLAS_TF32_TENSOR_OP_MATH`:

| | FP16 | TF32 |
|---|---|---|
| accuracy, `matmul` 512^3 | 8.61e-5 | 8.70e-5 |
| accuracy, `matrix_power` 512^3 | 9.99e-5 | 1.02e-4 |
| accuracy, `batched_matmul` 512^3 | 9.46e-5 | 9.52e-5 |
| speed, `matrix_power` 4048^3 | 23825 | 14239 (**-40%**) |
| speed, `matmul` 4048^3 | 14073 | 10288 (-27%) |

Identical accuracy (TF32 marginally worse), 27-40% slower. Reverted. The engine
is back on FP16 tensor cores, with D2 and D5 retained.

### The real hazard, still OPEN

FP16 tops out at 65504 and goes subnormal below ~6.1e-5. Operands outside that
window break, and the benchmark cannot see it because `create_inputs` scales
everything by `1/(sqrt(n)*1.1)`.

**The fix is not an FP64 correction.** This is a *range* problem, not a
precision problem, and power-of-two scaling is exact in floating point and
O(n^2). Measured on `batched_matmul` (4,256^3), scaling both operands so
max|X| ~ 2^14, then unscaling the result:

| input scale | raw FP16 | pow2-scaled |
|---|---|---|
| 1e5 | **inf** | 9.38e-5 |
| 1e8 | **inf** | 8.93e-5 |
| 1e-6 | 8.00e-3 | 9.63e-5 |
| 1e-10 | **1.00** (total loss) | 9.85e-5 |

It is also robust to wide dynamic range *within* a matrix, because relative
error is dominated by the largest terms of each dot product: spreading entries
over 10^2 -> 10^15 degrades scaled FP16 only from 1.88e-4 to 5.08e-4.

**Where it must live.** A naive CuPy guard (`cp.abs(A).max()`) costs 44-75% of
the GEMM it protects — unusable. The max-abs reduction has to be **fused into
the FP64->FP32 conversion pass the engine already runs**, which reads every
element anyway; the rescale then happens on the FP32 workspace (half the
traffic) and only when the guard trips, and the unscale folds into the existing
FP32->FP64 write-back. Cost in range: ~0. Cost out of range: one extra FP32
pass. That is the "expensive correction once in a while" shape, with a cheap
correction.

## D7. RESOLVED — FP16 range guard across the plain tensor-core tier

Same hazard as D6, worse failure mode: the plain tier has no FP32 term carrying
the result, so underflow is **finite, silent and total** (rel err 1.000 below
max|A| ~ 1e-7). Overflow is detectable after the fact; underflow is not, so the
guard must be eager.

Fixed with `convert_reduce`: one kernel doing the strided gather, the FP64->FP32
conversion and a fused shared-memory max-abs reduction, plus exact power-of-two
rescaling. `matrix_dot` and `tensor_matrix_multiply` re-guard between iterations
because magnitude compounds (max|A| = 300 gives max|A^2| ~ 4.6e7).

Cost 0-17%, smallest at large shapes — against 27-40% for D3, which would have
fixed the same hazard by switching to TF32. Three other implementations were
built and rejected first, each for a different measured reason. Details:
`V52_CHANGES.md` D7.

**The Rust engine had the identical bug — same threshold, same magnitudes, same
deprecated `TENSOR_OP_MATH`, and a source comment recording the same misreading
of the enum. Now fixed the same way** (fused max-abs + exact power-of-two
rescale), at -1.2% to -12.3%. See `V52_CHANGES.md` D7 and
`tensor_core_engine_rust/RESULTS.md`.

## D6. RESOLVED — tc_split returned NaN above FP16 range; fixed with a lazy range guard

The **high-accuracy** tier (rel ~1e-6) went non-finite exactly at the FP16
boundary, because its GEMMs inherit `CUBLAS_TENSOR_OP_MATH` (the
`COMPUTE_32F_FAST_16F` alias, not TF32 as its own comments claimed). Worse than
D3: this is the tier a caller picks *because* they want accuracy, and it failed
silently into NaN rather than degrading.

**Fixed 2026-09-18** in `matmul_tc_split` and `batched_matmul_tc_split`.
`improved_matmul32` was checked and is not exposed (finite at max|A| = 1.5e6).
Full rationale and measurements: `V52_CHANGES.md` D6.

The design is the one this lab uses elsewhere — cheap correct base, expensive
correction only when needed — with two refinements that fell out of measuring:

- **The correction is power-of-two scaling, not FP64.** This is a range problem,
  not a precision problem. Scaling by 2^e is exact in binary floating point and
  commutes with the Dekker split, so it perturbs no digit. No expensive tier is
  needed at all.
- **Detection is lazy, not up-front.** A pre-pass reduction over the inputs cost
  6-24% on the in-range path (worst at small n). Instead the non-finite count
  rides on the accumulate pass that already reads every output element; the
  reduction over the inputs happens only after a failure is seen. Fast-path cost
  is now below measurement noise.

## D3. REVERTED — TF32 bought no accuracy, only range, at 40% cost

**Original hypothesis (wrong):** that `CUBLAS_TENSOR_OP_MATH` at
`cuda_matlib.cuf:138` was a no-op leaving the `cublasSgemm*` paths on FP32.

**What it actually is:** the deprecated alias for `CUBLAS_COMPUTE_32F_FAST_16F`
— FP16 tensor cores. FP16 and TF32 both carry a 10-bit mantissa.

**Tried, measured, reverted.** Switching to `CUBLAS_TF32_TENSOR_OP_MATH`:

| | FP16 | TF32 |
|---|---|---|
| accuracy, `matmul` 512^3 | 8.61e-5 | 8.70e-5 |
| accuracy, `matrix_power` 512^3 | 9.99e-5 | 1.02e-4 |
| accuracy, `batched_matmul` 512^3 | 9.46e-5 | 9.52e-5 |
| speed, `matrix_power` 4048^3 | 23825 | 14239 (**-40%**) |
| speed, `matmul` 4048^3 | 14073 | 10288 (-27%) |

Identical accuracy (TF32 marginally worse), 27-40% slower. Reverted. The engine
is back on FP16 tensor cores, with D2 and D5 retained.

### The real hazard, still OPEN

FP16 tops out at 65504 and goes subnormal below ~6.1e-5. Operands outside that
window break, and the benchmark cannot see it because `create_inputs` scales
everything by `1/(sqrt(n)*1.1)`.

**The fix is not an FP64 correction.** This is a *range* problem, not a
precision problem, and power-of-two scaling is exact in floating point and
O(n^2). Measured on `batched_matmul` (4,256^3), scaling both operands so
max|X| ~ 2^14, then unscaling the result:

| input scale | raw FP16 | pow2-scaled |
|---|---|---|
| 1e5 | **inf** | 9.38e-5 |
| 1e8 | **inf** | 8.93e-5 |
| 1e-6 | 8.00e-3 | 9.63e-5 |
| 1e-10 | **1.00** (total loss) | 9.85e-5 |

It is also robust to wide dynamic range *within* a matrix, because relative
error is dominated by the largest terms of each dot product: spreading entries
over 10^2 -> 10^15 degrades scaled FP16 only from 1.88e-4 to 5.08e-4.

**Where it must live.** A naive CuPy guard (`cp.abs(A).max()`) costs 44-75% of
the GEMM it protects — unusable. The max-abs reduction has to be **fused into
the FP64->FP32 conversion pass the engine already runs**, which reads every
element anyway; the rescale then happens on the FP32 workspace (half the
traffic) and only when the guard trips, and the unscale folds into the existing
FP32->FP64 write-back. Cost in range: ~0. Cost out of range: one extra FP32
pass. That is the "expensive correction once in a while" shape, with a cheap
correction.

## D6. `batched_matmul_tc_split` returns NaN above FP16 range — OPEN, worse than D3

Found while testing D3's replacement. The **high-accuracy** tier (rel ~1e-6)
goes non-finite exactly at the FP16 boundary, because its
`cublasSgemmStridedBatched` calls (`cuda_matlib.cuf:1502,1513,1524`) inherit the
handle math mode:

| max abs operand | tc_split rel err |
|---|---|
| 1.500e+04 | 1.06e-06 |
| 9.000e+04 | **inf** |
| 1.500e+06 | **inf** |

Arguably more urgent than D3: this is the tier a caller picks *because* they
want accuracy, and it fails silently into NaN rather than degrading. The same
scaling guard fixes it, and `matmul_tc_split`/`improved_matmul32` should be
checked for the same exposure.

## D4. Rows that are measurement noise, not engine differences

- `matrix_multiply (512)` reads 2.21x, between 1.06 at (256) and 0.97 at (640).
  Fortran's own curve is the outlier: 723 -> **1737** -> 5478 GFLOPS at
  256/512/640. Rust's value sits on trend.
- `matrix_multiply (1024)` at 1.40x is probably the same.

`calculate_metrics` takes min-of-3, which is the right statistic, but
`warmup_gpu()` runs once per *operation* rather than per shape, and the Fortran
and Rust measurements live in separate cells executed minutes apart. Don't read
a single-shape deviation under ~1.3x as structural.

*To settle:* interleave both engines in one loop per shape, min of N.

## D5. Rows that are sound

- **`batched_matmul_fp64`, ratio ~1.00 at every size** — the useful control.
  Both plateau at 208-216 GFLOPS, ~90% of the 4060's ~236 GFLOPS FP64 peak
  (1/64 rate), with identical results and zero error. Both are at the hardware
  limit; there is nothing left to distinguish, which is the expected outcome
  and evidence the harness is sound.
- **GEMV rows (`vector_matrix`, `matrix_vector`, `vector_matrix_optimised`),
  Rust 0.82-0.98** — 24 of 24 points on one side is not noise. These are
  launch-latency bound (<60 GFLOPS); Rust pays slightly more fixed overhead per
  call. Confirm with an empty-kernel round-trip.
- **`matrix_power` 1.08-1.68x decaying to 0.98 at 4048** — wrapper overhead, as
  the notebook's cell 0 predicts (Rust skips `asfortranarray` copies and
  per-stage syncs). Decay toward 1.0 as compute dominates is the signature of a
  fixed-cost advantage, not a faster GEMM.

## Note on "outperforms CuPy"

`run_benchmark` compares against `cp.matmul` in **FP64** while most engine paths
run TF32, so those rows are mixed-precision wins — which is the engine's whole
purpose, but should be stated at that precision rather than as a like-for-like
result. The exception is `batched_matmul_fp64`: equal precision, and both
engines are already at FP64 hardware peak, where CuPy calls the same
`cublasDgemmStridedBatched`. Expect parity there.
