# v5.2 changes

Four changes: the strided-batch GEMM formulation (D2), the strided-batch layout
contract (D5), the tc_split FP16 range guard (D6), and the FP16 range guard
across the whole plain tensor-core tier (D7). **A second pass on 2026-09-20
added D8-D13** — Dgemv dispatch parity, the `batched_vector` formulation, the
fused epilogue's compute type, `matrix_power(mode=)`, the benchmark harness, and
a `rel_err` column; see the dated section at the end of this file. A third (D3, the cuBLAS handle math mode) was tried and
**reverted** — it bought no accuracy and cost up to 40%; see ENGINE_DIFFERENCES
D3 and the note at the top of `initialize_cublas`. v5.1 is preserved
verbatim in
`v5.1_reference/` (source + built `.so`), and both are loadable side by side —
`TensorMatrixOps(lib_path=".../v5.1_reference/cuda_matlib_v5.1.so")`.

## D3 — `CUBLAS_TENSOR_OP_MATH` -> `CUBLAS_TF32_TENSOR_OP_MATH`

`cuda_matlib.cuf:138` (`initialize_cublas`). Routines calling `cublasGemmEx`
pass `CUBLAS_GEMM_DEFAULT_TENSOR_OP` per call and are unaffected. Routines
calling `cublasSgemmStridedBatched` have no per-call algo argument and depend
entirely on the handle math mode:

`batched_matmul` (818), `batched_vector_multiply` (965), `tensor_4d_matmul`
(1136), `tensor_5d_matmul` (1246), `batched_matmul_tc_split` (1502/1513/1524).

### What was actually wrong

Not what `ENGINE_DIFFERENCES.md` D3 originally hypothesised. The deprecated
enum is **not** a no-op: `cublas_api.h` documents it as *"same effect as using
`CUBLAS_COMPUTE_32F_FAST_16F`"*, so those paths were running on **FP16** tensor
cores.

FP16 and TF32 both carry a 10-bit mantissa, so **accuracy is identical** and the
v5 review's tier table ("~1e-4 rel, TF32 10-bit mantissa") was numerically right
about the error while naming the wrong format. The real difference is exponent
range: FP16 tops out at 65504 and goes subnormal below ~6.1e-5; TF32 keeps
FP32's range.

Measured, `batched_matmul` at (8, 256, 256), varying input magnitude:

| input scale | v5.1 (FP16) | v5.2 (TF32) |
|---|---|---|
| 1e0 (benchmark scale) | 9.23e-05 | 9.19e-05 |
| 1e3 | 9.45e-05 | 9.51e-05 |
| 1e4 (near FP16 max) | 9.38e-05 | 9.40e-05 |
| **1e5 (above FP16 max)** | **non-finite** | 1.03e-04 |
| 1e-4 (near FP16 min normal) | 1.02e-04 | 1.01e-04 |
| **1e-6 (below FP16 min normal)** | **7.92e-03** (86x worse) | 9.22e-05 |

The benchmark could never have caught this: `create_inputs` scales every operand
by `1/(sqrt(n)*1.1)`, which parks all test data in the middle of FP16's range.

### What it costs

FP16 tensor cores run ~2x TF32 on this part, so the swap is not free.

**Corrected 2026-09-18.** The figures first recorded here (-6.5% to -17.6%) came
only from `batched_matmul`, which at these shapes is bound by its FP64<->FP32
conversion passes — that masks most of the GEMM difference. On compute-bound
paths the full tensor-core ratio shows. Min-of-7, RTX 4060, GFLOPS:

| op / shape | v5.1 (FP16) | v5.2 (TF32) | change |
|---|---|---|---|
| `matmul` 1024^3 | 5925 | 4511 | -24% |
| `matmul` 2048^3 | 8910 | 6687 | -25% |
| `matmul` 4048^3 | 14073 | 10288 | **-27%** |
| `matrix_power` 1024^3 | 12999 | 7813 | **-40%** |
| `matrix_power` 2048^3 | 18837 | 12417 | -34% |
| `matrix_power` 4048^3 | 23825 | 14239 | **-40%** |
| `batched_matmul` 2048^3 (conversion-bound) | 8582 | 7069 | -17.6% |

So the real price of FP32 exponent range is **24-40% on the compute-bound TF32
paths**, not under 18%. These ops are untouched by D2/D5; this is D3 alone.

**This is a judgement call, not an unambiguous fix, and the price is higher
than first reported.** v5.2 trades 24-40% on the compute-bound TF32 paths for
FP32 exponent range. If a caller can guarantee operands
stay inside FP16 range, v5.1's setting is the faster choice — the revert is one
enum. A per-call or init-time selector would be better than either default;
not done here to keep v5.2 to a single change.

### What this does *not* explain

`batched_vector_multiply`'s FP32-level accuracy (~1e-7 rel, vs the ~1e-4 its
tier claims). The enum swap left it **bit-identical** across all shapes tested
(16..1024). Its GEMM is `n x 1 x n`, and cuBLAS will not dispatch a
single-column GEMM to tensor cores under any math mode — it goes to a GEMV-like
kernel. So that row's accuracy is a consequence of the batch-of-GEMV
formulation, not of the handle setting, and `ENGINE_DIFFERENCES.md` D1's
"precision confound" framing was wrong.

## Note on the v5.1 code review

`CODE_REVIEW_TENSOR_CORE_ENGINE_V5.md` is thorough — it caught the `CUDA_R_TF32
= 8` constant as a "live grenade", and at line 275 it discusses
`cublasSetMathMode(CUBLAS_DEFAULT_MATH)` save/restore behaviour elsewhere in the
file. But it never examined the argument at `cuda_matlib.cuf:138`, so the
deprecated enum went unflagged, and the tier table then described those paths as
TF32 on the strength of their measured error.

Both formats give ~1e-4 at benchmark scale, so **no accuracy measurement could
have distinguished them** — the review would have had to read the enum against
the cuBLAS header, or test outside FP16 range. Worth adding to the review
checklist: for any tier claim, check the *format* is what's named, not just that
the error magnitude matches, and probe at least one input scale far from 1.0.


---

# D2 — `strided_batch_multiply`: serial loop -> chunked batched GEMM

`cuda_matlib.cuf`. The v5/v5.1 body looped over the batch issuing one
`cublasGemmEx` plus **two device-wide syncs per slice** — 8 batches cost 8 GEMMs
and 16 syncs. This is review finding **P3**, which v5.1 applied to
`tensor_4d_matmul`/`tensor_5d_matmul` but never to this routine; only the B6
64-bit offset fix landed here.

Now: convert a whole chunk in one `!$cuf kernel do(3)`, multiply it with one
`cublasGemmStridedBatchedEx`, convert back in one kernel. Two syncs per *chunk*
instead of per slice.

**`cublasGemmStridedBatchedEx`, not `cublasSgemmStridedBatched`** (which is what
`tensor_4d_matmul` uses), to keep the per-call `CUBLAS_GEMM_DEFAULT_TENSOR_OP`
this routine already had.

> **Correction.** An earlier draft of this file claimed that therefore "its math
> mode does not depend on the handle setting that D3 alters, so the two changes
> stay independent." **That is measurably false.** With the serial loop held
> identical and only the D3 enum differing, (8,2048^3) runs 6424 GFLOPS under
> `CUBLAS_TENSOR_OP_MATH` vs 5489 under `CUBLAS_TF32_TENSOR_OP_MATH` — **-14.6%**.
> The handle math mode influences `cublasGemmEx` dispatch even with an explicit
> per-call algo. D3's cost applies to this routine too; see "D3 also costs here"
> below.

Arbitrary `stride_a/b/c` still work: the workspace is always packed
(`mk`/`kn`/`mn`), and the conversion kernels read strided, write packed.

## Measured — D2 in isolation

Both builds use the v5.2 math mode, so this isolates the loop change. Median of
3 independent process runs, min-of-9 each, RTX 4060, GFLOPS:

| shape | serial loop | chunked batched | speedup |
|---|---|---|---|
| (8, 16^3) | 0.2 | 0.8 | **~4x** |
| (8, 32^3) | 1.9 | 5.9 | **3.1x** |
| (8, 64^3) | 15.4 | 46.0 | **2.99x** |
| (8, 128^3) | 122.2 | 298.7 | **2.44x** |
| (8, 256^3) | 678.5 | 1047.1 | 1.54x |
| (8, 512^3) | 1920.2 | 2138.1 | 1.11x |
| (8, 1024^3) | 3869.9 | 3912.3 | 1.01x |
| (8, 2048^3) | 5463.5 | 5424.4 | 0.99x |

Gain where the syncs dominated, parity once the GEMM does.

**Run-to-run variance at 1024^3/2048^3 is +-10%.** A single run of this pair once
showed 3428 vs 4012 and read as a 15% D2 regression; it did not reproduce over
3 reps. Do not draw conclusions at those two shapes from one measurement — this
is the same confound as D4 in ENGINE_DIFFERENCES.md.

## D3 also costs here

Separately from D2, and only visible once the two were measured apart. Identical
serial loop, only the handle math mode differing:

| shape | `TENSOR_OP_MATH` (FP16) | `TF32_TENSOR_OP_MATH` | change |
|---|---|---|---|
| (8, 512^3) | 1916.8 | 2052.5 | +7.1% |
| (8, 1024^3) | 4114.5 | 4011.6 | -2.5% |
| (8, 2048^3) | 6424.1 | 5488.6 | **-14.6%** |

So a v5.1 -> v5.2 comparison of this op nets D2's gain against D3's loss: large
net win at small shapes, roughly -15% at (8,2048^3). That -15% is the price of
FP32 exponent range documented in the D3 section, not a defect in D2.

## Cost: peak workspace

Chunking caps each pool slab at **8M FP32 elements (32 MiB)**, so peak workspace
is **<=96 MiB regardless of `batch_size`**.

The cap was chosen by measurement, not guessed: 128 MiB, 32 MiB and 16 MiB slabs
were compared over 3 runs each. 128 MiB and 32 MiB are indistinguishable at every
shape from 16^3 to 2048^3 (within ~2%), so the 4x smaller footprint is free.
16 MiB starts to cost at (8,2048^3) (~5015 vs ~5424).

| shape | v5.1 serial | v5.2 chunked (32 MiB cap) |
|---|---|---|
| (8, 2048^3) | ~50 MB | ~100 MB |
| (8, 512^3) | ~3 MB | ~25 MB |

Lower the `8388608_8` constant to trade throughput for footprint; raising it
buys nothing.

## Correctness

Verified against `cp.matmul` — 0 failures across batch 1..12, square and
non-square (`(3,100,200,300)`, `(5,300,100,7)`, `(7,33,65,129)`), and a
multi-chunk cases `(12, 1800^3)` (6 chunks of 2) and `(8, 2048^3)` (4 chunks of 2). Errors land at the TF32 tier
(~1e-4) where tensor cores engage and ~1e-7 on skinny shapes where cuBLAS picks
a non-tensor kernel. Full engine regression clean; `strided_batch` now returns
bit-comparable accuracy to `batched_matmul` at the same shape (1.022e-04 both),
which it should, since they are now the same kind of call.

## Against Rust

Indicative only — the Rust column is from the earlier notebook run (min-of-3,
separate session), not the same session as these numbers. A same-run
re-benchmark is the way to settle it.

| shape | Fortran v5.2 | Rust (earlier run) |
|---|---|---|
| (8, 16^3) | 0.7 | 0.72 |
| (8, 64^3) | 47.0 | 45.1 |
| (8, 128^3) | 330.0 | 335.0 |
| (8, 256^3) | 1231.6 | 1992.6 |
| (8, 512^3) | 2257.1 | 2457.2 |
| (8, 1024^3) | 3885.8 | 2218.0 |
| (8, 2048^3) | 5546.9 | 3766.7 |

The 3x Rust advantage at small sizes is gone. Rust retains a lead around
(8,256^3); its large-size deficit is its own F-order relayout pass, unchanged
here.


---

# D5 — `strided_batch_multiply`: consume arbitrary strides, no host relayout

Found while investigating why Rust still led 1.82x at (8,256^3) after D2.

## Diagnosis

The decisive comparison was Fortran-vs-Fortran, not Fortran-vs-Rust: in one
process, same shape, same math tier, `batched_matmul` ran ~1.6x faster than
`strided_batch_matmul` (1682 vs 1068 GFLOPS at (8,256^3)). Same engine, same
convert/GEMM/convert structure — so the gap was in how each is fed.

Two candidates were built and measured, and **both were refuted**: 64-bit index
arithmetic in the D2 kernels (1047 vs 1068, no change) and
`cublasGemmStridedBatchedEx` vs `cublasSgemmStridedBatched` (1082 vs 1068).

The cause was the Python wrapper. It did `cp.asfortranarray(a.transpose(1,2,0))`
on both inputs and `cp.ascontiguousarray(c_f.transpose(2,0,1))` on the output —
three full FP64 relayouts, because the kernel demanded column-major slices:

| shape | wrapper relayouts | total call | share |
|---|---|---|---|
| (8,128^3) | 43 us | 112 us | 38% |
| (8,256^3) | 120 us | 254 us | **47%** |
| (8,512^3) | 433 us | 1007 us | 43% |
| (8,1024^3) | 1553 us | 4398 us | 35% |

`batched_matmul` never pays this (it declares `a(k,m,batch_size)` and folds the
reversal into the conversion kernel), and neither does Rust.

## A first attempt that was wrong

The first fix flipped the contract to ROW-major and fused the transpose into the
conversion kernels. Measured against C-order inputs it looked excellent
(1.2-2.3x). **That measurement was invalid**: the benchmark builds its operands
with `cp.asfortranarray(...)`, so the wrapper's new `cp.ascontiguousarray()` was
a no-op in the test and a full FP64 relayout in the notebook. It only moved the
copy from one layout to the other:

| shape | v5.1 wrapper | row-major attempt | |
|---|---|---|---|
| (8,256^3) | 1050 | 1001 | -5% |
| (8,512^3) | 2007 | 1820 | -9% |
| (8,1024^3) | 3692 | 1836 | **-50%** |
| (8,2048^3) | 5150 | 3005 | **-42%** |

Lesson: measure with the memory order the caller actually uses. A layout fix
validated against the wrong layout will invert in production.

## The actual change

The kernel now takes **full element strides per array** — (batch, row, col) —
and reads through them, so any memory order works with no host-side copy.
C-order (batch,m,k) is `(m*k, k, 1)`; F-order is `(1, batch, batch*m)`. The
wrapper reads `arr.strides` and passes them; it no longer touches the data.
Strides are `c_int64`, so a large batch cannot overflow them (review B6 class).

**ABI change:** `py_strided_batch_multiply` now takes 9 stride arguments in
place of 3. No caller outside `tensor_matrix_ops.py`; `test/` is a
self-contained older snapshot with its own matching `.so`.

## Measured

Against F-order inputs — what the benchmark actually passes. Median of 2 runs,
min-of-9, GFLOPS:

| shape | v5.1 wrapper | strides | speedup |
|---|---|---|---|
| (8, 16^3) | 0.7 | 1.5 | **2.14x** |
| (8, 32^3) | 5.9 | 11.7 | **1.98x** |
| (8, 64^3) | 46.6 | 84.3 | **1.81x** |
| (8, 128^3) | 303.3 | 500.6 | **1.65x** |
| (8, 256^3) | 1058.0 | 1878.1 | **1.77x** |
| (8, 512^3) | 2006.9 | 3413.5 | **1.70x** |
| (8, 1024^3) | 3659.6 | 6388.2 | **1.75x** |
| (8, 2048^3) | 5164.6 | 8014.6 | **1.55x** |

C-order inputs also improve (476 / 1714 / 3002 / 4500 / 6856 at
128^3..2048^3). F-order now runs *faster* than C-order at large shapes: reading
down a column with batch-stride 8 coalesces better across threads than C-order's
k-stride.

At (8,2048^3), 8015 GFLOPS against v5.1's 5165 — so D5 comfortably outweighs
D3's cost on this op.

## Confirmed in the notebook (authoritative)

`benchmark_v5_1_RXT4060_fortran_vs_rust.ipynb`, RTX 4060, after a kernel
restart. This is D2+D5 combined and already net of D3's cost:

| shape | v5.1 | v5.2 | speedup | vs Rust |
|---|---|---|---|---|
| (8, 16^3) | 0.24 | 1.33 | **5.64x** | 1.89x ahead |
| (8, 32^3) | 1.89 | 10.63 | **5.62x** | 1.85x |
| (8, 64^3) | 15.2 | 80.07 | **5.26x** | 1.75x |
| (8, 128^3) | 122.3 | 474.9 | **3.88x** | 1.40x |
| (8, 256^3) | 691.6 | 1789.9 | **2.59x** | 1.07x |
| (8, 512^3) | 1984 | 3321.5 | 1.67x | 1.38x |
| (8, 1024^3) | 4190 | 5964.5 | 1.42x | 2.71x |
| (8, 2048^3) | 5919 | 7699.1 | 1.30x | 2.07x |

Fortran now leads Rust at every shape, including the (8,256^3) case that
prompted D5. Rust's numbers are unchanged across all runs; its remaining deficit
at large shapes is its own F-order relayout (ENGINE_DIFFERENCES D2).

## Correctness

0 failures across **both memory orders**, batch 1..12, square and non-square
(`(3,100,200,300)`, `(5,300,100,7)`, `(7,33,65,129)`), and multi-chunk
(`(12,1800^3)`, `(8,2048^3)`). Full engine regression clean.


---

# D6 — tc_split FP16 range guard

`matmul_tc_split` and `batched_matmul_tc_split` returned **NaN** for operands
above the FP16 maximum. Their T2/T3 GEMMs run under `CUBLAS_TENSOR_OP_MATH`,
which is the deprecated alias for `CUBLAS_COMPUTE_32F_FAST_16F` — FP16 tensor
cores, not the TF32 their own comments claimed. Operands above 65504 overflowed
to inf, and the failure was silent, in the tier callers choose *because* they
want accuracy. Comments corrected alongside the fix.

`improved_matmul32` also manipulates the handle math mode and was checked: it is
**not** exposed (finite, rel 5.2e-7, at max|A| = 1.5e6). Left alone.

## Why scaling, not an FP64 correction

This is a **range** problem, not a precision problem. Scaling by a power of two
is exact in binary floating point and commutes with the Dekker split —
`float(A*2^e) == float(A)*2^e`, and the low part scales with it — so it perturbs
no digit of the result, which is unscaled by `2^-(ea+eb)` in the existing
accumulate. No expensive tier is needed.

Scaling *up* also lifts the low parts (`|Al| ~ 2^-24 |Ah|`) out of FP16
subnormals, so the guard improves accuracy rather than merely avoiding NaN —
visible in the dynamic-range results below.

## Why detection is lazy

The first implementation reduced `max|A|`/`max|B|` up front, fused into the
split pass. Correct, but it cost the in-range path **6-24%** (medians of 5;
worst at small n), because each reduction forces its own device-to-host
synchronisation:

| n | no guard | eager guard |
|---|---|---|
| 512 | 3671 | 2804 (-24%) |
| 1024 | 7110 | 5940 (-16%) |
| 2048 | 11034 | 9963 (-10%) |
| 4048 | 13181 | 12353 (-6%) |

Restructured so the non-finite count rides on the accumulate pass that already
reads every output element, and the input reductions run **only after a failure
is observed**. Fast-path cost is now below measurement noise — the guarded build
measures 1-13% *faster* in clean medians, which a guard cannot actually do:

| op / n | no guard | lazy guard |
|---|---|---|
| `matmul_tc_split` 512 | 3566 | 3367 |
| `matmul_tc_split` 2048 | 10813 | 10976 |
| `matmul_tc_split` 4048 | 13100 | 12980 |
| `batched_matmul_tc_split` 256 | 2732 | 2987 |
| `batched_matmul_tc_split` 1024 | 9478 | 9610 |

On failure the routine rescales and retries once; if it is still non-finite it
prints the exponents and returns -2 rather than handing back NaN.

## Measured — the fix

`matmul_tc_split`, n=512, magnitude sweep:

| max abs operand | before | after |
|---|---|---|
| 1.5e+00 | 5.620e-07 | 5.620e-07 |
| 1.5e+04 | 4.993e-07 | 4.993e-07 |
| 1.35e+05 | **inf** | 5.431e-07 |
| 1.5e+05 | **inf** | 5.537e-07 |
| 1.5e+08 | **inf** | 5.434e-07 |
| 1.5e-10 | 5.570e-07 | 5.570e-07 |

In-range values are bit-identical, so nothing regressed.

`batched_matmul_tc_split`, dynamic range within a matrix — the case that
returned NaN from 1e6 upward:

| dynamic range | before | after |
|---|---|---|
| 1e2 | 4.616e-07 | 4.491e-07 |
| 1e6 | **nan** | 2.434e-07 |
| 1e9 | **nan** | 2.274e-07 |
| 1e12 | **nan** | 1.763e-07 |
| 1e15 | **nan** | 1.641e-07 |

Accuracy *improves* with wider dynamic range, because the scale-up lifts the low
parts out of FP16 subnormals — the predicted bonus.


---

# D7 — FP16 range guard across the plain tensor-core tier

D6 fixed `tc_split`. The plain tier had the same exposure and a worse failure
mode, because it has no FP32 term carrying the result:

| max abs operand | error |
|---|---|
| 1.35e+05 | **inf** (detectable) |
| 1.5e-06 | 5.6e-03 |
| 1.5e-07 | 1.2e-01 |
| 1.5e-08 | **1.000 — total loss, finite, silent** |

Overflow is catchable after the fact; **underflow is not**. So unlike D6 the
guard here has to be eager, which is what made it expensive to get right.

## Four implementations, three rejected

Each was built and measured. `batched_matmul`, medians of 5-7, GFLOPS:

| | 256^3 | 512^3 | 1024^3 | 2048^3 |
|---|---|---|---|---|
| no guard | 1812 | 3652 | 4889 | 8586 |
| `!$cuf kernel` reduction fused into conversion | 920 | 2387 | 5588 | 10022 |
| cuBLAS `Isamax` on the FP32 slab | 1266 | 2928 | 4466 | 7964 |
| custom block reduction, separate pass | 1297 | 3015 | 4503 | 7954 |
| **custom block reduction, FUSED** | **1497** | **3474** | **5313** | **8492** |

What each attempt taught:

- **`!$cuf kernel` reduction** (-49% at 256^3): nvfortran's reduction codegen,
  not memory traffic. Rejected.
- **`Isamax`** (-30%/-7%): the large-n cost was exactly the time to re-read the
  two FP32 slabs (268 MB ~ 1.07 ms at 2048^3). That is why the reduction has to
  be *fused* into the conversion, not run beside it.
- **Fused, 1-D flat index**: two 64-bit integer divisions *per element* made it
  slower than the `do(3)` it replaced. A 2-D grid (x walks the packed index, y
  walks the planes) moves the division to once per block per plane: 2048^3 went
  7487 -> 9392.
- **`d_maxabs = 0.0`** is a BLOCKING `cudaMemcpy` at the head of every call.
  Replaced with an async kernel. (Helped less than predicted.)

`convert_reduce` is the result: one kernel doing the strided gather, the
FP64->FP32 conversion and a shared-memory max-abs tree reduction with one
`atomicmax` per block, result device-resident. Every plain-tier conversion in
the file is the same "gather through (batch,row,col) strides -> convert -> pack",
so one kernel serves them all.

## The iterative routines need more

`matrix_dot` and `tensor_matrix_multiply` compound magnitude: max|A| = 300 is
comfortably inside FP16 range, but max|A^2| reaches 4.6e7. cuBLAS accumulates in
FP32, so only each GEMM's *inputs* must be in range -- but every squaring's FP32
output becomes the next input. Both routines therefore re-measure and re-scale
the live buffer before each further GEMM (`guard_buf`), tracking `etot`, the
exponent the buffer carries relative to the true value, and dividing it out in
the write-back.

**`funscale` must be FP64.** It was FP32 in the first version, and with
`etot = 158` (reached from max|A| ~ 8.5e-12 at p=4) `2^-158 = 2.7e-48`
underflows FP32's min normal and silently became zero. Caught by testing at
1e-12; the routine was correct at every scale where the guard did not trip,
which is the signature of an unscale bug.

## Cost

| op / shape | v5.1 | v5.2 | |
|---|---|---|---|
| `matmul` 4048^3 | 14822 | 14851 | ~0 |
| `matmul` 2048^3 | 9143 | 9134 | -0.1% |
| `matmul` 1024^3 | 5968 | 5471 | -8.3% |
| `matrix_power` 4048^3 | 25274 | 24582 | -2.7% |
| `matrix_power` 2048^3 | 18890 | 17711 | -6.2% |
| `matrix_power` 1024^3 | 13239 | 11266 | -14.9% |
| `batched_matmul` 256^3 | 1812 | 1497 | -17% |

**0-17%, smallest where absolute time is largest.** `matrix_power` costs most
because the per-iteration guard adds a synchronisation per squaring. The
residual small-shape cost is fixed per-call overhead (extra launches, one
blocking device-to-host read for the host-side branch); four block-count
configurations were tried and the differences were inside the noise.

For comparison, D3 -- switching the handle to TF32, which would have fixed the
same hazard -- cost 27-40% and was reverted.

## Coverage

Guarded: `matmul`/`tensor_matrix_multiply`, `matrix_dot`, `batched_matmul`,
`strided_batch_multiply`, `tensor_4d_matmul`, `tensor_5d_matmul`,
`batched_vector_multiply`, `matmul_tc_split`, `batched_matmul_tc_split`.

Not guarded, by measurement rather than omission: `improved_matmul32` (finite,
rel 5.2e-7, at max|A| = 1.5e6), `batched_matmul_fp64` and the GEMV paths (FP64,
no FP16 involved).

Verified: 0 failures over 7 magnitudes x 6 routines, plus `matrix_power` over 8
magnitudes x 4 powers (including p=8, three squarings), both memory orders,
square and non-square, multi-chunk. Full engine regression clean.

## The Rust engine has the identical bug

Tested directly, same sweep, same shapes:

| max abs operand | Fortran (pre-guard) | Rust `batched` | Rust `strided` |
|---|---|---|---|
| 1.35e+05 | non-finite | **non-finite** | **non-finite** |
| 1.5e-06 | 5.599e-03 | **5.867e-03** | 5.867e-03 |
| 1.5e-08 | 1.000 | **1.000** | 1.000 |

Same threshold, same magnitudes. `rust_matlib/src/lib.rs` sets the same
deprecated `TENSOR_OP_MATH` on its handle, and its comment records the same
misreading:

```rust
/// TENSOR_OP_MATH + COMPUTE_32F + GEMM_DEFAULT_TENSOR_OP = 17.7 TFLOPS,
/// COMPUTE_32F_FAST_TF32 (any mode) = 12.5 TFLOPS, same TF32 accuracy.
```

That 17.7-vs-12.5 TFLOPS gap is FP16 versus TF32 — the author measured the
precision difference and attributed it to "mode", concluding both gave TF32
accuracy. They give the same *mantissa* (10 bits) and therefore the same error
at benchmark scale, which is exactly why neither engine's accuracy columns ever
flagged it. Rust's `tc_split` survives underflow (it converts with an explicit
`cvt_rna_tf32_f32`) but still goes non-finite on overflow.

**Rust is now fixed the same way** (2026-09-18): `narrow_maxabs_f64_f32` fuses
the max-abs reduction into the FP64->FP32 conversion, both operands share one
partials buffer so their maxima return in a single transfer, and the rescale is
the same exact power of two. `rs_matrix_power` re-guards between squarings.
Mode 0 of `tc_split` needed no guard — it converts with `cvt_rna_tf32_f32`, real
TF32 with FP32's range. Cost -1.2% to -12.3% (details in
`tensor_core_engine_rust/RESULTS.md`); an unfused reduction was measured first
at -10% to -37%, the same lesson as the Fortran side.

This is the clearest argument in the project for keeping two independent
implementations. The bug survived a thorough code review of the Fortran engine
and a from-scratch reimplementation in another language, because both inherited
the same misreading of one cuBLAS enum, and no accuracy measurement could
distinguish FP16 from TF32. What exposed it was comparing the two engines
closely enough that a 1.8x speed difference on one row demanded an explanation.

---

# 2026-09-20 — accuracy, comparability, and the benchmark itself

A second pass, driven by a single question: *why does Fortran still win
`matrix_multiply`, `vector_matrix` and `matrix_vector`?* Two of those three
turned out not to be real, the third was dispatch overhead rather than kernels,
and chasing them surfaced two genuine accuracy bugs and one measurement bug that
had been hiding results since the comparison notebook was written.

Everything below is measured on the RTX 4060 (Vengeance), FP64 inputs scaled by
`1/(sqrt(k)*1.1)`, min-of-N timing with the GPU clock-warmed.

## D8 — the Dgemv tier was never a kernel gap

`vector_matrix` and `matrix_vector` sat at 0.83-0.95 of Fortran. Both engines
call the **same** `cublasDgemv` — `cuda_matlib.cuf:1319/1348` and
`rust_matlib/src/lib.rs:1971/1987` — and `max_diff` is exactly `0.0` on every
row, so there was no math to be slower at. At n=16..1024 these ops are
20-38 us end to end and only a few microseconds of that is the GPU: the deficit
was a **constant ~3.5 us per call**, which is why it read as a fixed ratio
rather than a shape-dependent one.

Two fixes:

- `entry()` (`lib.rs`) did a full `cuCtxSynchronize` **on entry**. It bought
  nothing — `State::stream` is the context's default stream, the same legacy
  stream CuPy enqueues on, so anything CuPy produced is already ordered ahead —
  and it was stricter than the Fortran engine, which only syncs on exit.
- `_dev2` built an `x.T` view purely to read a pointer. `_sq_ptr`/`_vec_ptr` in
  `rust_matrix_ops.py` return `(array, ptr, trans)` without materialising it.

| | before | after |
|---|---|---|
| library boundary, matched op | Rust +0.6 to +0.7 us | **0.0 +/- 0.15 us** |
| end-to-end ratio (notebook) | 0.83-0.95 | **0.91-0.96** |

The residual ~1.7 us is CuPy/ctypes wrapper cost on both sides. A stripped
wrapper (no validation, no NumPy support) runs 1.3-1.6 us *faster* than Fortran,
so the headroom is real and Python-side; it was deliberately not taken, because
taking it means dropping input validation and the NumPy-in/NumPy-out contract.

**`matrix_multiply` was never a Fortran win** — it is parity (1.005-1.01x with
F-order inputs). The earlier appearance of a gap was a harness bug; see D13.

## D9 — `batched_vector` compared two different precision tiers

Rust was 1.46-1.54x "faster" here. It was not the same operation:

| | Fortran | Rust (before) |
|---|---|---|
| call | `cublasSgemmStridedBatched`, **N = 1** | one `cublasGemmEx`, batch folded into **N** |
| kernel cuBLAS picked | FP32 | FP16 tensor-core |
| max_diff | ~3e-8 | ~1.4e-5 to 5.6e-5 |

Both handles are `CUBLAS_TENSOR_OP_MATH` (the `FAST_16F` alias, D3), so with
N = batch cuBLAS is free to choose a tensor-core kernel and does; with N = 1 it
cannot. Proof that N is what decides: `strided_batch`, N = n, sits at ~1.5e-5 on
**both** engines.

`rs_batched_vector_matmul` now uses Fortran's N = 1 formulation. From row-major
buffers this needs no extra transpose, only the right leading dimensions:
`transa=T, lda=n, strideA=0`; `transb=T, ldb=batch, strideB=1` (so batch *b*
reads `pv[i*batch + b]`); `ldc=n, strideC=n`, which writes Y straight out as
(batch, n) row-major.

Result: error now **2.1e-8 to 5.2e-8**, at or below Fortran's, verified at
n=7..513 and batch=1..100 — **and it got faster, 1.46-1.54x -> 1.63-1.94x**.
The N=1 batched kernel suits these tiny shapes better than one wide tensor-core
GEMM. The expected speed-for-accuracy trade did not materialise.

## D10 — the fused epilogue was never pinned to a tier

Found while checking D9's neighbours: at **(8, 16, 16, 16) only**, the cuBLASLt
bias+relu epilogue gave Fortran 1.3e-7 and Rust 2.5e-4. Both descriptors already
requested `CUBLAS_COMPUTE_32F_FAST_TF32` (`cublaslt_bridge.c:134`,
`lib.rs`), so this was not a bug in either engine — it was the heuristic
returning an FP32 kernel for one layout and a TF32 kernel for the other.

`FAST_TF32` is *permission* to use TF32, not a pin. Asking for plain
`CUBLAS_COMPUTE_32F` forbids it:

| shape | before (both ~) | after (both) | throughput cost |
|---|---|---|---|
| (8, 16^3) | 1.3e-7 / 2.5e-4 | 1.3e-7 / 1.5e-7 | none |
| (8, 64^3) | 1.6e-4 | 1.4e-7 | -9% / -11% |
| (8, 256^3) | 7.6e-5 | 3.0e-7 | -15% / -20% |
| (8, 1024^3) | 4.1e-5 | 3.5e-7 | -22% / -31% |
| (1, 512x2048x128) | 2.2e-5 | 2.2e-7 | -11% / -13% |

**~1e-4 -> 8.8e-8..3.5e-7 at every shape on both engines, for 0-31% of the
throughput**, and Rust still leads 1.16-1.59x. Exact FP32 is now the default.

Implemented as **one switch shared by both engines**: `lt_compute_type()` in
`rust_matlib/src/lib.rs` and in `cublaslt_bridge.c`, both reading
`TME_EPILOGUE_TF32` (unset/`0` = exact FP32, `1` = the old TF32) and resolving
once per process. One variable moves both engines, so they cannot silently drift
into different tiers again — which is exactly how this was found.

Note for downstream callers: the **Fortran** engine's epilogue tier changed too.
Anything that assumed ~1e-4 from `batched_matmul_bias`/`bias_relu` now gets
~1e-7 and somewhat less throughput.

## D11 — `matrix_power` gained `mode=`

Context first, because the `max_diff` column badly misrepresents this operation.
That column is **absolute**, and `|A^4|max` grows 13 -> 184 from n=256 to
n=4048 while `create_inputs` holds `|A.B|max` at ~0.22-0.25 for every n. So
`matrix_power` looks ~100x worse than `matrix_multiply` when, relative to the
result, it is ~2x:

| | relative error (TF32) |
|---|---|
| `matmul` | 5.5e-5 .. 7.3e-5 |
| `matrix_power` (p=4) | 6.0e-5 .. 1.7e-4 |

Same tier. The 1.1-2.4x is chaining: `power=4` is two GEMMs, and the first
one's error feeds the second.

`mode="tc_split"` routes every multiply in the binary-exponentiation chain
through `matmul_tc_split` with FP64 intermediates:

| n | TF32 GF / rel | tc_split GF / rel | cost |
|---|---|---|---|
| 256 | 933 / 9.7e-5 | 430 / 2.6e-7 | 0.46x |
| 1024 | 13279 / 6.5e-5 | 3217 / 3.5e-7 | 0.24x |
| 4048 | 22279 / 1.7e-4 | 4540 / 5.0e-6 | 0.20x |

**TF32 remains the default.** The ~4-5x is structural to the split tier, not
specific to `matrix_power`: `matmul` pays the same (18356 -> 4532 GF at n=4048,
0.25x) for the same accuracy, because tc_split is 3 GEMMs of which the dominant
one (T1, the exact-FP32 product) runs on **CUDA cores, not tensor cores**.

The chain lives **inside both `.so` files**, not in the wrappers, so C and
Fortran callers get it too:

- Rust: `rs_matrix_power(a, c, n, power, mode)`, an exact-size f64 scratch pool
  on `State`, and the mode-1 tc_split core factored out of `rs_matmul_tc_split`
  as `tc_split_cublas`.
- Fortran: `py_matrix_dot(a, c, n, power, mode)` and 2-D FP64 square pools
  `wp_q1..q3` in `workspace_pool.cuf`.

Both use **slot rotation**: every tc_split operand is read-only, so base and
result are tracked as slot ids that may alias, the base starts as the caller's
`a` without being copied, and each product goes to whichever of three slots
neither occupies. Copies for `power=4`: **5 -> 1** (the final write to `c`).
Fortran dispatches this through `mp_free_slot` / `mp_mul` / `mp_mul_into`
(3 + 16 select-case branches rather than one flat 48-way).

Verified across **powers 1-17** at n=16/64/257/512 on both engines — every bit
pattern exercises a different rotation path — plus NumPy round-trip, bad-mode
and `power=0` rejection. Zero failures.

### Three CUDA-Fortran traps, in order of how long they cost

1. A local `save` device allocatable inside a helper. Suspected, not the cause.
2. Rank-remapped 2-D device POINTERs passed into a `bind(c)` callee. Suspected,
   not the cause.
3. **The cause:** `matmul_tc_split` calls `wp_ensure_fp32_7`, and a device
   allocatable reallocated by a *callee* leaves the caller's already-captured
   module descriptors stale. It surfaced as an illegal address in the copy-back
   kernel **after** a GEMM that had itself succeeded, so the blame landed on the
   wrong line until it was bisected with per-step syncs. `workspace_pool.cuf`'s
   own NOTE warns about this class of bug. Fix: pre-size the FP32 pools with
   `wp_ensure_fp32_7` before the chain starts, so the callee's ensure is a no-op.

Also worth remembering: **a failed nvfortran compile still produced a loadable
`.so`**, because shared libraries tolerate undefined symbols. `nm -D
--undefined-only` caught a stale `wp_ensure_fp64_2_` that would otherwise have
failed only at call time. Check it after any change to a module's public
interface.

## D12 — what the benchmark was hiding

Three defects in the comparison notebook itself, all of which changed what the
summary table said:

- **`batched_matmul` and `batched_matmul_fused` never appeared at all.** The
  Rust cell assigned `batched_matmul_results` / `batched_matmul_fused_results` —
  the same names as the Fortran cell, not the `_rust`-suffixed ones the summary
  pairs on — so it silently overwrote the Fortran lists and the summary found no
  twin. Ten rows had been missing from every table ever produced. Renamed; Rust
  leads 1.21-1.66x on them at matching accuracy.
- **`vector_matrix_optimised` was not a second code path.** Its `tensor_op`
  called `vector_matmul`, the same method the `vector_matrix` cell already
  benchmarks. Deleted from all five notebooks (neither engine has an "optimised"
  single-vector entry point; there are exactly three vector entry points and all
  three already had cells). Useful parting gift: the two identical measurements
  differed by up to 7%, which fixed the harness noise floor at `repeat=3`.
- **`repeat=3` could not resolve these ops.** `time_operation`'s sample count is
  now chosen per shape by `repeats_for(flops)` — 25 samples under `SMALL_FLOPS`
  (50 MFLOP), 3 above, so the dispatch-bound rows stop wandering while a 0.6 s
  FP64 reference call at (8, 2048^3) still costs 3 samples. The two hand-rolled
  cells (`improved_matmul32`'s 4-tier table, `batched_matmul`'s 3-way table)
  bypass `run_benchmark` and were wired in via `max(REPEAT, repeats_for(flops))`
  — a floor-raiser, never a reducer, so their existing REPEAT=5/10 on big shapes
  is untouched.

That last one dissolved an apparent finding: `vector_matrix` (0.99 mean) and
`matrix_vector` (0.90 mean) looked like different results, and a 400-sample
measurement puts **both at ~0.93** with identical raw-boundary deltas.

A `matrix_power_tc_split` block was added to all five notebooks so the new mode
is visible as its own row rather than buried in a docstring.

## D13 — `rel_err`, and the harness bug that cost two wrong conclusions

`calculate_metrics` now also returns `rel_err` (`max_diff / max|reference|`) and
`ref_mag`, and the summary table, `print_results` and `generate_summary_report`
all carry a Rel Err column. Absolute `max_diff` is fine within one operation and
actively misleading across operations, for the reason set out in D11.

The bug that motivated a lot of the above: **`a = cp.asfortranarray(x) / scalar`
returns a C-order array.** CuPy elementwise ops produce C-order output
regardless of input order; only in-place `a /= scalar` preserves F-order (which
`create_inputs` correctly does). `tensor_matrix_ops.py` calls
`cp.asfortranarray()` on its inputs, which is a no-op on an F-order array and a
**real n^2 FP64 transpose copy** on a C-order one — 8 MB at n=1024. Getting this
wrong in a side harness made the Fortran engine look ~2x slower than it is and
produced two confidently wrong conclusions in one session ("Rust wins
`matrix_multiply` everywhere" — it is parity; "Rust is 1.6-1.8x faster at Dgemv"
— it is 0.93x) before the assertion `assert a.flags.f_contiguous` was added to
the harness.

General form, and the one worth carrying forward: **when a measured gap is far
larger than the mechanism can explain, suspect the harness before the code.**

## Where v5.2 stands

| operation | relative error | Rust / Fortran |
|---|---|---|
| `batched_matmul_fp64`, `vector_matrix`, `matrix_vector` | exact (0.0) | 0.89-0.96 (dispatch) |
| `batched_vector` | ~2e-8 | 1.63-1.94x |
| `batched_matmul`, `batched_matmul_fused` | ~8e-8 | 1.21-1.66x |
| `matrix_power` (tc_split) | 2.6e-7 .. 5.1e-6 | ~1.0 |
| `matrix_multiply`, `strided_batch`, `matrix_power` (TF32) | 5e-5 .. 2e-4 | 0.97-1.4x |

Two operations moved a full precision tier (`batched_vector`, the fused
epilogue), one gained a tier as an option (`matrix_power`), and the three tiers
are now *pinned* rather than left to a cuBLAS heuristic.

---

# v5.3 (started 2026-09-20) — the API boundary, found by a real model

## D14 — the FP64 boundary, found by a real model

`mlp_example_fortran_vs_rust_vs_cupy.ipynb` runs a 64-1024-1024-10 MLP
(sklearn digits, 98.0% test accuracy, trained in FP64 CuPy) and puts batch
inference through both engines and both CuPy tiers. Every backend classifies
identically — ~4e-7 relative on the logits, **zero of 450 digits change label**,
including the TF32 tier at ~7e-4 — so accuracy was never the question.

Throughput was. With FP64 entry points, at batch 16384:

| backend | GFLOPS |
|---|---|
| cupy (FP64) | 217 |
| **cupy (FP32)** | **5644** |
| fortran | 2866 |
| rust | 3446 |

The engines *lost* to plain unfused CuPy FP32. Taking one layer apart,
(1024 x 1024) @ (1024 x 16384):

| | ms |
|---|---|
| engine `matmul`, TF32 tier, no epilogue | 2.87 |
| cupy FP32 GEMM alone | 3.66 |
| cupy FP32 GEMM + bias + relu | 4.97 |
| cupy FP32 incl. casting both operands | 5.57 |
| engine fused epilogue, exact FP32, FP64 in/out | 6.93 |

The FP64 entry points narrow both operands, run the FP16 range guard's max-abs
reduction, and widen the result — **per call**, which for a chained model means
per layer. The op-level benchmark cannot see this: it times one call with the
conversion folded into it.

## The fix, and how far it goes

`rs_batched_matmul_bias_f32` (Rust) and `py_batched_matmul_bias_relu_f32` /
`py_batched_matmul_bias_f32` (Fortran), dispatched automatically — pass float32
arrays to `batched_matmul_bias_relu` and the conversion-free path is taken,
float32 in, float32 out.

Rust's operands reach cuBLASLt **untouched**: its descriptor already reads the
row-major buffers as their column-major transposes via TRANSA/TRANSB, so there
is no input copy at all. Fortran still stages A and B row-major -> column-major,
because the C bridge builds its layouts with OP_N; giving that side the same
treatment means a transposed variant in `cublaslt_bridge.c` and was left for
later. That asymmetry shows up directly in the results.

Per layer: engine fused epilogue **6.93 -> 6.04 ms** (Rust). End to end:

| batch | fortran | fortran (FP32) | rust | rust (FP32) | cupy (FP32) |
|---|---|---|---|---|---|
| 1024 | 2498 | 2466 | 3437 | **4093** | 5365 |
| 4096 | 2822 | 2788 | 3592 | **4411** | 5682 |
| 16384 | 2866 | 2947 | 3446 | **4127** | 5644 |

**Rust +19-23%, Fortran roughly flat** — the latter consistent with it keeping
the input staging. Accuracy is bit-for-bit what the FP64 path produced
(identical relative error at every shape tested, n=7..1024, batch 1..8).

## What is still in the way

Not FP64 any more — the **output transpose**. cuBLASLt's bias epilogue only
accepts column-major layouts, so D always lands transposed relative to the
row-major buffer the caller wants back. Measured at this shape:

| | ms |
|---|---|
| forced col-major -> row-major transpose of D | 1.30 |
| straight copy of the same bytes | 0.55 |
| the bias+ReLU pass the fusion saves | ~1.0 |

So the fusion buys ~1.0 ms and the transpose costs ~1.3 ms. That is the whole
reason the fused path lands near CuPy's unfused one instead of ahead of it, and
it is why the FP32 entry point alone does not close the gap.

## D15 — the transpose, removed by layout

Tried the obvious thing: keep the activations column-major. It needed a little
engine support, not a new kernel — `rs_batched_matmul_bias_f32` gained `b_cm`
and `c_cm` flags, `lt_plan` gained `b_cm` in its cache key (the descriptor sets
TRANSB=N and describes B as column-major (k x n) instead of row-major read
transposed), and the epilogue then writes straight into the caller's F-order
buffer with no transpose pass at all. The wrapper dispatches on layout: hand
`batched_matmul_bias_relu` an F-order 2-D activation block and it takes that
path, returning an F-order result which is itself a valid operand for the next
layer. Weights stay C-order.

Per layer, (1024 x 1024) @ (1024 x 16384):

| | ms | GFLOPS |
|---|---|---|
| cupy FP32 GEMM + bias + relu (no casts) | 5.00 | 6866 |
| engine fused, exact FP32, **FP64 in/out** | 6.84 | 5024 |
| engine fused, **FP32 in/out** (D14) | 6.00 | 5730 |
| engine fused, **FP32 + F-order activations** | **4.90** | **7012** |

The saving is 1.10 ms against a transpose measured at 1.33 ms — the prediction
held. And at the layer level the fused path is now **ahead of unfused CuPy
FP32**, which is where it should have been all along.

End to end on the MLP (GFLOPS, Rust):

| batch | cupy (FP32) | rust | rust (FP32) | rust (FP32, F-order) |
|---|---|---|---|---|
| 64 | 578 | 644 | 659 | **864** |
| 256 | 2248 | 1901 | 1943 | **2476** |
| 4096 | 5683 | 3644 | 4480 | 5255 |
| 16384 | 5734 | 3488 | 4143 | 5636 |

Ahead of CuPy at small and moderate batches, level (98%) at the largest — the
remaining difference being the `asfortranarray` conversion, which is inside the
timed region here but would not exist in a pipeline that already held its
activations column-major.

Accuracy is unchanged throughout: ~4.2e-7 relative on the logits, **zero**
prediction flips, identical test accuracy, for every backend in the table.

## D16 — the same path in Fortran, and the bigger jump of the two

`cublaslt_bridge.c` gained a transposed-A variant, keyed in the plan cache by a
new `ta` field: A row-major (m x k) consumed via `CUBLASLT_MATMUL_DESC_TRANSA =
OP_T` with layout (k x m), B already column-major (k x n) via OP_N, D
column-major (m x n) straight into the caller's buffer. Exposed as
`lt_matmul_bias_fp32_t` / `lt_matmul_bias_relu_fp32_t`, wrapped by
`py_batched_matmul_bias_relu_f32_t` / `_bias_f32_t`, and dispatched by the same
layout test the Rust wrapper uses.

`lt_fused_common_f32_t` stages **nothing** — no `wp_*` at all, just the bridge
call and a sync — where the C-order path stages A and B row-major to
column-major regardless of dtype *and* transposes the result back.

| one layer, (1024 x 1024) @ (1024 x 16384) | ms | GFLOPS |
|---|---|---|
| cupy FP32 GEMM + bias + relu (no casts) | 4.98 | 6902 |
| **fortran** FP32, C-order activations | 7.68 | 4473 |
| **fortran** FP32, F-order activations | **4.80** | **7162** |
| **rust** FP32, C-order activations | 6.08 | 5648 |
| **rust** FP32, F-order activations | **4.84** | **7093** |

Fortran's is the larger jump — **-37%** against Rust's -20% — because F-order
removes *both* of its costs at once, the input staging and the output
transpose, where Rust only ever paid the latter. The two engines end up level,
as they have on every other op once the formulations match.

End to end on the MLP (GFLOPS):

| batch | cupy (FP32) | fortran | rust | fortran (FP32, F) | rust (FP32, F) |
|---|---|---|---|---|---|
| 64 | 581 | 462 | 638 | **862** | **864** |
| 256 | 2229 | 1406 | 1880 | **2461** | **2482** |
| 4096 | 5684 | 2744 | 3623 | 5267 | 5252 |
| 16384 | 5715 | 2902 | 3504 | 5573 | 5498 |

Ahead of CuPy FP32 at small and moderate batches, ~97% of it at the largest —
the residual being the `asfortranarray` conversion inside the timed region,
which a pipeline already holding activations column-major would not pay.

Accuracy is untouched by any of it: ~4.2e-7 relative on the logits and **zero**
prediction flips for all eight backends.

## What is still open

- **Chain several layers in one call**, so intermediate activations never return
  to the caller at all.
- The F-order path is single-batch on both engines (`batch == 1`), which is what
  a per-layer activation block is. Batched F-order would need a stride
  convention nothing currently asks for.

## The methodological point

The op-level benchmark is six months of careful work and it could not have found
this, because every cost here is *per call* and it measures one call at a time.
It took a real model, with layers chained, to expose an API boundary that is
invisible one operation at a time — and a like-for-like FP32 baseline to make
the gap visible at all. Against FP64 CuPy the same engines look 13-16x faster.
