# v5.2 changes

Four changes: the strided-batch GEMM formulation (D2), the strided-batch layout
contract (D5), the tc_split FP16 range guard (D6), and the FP16 range guard
across the whole plain tensor-core tier (D7). A third (D3, the cuBLAS handle math mode) was tried and
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
