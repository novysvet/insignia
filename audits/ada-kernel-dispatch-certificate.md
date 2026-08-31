# Certifiable Ada kernel dispatch

## Status

This deliverable treats dispatch as a finite statistical decision problem, not as an occupancy heuristic. It adds a CPU-only reference implementation and reproducible synthetic evaluation:

- `tools/ada_dispatch_certificate.py`
  - sm_89 resource-residency calculator;
  - `ptxas -v` resource parser;
  - paired, time-uniform best-arm certificate;
  - monotone, piecewise-constant, low-rank, and safe structured simulators;
  - minimax and distributionally robust dispatch helpers;
  - finite compile/measure stopping certificate;
  - exact small-DAG scheduler with launch and workspace costs.
- `tools/test_ada_dispatch_certificate.py`
  - 20 CPU tests, including occupancy cross-check and analytical-reversal cases.
- `scratch/ada-dispatch-certificate/`
  - `dispatch-table.hpp`;
  - `dispatch-certificate.json`;
  - `best-arm-summary.csv`;
  - `evaluation.json`.

The emitted local table matches the accepted 4070 SUPER evidence. The artifact does not claim that the historical 21 timing repetitions constitute a family-wise certificate because their raw samples are not present in the supplied archive. The overclocked 4070 Ti SUPER is explicitly `unmeasured`; its table is absent rather than interpolated.

## 1. Decision object and certificate contract

Let `x` name the complete dispatch state relevant to latency:

\[
x=(\text{geometry},B,\text{alignment},\text{cache state},\text{operation mode},\text{launch context}).
\]

Let `K_x` be the finite set of exact kernels legal for `x`. A machine regime `m` includes GPU identity, clocks, power profile, driver/runtime, compiler/cubin, and the benchmark protocol. The latency random variable is

\[
T_m(k,x,z),
\]

where `z` is hidden transient state. A certificate names all of the following:

1. a machine and compiler fingerprint;
2. an estimand, including any timeout or clipping rule;
3. simultaneous confidence intervals or a stated absence of them;
4. an optimality tolerance `epsilon`;
5. the static table selected from those intervals;
6. invalidation triggers and a remeasurement plan.

A dispatch table without these fields is a cache of observations, not a speed guarantee.

## 2. Exact sm_89 resource residency

### 2.1 Architectural limits

For compute capability 8.9 the CUDA programming and Ada tuning guides give:

| Resource | sm_89 limit |
|---|---:|
| resident CTAs per SM | 24 |
| resident warps per SM | 48 |
| resident threads per SM | 1,536 |
| threads per CTA | 1,024 |
| 32-bit registers per SM | 65,536 |
| registers per CTA | 65,536 |
| registers per thread | 255 |
| shared capacity per SM | 100 KiB |
| user-addressable shared bytes per CTA | 99 KiB |

Ada reserves 1 KiB of shared capacity per resident CTA. The supported shared-memory carveouts are 0, 8, 16, 32, 64, and 100 KiB.

The model also carries allocation micro-constants in `AllocationAssumptions`:

- register allocation unit: 256 registers per warp;
- register-allocation warp granularity: four warps;
- shared-memory allocation unit: 256 bytes;
- CTA barrier slots: supplied by Nsight Compute or left unknown.

The first three defaults match the modern CUDA occupancy convention. A production certificate must record the values returned by `ncu_occupancy.get_gpu_data(8, 9)` or cross-check every compiled function with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. The supplied CTA widths are four and eight warps, so rounding the CTA warp count to a four-warp allocation unit does not change them. Every calculator result carries `resource_model_status` and `unmodeled_resource_limits`; the portable fallback marks a barrier-using kernel conditional until an exact barrier capacity is supplied or the runtime result matches.

### 2.2 Allocation formulas

For a compiled kernel with `t` threads per CTA, `r` registers per thread, requested shared memory `s`, and `h` CTA barriers, define

\[
w=\left\lceil \frac{t}{32}\right\rceil,
\qquad
\widetilde w=4\left\lceil \frac{w}{4}\right\rceil.
\]

With a 256-register warp allocation unit,

\[
R_{\rm warp}=256\left\lceil\frac{32r}{256}\right\rceil,
\qquad
R_{\rm CTA}=\widetilde w R_{\rm warp}.
\]

The allocated shared footprint is

\[
S_{\rm CTA}=256\left\lceil\frac{s+1024}{256}\right\rceil.
\]

The independent CTA limits are

\[
\begin{aligned}
n_{\rm block}&=24,\\
n_{\rm thread}&=\left\lfloor\frac{1536}{t}\right\rfloor,\\
n_{\rm warp}&=\left\lfloor\frac{48}{w}\right\rfloor,\\
n_{\rm reg}&=\left\lfloor\frac{65536}{R_{\rm CTA}}\right\rfloor,\\
n_{\rm smem}&=\left\lfloor\frac{102400}{S_{\rm CTA}}\right\rfloor,\\
n_{\rm barrier}&=\left\lfloor\frac{H_{\rm SM}}{h}\right\rfloor
\quad\text{when the barrier capacity is known.}
\end{aligned}
\]

The exact resource residency is

\[
n_{\rm CTA}=\min_i n_i,
\qquad
n_{\rm resident\ warps}=w n_{\rm CTA},
\qquad
O_{\rm theoretical}=\frac{w n_{\rm CTA}}{48}.
\]

Per-CTA launch feasibility is checked before this minimum: thread, register, and shared-memory maxima must hold. Spills do not make a launch infeasible, but they are recorded and can be a compile-search rejection rule.

The calculator exposes every individual limit and all tied limiting resources. There is no hidden “occupancy score.” The helper `validate_runtime_residency` compares the static count with the CUDA occupancy API for the exact cubin and dynamic shared-memory setting; a mismatch invalidates the resource assumptions.

### 2.3 Residency, grid underfill, achieved occupancy, issue limits

These quantities must not be conflated.

**Resource residency** is the static maximum above. It answers how many CTAs and warps can coexist on one SM.

**Grid underfill** limits the first or final wave when the grid has fewer blocks than total device capacity. The tool reports an upper bound from grid blocks and SM count. It does not call this achieved occupancy.

**Achieved occupancy** is active warps per active cycle measured over execution. It can be lower because of tails, imbalance, short kernels, preemption, and launch dependencies. The static result leaves `achieved_occupancy=null`.

**Issue utilization** depends on eligible warps in each SM subpartition, instruction dependencies, pipe availability, scoreboard state, barriers, and instruction fetch. The calculator deliberately does not infer it. A measurement campaign should collect the Nsight Compute scheduler statistics, including active, eligible, and issuing warps plus stall reasons. Higher residency can coexist with worse issue utilization.

### 2.4 Real resource points

The repository audits supply these compiled metadata:

| Kernel | Threads | Registers/thread | Requested shared | Resident CTAs/SM | Resident warps/SM | Theoretical occupancy | Static limiter |
|---|---:|---:|---:|---:|---:|---:|---|
| LUT DP4A, CTA4 | 128 | 40 | 2,048 B | 12 | 48 | 100% | threads/warps/registers |
| LUT DP4A, CTA8 | 256 | 40 | 2,048 B | 6 | 48 | 100% | threads/warps/registers |
| table-free DP4A, CTA8 | 256 | 40 | 0 B | 6 | 48 | 100% | threads/warps/registers |
| FP16 tensor-core prototype | 256 | 48 | 46,080 B | 2 | 16 | 33.3% | shared memory |
| exact IMMA single | 256 | 64 | 4,096 B | 4 | 32 | 66.7% | registers |
| exact IMMA pair | 256 | 62 | 4,096 B | 4 | 32 | 66.7% | registers |

These counts use the published architectural limits and recorded allocation defaults. Because the audited LUT kernels use a CTA barrier, promotion also requires the exact barrier micro-constant or a matching CUDA runtime occupancy result. Once that gate passes, the table immediately proves that the CTA4/CTA8 production choice is not an occupancy-percentage choice. Both widths can fill all 48 resident warp slots for the 40-register LUT kernel. CTA4 provides more resident CTAs; CTA8 provides more cooperating warps per output tile. Which organization wins depends on execution details and launch geometry.

The script also emits explicitly labeled synthetic resource rows for fixed multiplicities. They exist to exercise cliffs in the CPU simulator. They are not production `ptxas` facts.

## 3. What a roofline model proves

Let a static model provide operation count `F_k`, transferred bytes `Q_k`, a peak operation rate `P`, sustainable bandwidth `B`, and launch floor `L_k`. Then

\[
\underline T_k= L_k+\max\left(\frac{F_k}{P},\frac{Q_k}{B}\right)
\]

is a lower bound under the model's accounting assumptions. More detailed variants can add hard issue-pipe bounds and critical-path lower bounds.

This supports three valid uses:

- reject an impossible launch or resource configuration;
- identify a physical lower bound on latency;
- prune candidate `k` only when its lower bound is already no better than a measured incumbent upper bound, for example
  \[
  \underline T_k\ge U_{\rm incumbent}-\epsilon.
  \]

A lower roofline value does not prove a faster kernel. Actual latency may exceed the bound because the model omits dependency and queuing effects.

### 3.1 Non-identifiability theorem

Let `A(k)` be any analytical summary containing only operation count, byte count, instruction count, and theoretical occupancy. Suppose two kernels satisfy

\[
A(k_1)=A(k_2).
\]

No dispatcher that observes only `A` can determine their measured ordering for every legal machine state.

**Proof.** Construct two admissible timing worlds. In world `W_1`, set the omitted dependency and memory-level-parallelism penalties so that

\[
T(k_1)<T(k_2).
\]

In world `W_2`, exchange those omitted penalties while preserving all fields in `A`, giving

\[
T(k_2)<T(k_1).
\]

The analytical input is identical in both worlds, so any rule based only on `A` returns the same answer in both and is wrong in one. QED.

The result also holds when the roofline lower bounds differ. The kernel with the lower bound can incur an arbitrarily larger omitted penalty.

### 3.2 Concrete equal-count reversal

The CPU artifact defines two kernels with the same loads, arithmetic, reductions, register reservation, byte count, instruction count, and theoretical occupancy.

- `memory_parallel` schedules independent loads early, increasing outstanding memory requests, then uses a long arithmetic dependency chain.
- `compute_parallel` consumes loads more serially but rotates through independent arithmetic accumulators.

The synthetic measured latencies are:

| Cache state | memory-parallel | compute-parallel | Winner |
|---|---:|---:|---|
| cold DRAM | 11.7 us | 14.2 us | memory-parallel |
| hot L2 | 7.4 us | 6.6 us | compute-parallel |

Cold data rewards memory-level parallelism. Hot data reduces the load-latency penalty and exposes the arithmetic dependency chain. The common roofline point cannot encode this reversal.

### 3.3 Real analytically favored loser

The repository supplies a stronger measured example. The table-free E2M1 decoder and the LUT DP4A decoder both compiled to 40 registers per thread with no spills. The table-free arm used zero shared bytes; the LUT arm used 2,048 shared bytes plus a CTA barrier. A resource-only analysis therefore favors or ties the table-free arm.

Seven serialized measurements gave medians:

\[
T_{\rm LUT}=12.332\ \mu s,
\qquad
T_{\rm tablefree}=15.443\ \mu s.
\]

The analytically favored arm was 25.2% slower. The test suite asserts this reversal. The values remain historical measurements, not outputs of the analytical model.

A second CPU test places an analytically favored CTA8 arm 0.55 us behind CTA4 and feeds the direct procedure the same heavy-tailed, autocorrelated ABBA/BAAB process used elsewhere. Across 20 fixed test seeds, the procedure completed a direct certificate and selected CTA4 every time. The generated evaluation repeats this stress test over 100 seeds and labels the resulting frequency as empirical rather than a substitute for the confidence-sequence proof.

## 4. Paired sequential best-arm certification

### 4.1 Estimand and superblock

For one state `x` and two kernels `a,b`, use a randomized four-run superblock:

\[
ABBA \quad\text{or}\quad BAAB,
\]

chosen independently before the block. Each arm appears at the same mean time position, 2.5. If common latency drift within the block is affine, `c_0+c_1 t`, it cancels exactly in the arm mean difference.

Define the block contrast

\[
D_t=\frac{T_{b,1}+T_{b,2}}{2}-\frac{T_{a,1}+T_{a,2}}{2}.
\]

The implementation uses `a=CTA4`, `b=CTA8`, so negative values favor CTA8. The ABBA/BAAB order protects against linear drift and exposes order sensitivity. It does not assume the four raw runs are independent.

Heavy stalls make unbounded finite-sample inference difficult. The certificate therefore names a contrast cap `c` and uses

\[
\widetilde D_t=\operatorname{clip}(D_t,-c,c).
\]

The estimand is the conditional mean of this clipped contrast. Timeout and clipping rates must be reported. A campaign with frequent clipping has chosen too small a cap or an unstable regime.

### 4.2 Dependence assumption

Let `F_{t-1}` include all previous measurements, model fits, selected states, and stopping decisions. For each comparison `q`, assume the selected subsequence satisfies

\[
E[\widetilde D_{q,t}\mid F_{t-1}]=\Delta_q,
\qquad
\widetilde D_{q,t}\in[-c,c].
\]

This is a stable conditional-mean assumption for a named machine regime. It permits heavy tails before clipping, autocorrelation, and adaptive selection of which state to measure next. It is weaker than IID. It is invalid after a regime change, which is why clocks, compiler output, and device identity are certificate fields.

### 4.3 Time-uniform confidence sequence

For one comparison, Azuma-Hoeffding gives at a fixed sample count `n`:

\[
P\left(\left|\overline D_n-\Delta\right|\ge r\right)
\le 2\exp\left(-\frac{nr^2}{2c^2}\right).
\]

Allocate

\[
\alpha_{q,n}=\frac{6\alpha_q}{\pi^2 n^2}.
\]

Since `sum_n alpha_{q,n}=alpha_q`, a union bound over every possible stopping time gives the simultaneous radius

\[
\boxed{
 r_q(n)=c\sqrt{\frac{2}{n}
 \log\left(\frac{\pi^2n^2}{3\alpha_q}\right)}
 }.
\]

Thus, with probability at least `1-alpha_q`, the interval

\[
[\overline D_n-r_q(n),\overline D_n+r_q(n)]
\]

contains `Delta_q` for every `n`. Optional stopping and adaptive resampling do not invalidate it.

For `Q` state/kernel comparisons, set `alpha_q=alpha/Q`. A union bound gives family-wise coverage at least `1-alpha`. A sharper multiple-testing method may replace Bonferroni, but it must remain valid under sequential looks.

### 4.4 Stopping rule

For CTA8 minus CTA4 contrast interval `[L,U]`:

- CTA8 is `epsilon`-optimal if `U <= epsilon`;
- CTA4 is `epsilon`-optimal if `L >= -epsilon`;
- if both hold, either is valid, and the dispatcher retains the incumbent or lower-pressure arm;
- otherwise the state remains unresolved.

For more than two kernels, directly compare a candidate `k` with every surviving challenger `j`. Select `k` only when

\[
U(T_k-T_j)\le\epsilon
\quad\text{for all }j.
\]

The raw paired data should be retained so a new incumbent can be re-anchored without discarding earlier observations.

Exact best-arm identification uses `epsilon=0`. That is often uneconomic for microsecond near-ties. In the historical weighted-down table, B=1 and B=2 have median gaps of 0.001 and 0.007 us. An `epsilon=0.1 us` certificate can rationally retain CTA4 without spending thousands of blocks to learn a practically irrelevant sign.

### 4.5 Structural borrowing

Neighboring multiplicities and geometries can share structure:

- monotone gap curves;
- a fixed piecewise-constant segmentation;
- a low-rank machine/state/kernel latency matrix.

There are two distinct uses.

**Model-certified pooling.** If the structural restriction is externally justified and fixed before seeing timings, pooled observations can shrink intervals. The confidence calculation must be derived under that exact model. Selecting the number of segments or rank from the same data requires a selective-inference correction.

**Safe structured allocation.** Fit any useful model to predict difficult states and choose the next measurement, but stop only on the direct per-state confidence sequences above. This preserves the family-wise guarantee under arbitrary model misspecification. In a two-arm, all-states-must-be-certified table, allocation advice may not reduce total samples; its larger benefit appears with more arms, where it can avoid measuring implausible challengers.

The artifact labels monotone, piecewise, and low-rank model-only results as non-certificates. `safe_low_rank` uses the low-rank fit only for ordering and retains direct stopping.

### 4.6 Synthetic experiment

The simulator emits raw four-run latencies with:

- a global AR(1) hidden state;
- Student-t shocks;
- random within-block drift;
- rare positive stalls;
- randomized ABBA/BAAB order.

The evaluated design used `alpha=0.05`, `epsilon=0.12 us`, contrast cap `0.75 us`, and 24 states. The checked-in summary uses three Monte Carlo repetitions per cell, so the percentages are illustrative rather than precision estimates.

Selected results:

| Truth | Policy | Mean superblocks | Family exact-misselection | Family epsilon violation | Mean simple regret |
|---|---|---:|---:|---:|---:|
| low-rank + piecewise | independent direct | 732 | 0% | 0% | 0 |
| low-rank + piecewise | piecewise model | 176 | 0% | 0% | 0 |
| low-rank + piecewise | low-rank model | 96 | 0% | 0% | 0 |
| monotone | independent direct | 4,492 | 0% | 0% | 0 |
| monotone | monotone model | 1,256 | 0% | 0% | 0 |
| monotone | low-rank model | 1,152 | 100% | 100% | 0.0524 us |
| misspecified local-like | independent direct | 3,228 | 0% | 0% | 0 |
| misspecified local-like | safe low-rank allocation | 3,188 | 0% | 0% | 0 |
| misspecified local-like | monotone model | 2,956 | 100% | 100% | 0.0835 us |
| misspecified local-like | piecewise model | 840 | 33.3% | 33.3% | 0.00987 us |
| misspecified local-like | low-rank model | 896 | 100% | 100% | 0.1459 us |

The structured models can be dramatically cheaper when correct. The same compression creates large family error under localized sign changes. The safe method does not claim a sample reduction in this two-arm case; it preserves the direct certificate.

### 4.7 Economic stopping

Let unresolved state `x` be called `N_x` more times during the expected lifetime. If incumbent `i` and challenger `j` have simultaneous intervals, an upper bound on possible per-call saving is

\[
V_x^{\rm call}=\max(0,U_i-L_j).
\]

The maximum remaining lifetime value is

\[
V_{\rm remain}=\sum_x N_x V_x^{\rm call}.
\]

Stop tuning and retain the incumbent when

\[
V_{\rm remain}\le C_{\rm compile}+C_{\rm timing}+C_{\rm deployment}.
\]

This is part of the stopping certificate. An exact-sign campaign for a 0.001 us difference should normally fail this return-on-investment test.

## 5. Robust machine-specific dispatch

### 5.1 Interval regret

Suppose machine `m`, state `x`, kernel `k` has a simultaneous latency interval

\[
T_{m,x,k}\in[L_{m,x,k},U_{m,x,k}].
\]

A valid upper bound on regret from selecting `k` is

\[
R^U_{m,x}(k)=
\max\left(0,
U_{m,x,k}-\min_{j\ne k} L_{m,x,j}
\right).
\]

This is the exact worst-case simple regret over the rectangular simultaneous
interval set. With only one legal kernel, the regret is zero. Including the
selected arm in the inner minimum would add an artificial interval-width
penalty and is therefore incorrect.

The machine-specific minimax dispatcher chooses

\[
k^*_{m,x}=\arg\min_k R^U_{m,x}(k).
\]

This rule remains conservative when intervals overlap. It can retain a lower-code-cost incumbent as a tie-breaker.

### 5.2 Distributional shift over machines

Let `p_0` be a nominal distribution over target machines and let

\[
\mathcal P=\{p:\|p-p_0\|_1\le\rho\}.
\]

A portable table must use one kernel `k` for every machine. Its distributionally robust objective is

\[
\min_k\sup_{p\in\mathcal P}
\sum_m p_m R^U_{m,x}(k)+C_{\rm code}(k,x).
\]

For a finite machine set, the inner maximum moves up to `rho/2` probability mass from low-regret machines to high-regret machines. The tool implements that transfer exactly.

If the remote machine has no interval, its uncertainty is not zero. With only a timeout cap, its interval regret can be as large as that cap. This is why the artifact does not install the local table as a remote result.

### 5.3 When separate tables dominate

Let `k_p(x)` be the portable choice and `k_m(x)` the machine-specific choices.
For each machine/state pair, obtain a simultaneous lower confidence bound
`G^L_{m,x}` on

\[
T_{m,x,k_p}-T_{m,x,k_m}.
\]

A direct paired contrast is preferred. If only simultaneous marginal latency
intervals are available, the conservative bound

\[
G^L_{m,x}=L_{m,x,k_p}-U_{m,x,k_m}
\]

is valid. Construct the machine-specific table to retain the portable kernel
whenever this lower bound is non-positive. For that fallback construction,
separate tables are certified to dominate actual lifetime latency when

\[
\sum_{m,x}N_{m,x}
\max(0,G^L_{m,x})
>
C_{\rm tune}+C_{\rm table}+C_{\rm icache}+C_{\rm dispatch}.
\]

If a deployed machine-specific table uses a different kernel even where
`G^L_{m,x} <= 0`, the certificate must sum the signed lower bounds instead of
clipping them at zero.

Comparing the portable and separate worst-case regret objectives is also useful
for selecting a robust policy. The difference of two regret upper bounds is not
a lower confidence bound on realized latency saving, so it must not be used as
the lifetime-return certificate above.

If both CTA4 and CTA8 cubins are already present, a second static table costs only data bytes and a machine-fingerprint branch. Code-size and instruction-cache cost can be nearly zero. Separate tables then dominate at modest call volume whenever their measured choices differ. If separate tuning adds new specialized kernels, use their text size, cold instruction-fetch cost, and module-load time in the right-hand side.

A single portable table is preferable when measured optima agree, call volume is too low to repay tuning, or the confidence intervals are too wide to establish a material machine-specific gain.

## 6. Joint compile and measurement search

Register count, stack use, shared bytes, and spills are compilation outcomes. Treating source variants as fixed arms before compilation is incorrect.

### 6.1 Finite manifest

Start with a finite manifest `C` of compile candidates. Each entry contains source/template identity, flags, launch bounds, optional max-register cap, and expected code-size cost. No rule may append candidates indefinitely.

Compiling candidate `c` produces metadata

\[
M_c=(r_c,s_c,h_c,\text{stack}_c,\text{spill}_c,\text{cubin hash}_c).
\]

The parser ingests stable `ptxas -v` fields. The exact occupancy calculator then classifies launch feasibility. A policy may hard-reject spills for latency-critical variants or retain them as measurable arms.

### 6.2 Bounds and elimination

For every candidate, maintain an analytical lower bound `ell_c`. It may be weak, but it must be a lower bound. After measurement, maintain a simultaneous interval `[L_c,U_c]`.

Relative to incumbent upper bound `U_i`, candidate `c` is eliminated when

\[
\max(\ell_c,L_c)\ge U_i-\epsilon.
\]

An uncompiled candidate can be statically eliminated by the same rule. If its lower bound is too weak, compile it. Since the manifest is finite, compiling every survivor still terminates.

### 6.3 Stopping certificate

At termination every manifest entry is recorded as one of:

- infeasible after compile;
- spill-gate rejected;
- statically eliminated;
- statistically eliminated;
- selected;
- abandoned by the lifetime-value rule.

For unresolved candidate `c`, define the maximum lifetime value

\[
V_c^U=N\max(0,U_i-\ell_c)
\]

and include its remaining compile, measurement, and deployment work in
`C_c`. Candidate `c` may be economically abandoned when `V_c^U <= C_c`.
The checked-in demonstration classifies the table-free arm as
measured-eliminated, rejects a spilling arm, and abandons one optimistic
uncompiled arm because its maximum lifetime value cannot repay its remaining
work. This is a stopping certificate, not an open-ended autotuning loop.

Changing compiler version, flags, launch bounds, resource metadata, or cubin hash invalidates a speed certificate even when theoretical occupancy is unchanged. The SASS dependency graph and instruction mix may have changed.

## 7. Exact small-DAG schedule

A DFlash block can make per-kernel greedy dispatch wrong. Fusion can remove launches, preserve a cache-resident intermediate, or avoid synchronization. It can also increase transient workspace.

### 7.1 Model

Let the small DAG have node set `V`. An action `a` covers a legal fused subset `F_a subseteq V` and has:

- kernel latency `tau_a`;
- launch count `l_a`;
- synchronization cost `s_a`;
- transient workspace `w_a`;
- output cache tag `c_a`.

Completed nodes with unfinished successors keep their output buffers live. Let `W(D)` be live bytes after completed set `D`. An action is legal when it covers no completed node and every predecessor external to `F_a` is in `D`.

The conservative peak-workspace check is

\[
W(D)+w_a+\text{escaping outputs}(F_a)\le W_{\max}.
\]

### 7.2 Dynamic program

Use state `(D,c)`, where `D` is a completed-node bitmask and `c` is the current cache tag. The recurrence is

\[
J(D,c)=\min_{a\in A(D)}
\left[
\tau_a+l_a L+s_a+q(c,c_a)+J(D\cup F_a,c_a)
\right],
\]

subject to the workspace check. `q` is a measured or modeled cache-transition
cost. Every action strictly enlarges `D`, so the state graph is acyclic even if
`q` contains a measured cache credit. The implementation evaluates states in
completed-node-count order and records the peak workspace. Complexity is

\[
O(2^{|V|}|C||A|)
\]

up to priority-queue factors, appropriate for a small block DAG.

### 7.3 Demonstration

With a 24 MiB workspace cap and 2.75 us launch overhead, the exact synthetic schedule is:

1. `quantize_cta4`;
2. `fused_gate_activation_cta8`;
3. `down_cta8`;
4. `accumulate`.

It costs 23.92 us and peaks at 20 MiB. The best schedule restricted to single-node actions costs 27.80 us. Exact DAG optimization saves 3.88 us, about 14.0%, despite selecting a mixture of CTA widths.

For a decomposed approximation, suppose each local choice is within `epsilon_i` of its isolated optimum and every boundary interaction has range at most `Gamma_i`, including any lost fusion benefit. Then

\[
C_{\rm decomposition}-C^*
\le \sum_i\epsilon_i+\sum_i\Gamma_i.
\]

The bound is useful only when interaction ranges are measured tightly. The exact solver is preferable for the small DFlash DAG.

## 8. Emitted static table and current evidence status

The generated 4070 SUPER table is:

| Multiplicity B | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paired gate/up | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| down store | 4 | 4 | 8 | 4 | 4 | 4 | 4 | 4 |
| packed down store | 4 | 4 | 8 | 4 | 4 | 4 | 4 | 4 |
| weighted down | 4 | 4 | 8 | 8 | 4 | 4 | 4 | 4 |

The historical paired median `CTA8-CTA4` differences, in microseconds, are retained in the JSON certificate:

```text
B:                  1       2       3       4       5       6       7       8
gate/up pair:   -0.378  -0.328  -0.702  -0.205  -0.267  -0.453  -0.271  -1.276
down store:     +0.561  +0.091  -0.323  +0.188  +1.008  +0.498  +0.411  +0.608
down weighted:  +0.001  +0.007  -0.049  -0.033  +0.651  +0.441  +0.479  +0.344
```

Certificate status:

- 4070 SUPER: `historical_evidence_requires_raw_recertification`;
- 4070 Ti SUPER OC: `unmeasured`;
- remote table: `null`;
- family-wise error claim for the historical table: `null`.

The C++ header sets both `kRtx4070SuperTableCertified=false` and `kRtx4070TiSuperTableCertified=false`. The local arrays remain available as historical incumbents, but neither machine is presented as carrying a current family-wise certificate.

## 9. Minimal remeasurement plan

### 9.1 Fingerprint gate

Before timing, record:

- GPU UUID, SM count, driver/runtime;
- locked clocks or complete overclock/power profile;
- source commit and benchmark hash;
- nvcc/ptxas version plus complete flags;
- cubin hash;
- registers, shared bytes, stack, spills, and barriers.

If the cubin and all regime fields match a valid certificate, reuse it. A resource count match alone is insufficient.

### 9.2 Compile gate

Compile incumbent and challenger variants for every affected state. Parse `ptxas -v`, run the occupancy calculator, and compare its resident CTA count with the CUDA occupancy API. Reject launch-infeasible variants and apply the declared spill rule.

### 9.3 Sentinel block

Measure these historically narrow or boundary-sensitive states first:

```text
gate_up_pair:B4
down_store:B2
down_store:B4
down_weighted:B1
down_weighted:B2
down_weighted:B3
down_weighted:B4
```

Use at least one randomized four-run superblock per sentinel. A sign reversal or interval inconsistent with the old effect invalidates the old table and prioritizes neighboring states.

### 9.4 Full family and adaptive continuation

For the three independently timed roles, run one superblock in every `B=1..8` state: 24 superblocks, 96 timed launches. If expanded and packed store no longer share compiled behavior, treat them separately, giving 32 states and 128 timed launches.

Then sample only unresolved comparisons. A structural model may choose the order. Promotion requires a direct interval for every table entry unless a separately certified structural theorem covers it.

Stop when every state is `epsilon`-optimal at family-wise level `alpha`, or when the remaining lifetime value cannot repay more tuning.

### 9.5 New clocks and remote GPU

A clock, power, or overclock change creates a new regime. Do not frequency-scale old latencies into a confidence statement; compute-bound and memory-bound kernels can scale differently.

The overclocked 4070 Ti SUPER requires the same direct 24-state campaign. The local table may be the incumbent used for comparisons, but it is not a remote result before those intervals exist.

## 10. Reproduction

From the repository root:

```bash
python -m unittest -v tools/test_ada_dispatch_certificate.py
python tools/ada_dispatch_certificate.py occupancy \
  --threads 256 --registers 48 --static-shared 46080 --barriers 1
python tools/ada_dispatch_certificate.py evaluate \
  --output scratch/ada-dispatch-certificate --trials 3
```

The checked-in tests are hardware-free. GPU promotion still requires exactness gates, compiler metadata, runtime occupancy validation, and the paired timing campaign.

## 11. References

- NVIDIA, *CUDA Programming Guide, Compute Capabilities*, table entries for compute capability 8.9.
- NVIDIA, *Ada GPU Architecture Tuning Guide*, occupancy and shared-memory carveout sections.
- NVIDIA, *CUDA C++ Best Practices Guide*, register allocation granularity and occupancy discussion.
- NVIDIA, *Nsight Compute Profiling Guide*, launch occupancy metrics and scheduler statistics.
- NVIDIA, *Nsight Compute Occupancy Calculator Python Interface*, `get_gpu_data` allocation fields.
- Repository evidence: `audits/s11-nvfp4-direct-execution.md` and `audits/nvfp4-fp16-tc-frontier.md`.
