# Problem 6: optimal exact NVFP4 prefill schedule with finite spill memory

Repository: https://github.com/novysvet/insignia  
Branch: `codex/glm53-dflash2-4070-super`  
Hardware required: none

## Engine facts

GLM-5.3-Flash has 45 layers: 34 recurrent KDA layers and 11 causal MLA layers.
The first three MLPs are dense; the remaining 42 layers have 288 experts with
Top-8 routing. Each expert record is 13.5 MiB NVFP4. Every token at layer `l`
depends on its layer-`l-1` hidden state; KDA and MLA also impose causal state
dependencies within a layer.

The old prompt path processed fixed token chunks through all layers, repeatedly
loading experts that reappeared in later chunks. The current exact full-prompt
layer-major path spills/restores host residuals so one layer can process the
whole prompt before advancing. On a 938-token prompt it changed prefill from
157.845 s to 69.187 s (5.94 to 13.56 token/s), reduced O_DIRECT expert bytes by
77.8%, and kept logits/IDs exact. Approximately 5.56 GB of bidirectional host
spill displaced about 549.8 GiB of NVMe traffic. This is a measured point, not
a proof of global optimality.

## Mathematical problem

Given a prompt length `T`, exact per-token route sets as they become available,
host/VRAM capacities, expert and residual sizes, and symbolic service curves,
find an exact execution schedule minimizing makespan. The schedule may choose
different token tile sizes per layer, retain selected experts between tiles,
spill hidden/residual/KDA state, and overlap SSD, H2D, GPU compute, and host
spill. It cannot evaluate a router before its hidden input exists or alter any
causal/FP32 operation order.

## Proof obligations

1. Formalize the layer/token dependency DAG for dense, KDA, MLA, router, and
   MoE operations. State which operations may legally reorder and which may
   not.
2. For one sparse layer with known routes and cache capacity `C`, prove the
   minimum possible expert record reads. Characterize when one read per
   distinct expert is attainable.
3. Extend to all layers with finite residual spill. Establish hardness of the
   general tiling problem and solve a nontrivial special case exactly.
4. Derive a lower bound on prefill makespan combining:

   ```text
   unavoidable distinct expert bytes,
   host spill/restore bytes,
   H2D bytes,
   causal attention/KDA compute, and synchronization critical paths.
   ```

   Do not sum resources that can overlap; use a valid max/cut/critical-path
   argument.
5. Determine the optimality structure of tile size. Give conditions for a
   single global chunk, per-layer chunks, or a wavefront schedule to dominate.
   Construct a counterexample to the claim that the largest fitting chunk is
   always optimal.
6. Incorporate multi-row NVFP4 execution: an expert weight pass can serve all
   routed rows in a tile, but accumulator/register pressure changes compute
   cost. Jointly choose tile size and per-expert row batching.
7. Prove exactness of spill/restore and schedule transformations under the
   engine's floating-point determinism law. In particular, expert contributions
   for one row must be accumulated in original route order.
8. Add online route discovery. Bound regret of a schedule that knows only
   earlier layers/tokens relative to a clairvoyant schedule.

## CPU deliverables

- A discrete-event simulator with finite host/VRAM buffers and independently
  parameterized SSD, H2D, GPU, and spill engines.
- An exact dynamic program or mixed-integer oracle for small prompts, used to
  verify the proposed approximation.
- Synthetic workloads spanning one-hot expert reuse, uniform routing, bursty
  DFlash-like unions, and adversarial cache thrash.
- A schedule trace verifier for dependencies, capacities, and route-order
  preservation.
- A decision table mapping measured route unions and service curves to chunk
  and row-batch choices.

## Engine gate

The proposed implementation must beat the existing full-prompt layer-major
path, not the obsolete chunk-major baseline. Use short focused prompts first;
run the 938-token campaign only if the model predicts a material gain. Report
prefill wall, token/s, expert reads/bytes, host spill bytes, H2D bytes, compute
time, and exact logit/ID parity. Kill a design whose predicted improvement is
below 5% after all spill and synchronization costs.
