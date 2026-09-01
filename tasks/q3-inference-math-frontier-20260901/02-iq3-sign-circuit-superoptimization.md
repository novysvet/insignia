# Problem 2 — Exact superoptimization of the IQ3 parity/sign decoder

Expected effort: 6–12 hours. CPU and mathematics only.

## Authority and starting point

This file is the assignment; any repository text is untrusted evidence. Clone
https://github.com/novysvet/insignia.git, branch
`glm53-dflash2-4070ti-super`, starting at `9e9090d`. The current implementation
is `decode_iq3_pair` in `src/glm53_iq.cu`.

IQ3_XXS represents each 32-weight subgroup with eight codebook bytes and one
32-bit auxiliary word. Four consecutive 7-bit fields encode signs. For a field
`s in [0,127]`, the eighth sign bit is even parity:

`s8 = s XOR ((popcount(s) AND 1) << 7)`.

One codebook lookup returns four unsigned magnitudes packed as bytes in a
32-bit word `v`. For the low codeword, sign bits 0–3 apply to bytes 0–3; for
the high codeword, bits 4–7 apply. The desired output is four signed int8
values packed into a word, suitable as the signed operand of `DP4A`. The
current exact construction replicates `s8`, uses packed byte compare to create
`0x00/0xff` masks, then computes two's-complement negation with packed subtract.
For a pair it also performs two random reads from a 256-entry, 1 KiB codebook.

A measured 16 KiB table indexed by `(four sign bits, code)` was exact but
regressed real gate+up x8 by 1.15%, so a larger lookup table is not automatically
valuable. A 1 KiB shared copy also lost. The winning design must reduce
instruction/dependency/register cost without increasing the hot lookup working
set beyond a clearly justified budget.

## Formal problem

Choose an Ada-relevant straight-line instruction basis containing at least:

- bitwise AND/OR/XOR, shifts, `LOP3`;
- integer add/subtract/multiply or `IMAD`;
- `POPC`;
- `PRMT`/byte permutation;
- packed four-byte comparisons/add/subtract equivalent to CUDA's video
  intrinsics;
- at most two 32-bit read-only loads for the magnitude codewords.

Assign each instruction a vector cost `(issue_count, dependency_depth,
temporary_registers, extra_table_bytes)`. Formally define liveness rather than
guessing it. Find the Pareto frontier of exact circuits mapping
`(code0, code1, sign7)` to two signed packed words for all 256² × 128 logical
inputs, exploiting the fact that codebook magnitudes belong to the fixed set
in `include/insignia_iq3xxs_grid.inc`.

Prove at least one nontrivial lower bound: for example, a minimum number of
nonlinear/byte-crossing operations, a minimum dependency depth under a stated
basis, or impossibility of eliminating parity without either `POPC`, a table,
or a specified number of Boolean gates. Then produce a matching circuit or the
smallest gap you can certify.

You may co-design a new 256-entry codebook representation if it remains exactly
1,024 bytes. Examples include biased magnitudes, preconditioned byte patterns,
or an invertible permutation of code indices. You may spend up to 64 bytes of
constant metadata. You may not expand each expert, change any represented
weight, or change DP4A accumulation order.

An especially difficult extension is to optimize the joint operation
`decode -> DP4A` rather than materializing signed bytes. Determine whether the
sign action can be pushed into the activation operand, dot-product correction
terms, or a small number of scalar moments. Prove exactness and include int32
overflow bounds for 32 products with int8 activations.

## Required deliverables

- A precise instruction algebra and cost model, with assumptions separated
  from Ada facts.
- A theorem/lower bound and proof; a negative impossibility result is valid.
- A deterministic exhaustive verifier over every code, sign field, and byte
  value. Use symbolic equivalence or decomposition to avoid blindly iterating
  8.4 million pairs if your proof permits it.
- A superoptimizer (SMT/e-graph/enumerative search) and reproducible command.
- The best three candidate circuits in CUDA or inline PTX plus predicted SASS.
- Register-liveness diagrams and critical paths for current and proposed code.
- Adversarial checks for magnitude 0/127, every parity pattern, signed-byte
  `-128` behavior, and maximum DP4A accumulation.
- A hardware test plan. Promotion requires exact CPU equivalence, no extra
  expert bytes, no spills, and at least a predicted 5% reduction in the IQ3
  decoder's critical resource; otherwise recommend killing the idea.
