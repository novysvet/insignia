#!/usr/bin/env python3
# Generates the cut-verify (tail-skip) draft patches under scratch/seqverify.
# Reads the staged copy of src/glm53_generate.cu, applies anchored edits, and
# emits unified diffs. Never touches the real src/ tree.
import pathlib, sys

root = pathlib.Path(__file__).resolve().parent
src_path = root / "src" / "glm53_generate.cu"
text = src_path.read_text(encoding="utf-8")

def replace_once(text, old, new, tag):
    n = text.count(old)
    if n != 1:
        sys.exit(f"anchor {tag!r} matched {n} times (need exactly 1)")
    return text.replace(old, new, 1)

# ---------------------------------------------------------------- H1: counters
old = """                int rounds = 0, verified_rounds = 0, empty_rounds = 0;
                std::array<int, 8> accept_hist{};
"""
new = """                int rounds = 0, verified_rounds = 0, empty_rounds = 0;
                std::array<int, 8> accept_hist{};
                int cut_rounds = 0, cut_cont_rows = 0;
"""
text = replace_once(text, old, new, "H1 counters")

# ------------------------------------------------------------ H2: cut env knob
old = """                const bool adaptive_k_on = [] {
                    const char *value = std::getenv("INSIGNIA_GLM53_DF_ADAPTIVE_K");
                    return !value || std::atoi(value) != 0;
                }();
                double accept_ema = 0.0;
"""
new = """                const bool adaptive_k_on = [] {
                    const char *value = std::getenv("INSIGNIA_GLM53_DF_ADAPTIVE_K");
                    return !value || std::atoi(value) != 0;
                }();
                // Trusted-prefix cut for batch verify (tail-skip I/O). The
                // batched verify pass stages the per-layer expert union of ALL
                // drafted rows before any acceptance decision exists; when a
                // round rejects at row m, every record only the tail selected
                // was read for nothing. The cut runs the batched head on the
                // first `prefix` rows only (union I/O d(prefix)), and when the
                // whole head accepts -- and only then -- finishes the round
                // row-at-a-time from the boundary state, so the rejected
                // tail's experts are never read in any mode. Records per
                // verified round become d(max(prefix, min(m, draft_k)))
                // instead of d(draft_k).
                //   unset  = adaptive: prefix = clamp(int(accept_ema)+1, 2, k)
                //   N >= 1 = forced head size N every round
                //   -1     = off (legacy batch / row-sequential selection)
                // Forced DF_SEQ_VERIFY / DF_BATCH_VERIFY also disable it so
                // the old A/B arms stay byte-comparable.
                int cut_prefix_forced = 0;
                if (const char *value = std::getenv("INSIGNIA_GLM53_DF_CUT_PREFIX"))
                    cut_prefix_forced = std::atoi(value);
                double accept_ema = 0.0;
"""
text = replace_once(text, old, new, "H2 cut knob")

# ------------------------------------------------- H3: per-round head selection
old = """                    if (position + 1 >= insignia::glm53::DFlash2Drafter::kMaxCtx) break;
                    int round_verify_k = verify_k;
                    if (adaptive_k_on && accept_ema_init)
                        round_verify_k = std::clamp(int(accept_ema * 1.3) + 1, 2, verify_k);
                    const int draft_k = std::min(round_verify_k, generate - int(generated.size()));
"""
new = """                    if (position + 1 >= insignia::glm53::DFlash2Drafter::kMaxCtx) break;
                    const bool cut_available =
                        df_verify_mode_env == 0 &&
                        (cut_prefix_forced > 0 ||
                         (cut_prefix_forced == 0 && accept_ema_init));
                    int round_verify_k = verify_k;
                    if (adaptive_k_on && accept_ema_init && !cut_available)
                        round_verify_k = std::clamp(int(accept_ema * 1.3) + 1, 2, verify_k);
                    const int draft_k = std::min(round_verify_k, generate - int(generated.size()));
                    // Trusted head size for this round. With the cut active
                    // the EMA no longer shrinks the draft length (tail rows
                    // only cost sequential forwards after a fully accepted
                    // head); it picks the head instead. prefix == draft_k
                    // degenerates to today's plain batch path.
                    int prefix = draft_k;
                    if (cut_available)
                        prefix = cut_prefix_forced > 0
                                     ? std::min(cut_prefix_forced, draft_k)
                                     : std::clamp(int(accept_ema) + 1, 2, draft_k);
"""
text = replace_once(text, old, new, "H3 head size")

# --------------------------------------------------- H4: seq gating + cut_used
old = """                    bool df_seq_verify =
                        df_verify_mode_env == 1 ||
                        (df_verify_mode_env == 0 && accept_ema_init &&
                         accept_ema < 0.70 * draft_k);
"""
new = """                    bool df_seq_verify =
                        df_verify_mode_env == 1 ||
                        (df_verify_mode_env == 0 && prefix == draft_k &&
                         accept_ema_init && accept_ema < 0.70 * draft_k);
                    // The cut subsumes the adaptive row-sequential pick: a
                    // cut round IS a small batch head plus a row-sequential
                    // tail that only runs after full head acceptance, so the
                    // low-acceptance regime routes here instead of seq mode.
                    const bool cut_used = prefix < draft_k && !df_seq_verify;
"""
text = replace_once(text, old, new, "H4 seq gating")

# --------------------------------------------------------- H5: the cut branch
old = """                        runner.capture_offset_ = 0;
                        runner.verify_may_rollback_ = true;
                    } else {
                        const std::vector<int> verify_candidates(
"""
new = """                        runner.capture_offset_ = 0;
                        runner.verify_may_rollback_ = true;
                    } else if (cut_used) {
                        // Trusted-prefix cut: batch the first `prefix` rows
                        // (dense compute amortized, expert union d(prefix)),
                        // then either stop at the in-head first mismatch or,
                        // on full head acceptance, continue row-at-a-time.
                        // A row is forwarded only after its candidacy is
                        // proven (arg[matched-1] == candidates[matched]), so
                        // after a full head the recurrent state already
                        // stands at the accepted boundary and the sequential
                        // tail never needs a rollback.
                        ++cut_rounds;
                        const std::vector<int> head(candidates.begin(),
                                                    candidates.begin() + prefix);
                        const std::pair<int, std::vector<int>> verdict =
                            runner.verify_round(head, position + 1);
                        for (int r = 0; r < prefix; ++r)
                            arg[size_t(r)] = verdict.second[size_t(r)];
                        while (matched < prefix &&
                               arg[size_t(matched - 1)] == candidates[size_t(matched)])
                            ++matched;
                        if (matched == prefix) {
                            // Full head acceptance: the KDA state, conv
                            // history and drafter captures are exactly at the
                            // p+prefix boundary. Continue row-sequentially;
                            // every forwarded row was already accepted, so
                            // the state stays at the boundary for the tail
                            // too. Snapshots are skipped for these single-row
                            // prefills (they can never roll back).
                            runner.verify_may_rollback_ = false;
                            while (matched < draft_k &&
                                   arg[size_t(matched - 1)] ==
                                       candidates[size_t(matched)]) {
                                runner.capture_offset_ = matched;
                                arg[size_t(matched)] = runner.verify_token(
                                    candidates[size_t(matched)], position + 1 + matched);
                                ++matched;
                                ++cut_cont_rows;
                            }
                            runner.capture_offset_ = 0;
                            runner.verify_may_rollback_ = true;
                        }
                        // matched < prefix here means the head itself
                        // rejected: the shared rollback below replays the
                        // accepted prefix from the round snapshot, exactly
                        // like the plain batch path.
                    } else {
                        const std::vector<int> verify_candidates(
"""
text = replace_once(text, old, new, "H5 cut branch")

# ------------------------------------------------------- H6: rollback gating
old = """                    if (matched < draft_k && !df_seq_verify)
                        runner.rollback_kda(matched, position + 1);
"""
new = """                    if (matched < draft_k && !df_seq_verify &&
                        !(cut_used && matched >= prefix))
                        runner.rollback_kda(matched, position + 1);
"""
text = replace_once(text, old, new, "H6 rollback gating")

# --------------------------------------------------------- H7: summary print
old = """                std::printf("  accepted histogram");
                for (int count = 0; count <= verify_k; ++count)
                    if (accept_hist[size_t(count)])
                        std::printf(" %d:%d", count, accept_hist[size_t(count)]);
                std::printf("\\n");
"""
new = """                std::printf("  accepted histogram");
                for (int count = 0; count <= verify_k; ++count)
                    if (accept_hist[size_t(count)])
                        std::printf(" %d:%d", count, accept_hist[size_t(count)]);
                std::printf("\\n");
                if (cut_rounds)
                    std::printf("  cut verify used on %d/%d verified rounds "
                                "(%d sequential continuation rows)\\n",
                                cut_rounds, verified_rounds, cut_cont_rows);
"""
text = replace_once(text, old, new, "H7 summary print")

primary = root / "glm53_generate.cut.cu"
primary.write_text(text, encoding="utf-8", newline="\n")
print("primary patched copy written:", primary)

# ---------------------------------------------------------------------------
# Waves fallback: applies ON TOP of the primary. Replaces the sequential
# continuation with batched wave continuations (INSIGNIA_GLM53_DF_CUT_WAVES).
# ---------------------------------------------------------------------------
old = """                        if (matched == prefix) {
                            // Full head acceptance: the KDA state, conv
                            // history and drafter captures are exactly at the
                            // p+prefix boundary. Continue row-sequentially;
                            // every forwarded row was already accepted, so
                            // the state stays at the boundary for the tail
                            // too. Snapshots are skipped for these single-row
                            // prefills (they can never roll back).
                            runner.verify_may_rollback_ = false;
                            while (matched < draft_k &&
                                   arg[size_t(matched - 1)] ==
                                       candidates[size_t(matched)]) {
                                runner.capture_offset_ = matched;
                                arg[size_t(matched)] = runner.verify_token(
                                    candidates[size_t(matched)], position + 1 + matched);
                                ++matched;
                                ++cut_cont_rows;
                            }
                            runner.capture_offset_ = 0;
                            runner.verify_may_rollback_ = true;
                        }
"""
new = """                        if (matched == prefix) {
                            // Full head acceptance: the KDA state, conv
                            // history and drafter captures are exactly at the
                            // p+prefix boundary. Finish the tail either
                            // row-at-a-time (default) or in batched waves
                            // (INSIGNIA_GLM53_DF_CUT_WAVES) that keep the
                            // dense compute amortized. Waves MUST run with
                            // snapshots suppressed: a mid-wave reject rolls
                            // back to the HEAD snapshot and replays rows
                            // 0..matched-1 from the archive (that is exactly
                            // what rollback_kda does). A wave taking its own
                            // snapshot would make that replay double-apply
                            // the head rows.
                            static const bool cut_waves =
                                std::getenv("INSIGNIA_GLM53_DF_CUT_WAVES") != nullptr;
                            runner.verify_may_rollback_ = false;
                            if (!cut_waves) {
                                while (matched < draft_k &&
                                       arg[size_t(matched - 1)] ==
                                           candidates[size_t(matched)]) {
                                    runner.capture_offset_ = matched;
                                    arg[size_t(matched)] = runner.verify_token(
                                        candidates[size_t(matched)], position + 1 + matched);
                                    ++matched;
                                    ++cut_cont_rows;
                                }
                            } else {
                                int done = prefix;
                                while (done < draft_k &&
                                       arg[size_t(done - 1)] ==
                                           candidates[size_t(done)]) {
                                    const int wave = std::min(draft_k - done, prefix);
                                    runner.capture_offset_ = done;
                                    const std::vector<int> rows(
                                        candidates.begin() + done,
                                        candidates.begin() + done + wave);
                                    const std::pair<int, std::vector<int>> wv =
                                        runner.verify_round(rows, position + 1 + done);
                                    for (int r = 0; r < wave; ++r)
                                        arg[size_t(done + r)] = wv.second[size_t(r)];
                                    // Wave row 0 was pre-accepted by the loop
                                    // entry check (arg[done-1] ==
                                    // candidates[done]).
                                    int wmatched = 1;
                                    while (wmatched < wave &&
                                           arg[size_t(done + wmatched - 1)] ==
                                               candidates[size_t(done + wmatched)])
                                        ++wmatched;
                                    matched = done + wmatched;
                                    cut_cont_rows += wmatched;
                                    done += wmatched;
                                    if (wmatched < wave) {
                                        // Mid-wave reject: restore the head
                                        // snapshot and replay the accepted
                                        // prefix; every accepted row of this
                                        // round is in the archive.
                                        runner.rollback_kda(matched, position + 1);
                                        break;
                                    }
                                }
                            }
                            runner.capture_offset_ = 0;
                            runner.verify_may_rollback_ = true;
                        }
"""
text = replace_once(text, old, new, "W1 waves continuation")
waves = root / "glm53_generate.cut-waves.cu"
waves.write_text(text, encoding="utf-8", newline="\n")
print("waves patched copy written:", waves)
