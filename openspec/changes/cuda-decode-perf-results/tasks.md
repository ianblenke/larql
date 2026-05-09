# cuda-decode-perf-results — tasks

## 1. Consolidate session results

- [x] 1.1 `proposal.md` records the bench progression checkpoint
      table (decode ms/tok, tok/s, prefill ms) for every
      branch shipped this session.
- [x] 1.2 Branches catalogued by category (8 wins, 5 documented
      negative results) with the mechanism for each.
- [x] 1.3 Remaining decode bucket profile + per-bucket
      optimization status.

## 2. Document next-step paths

- [x] 2.1 Path A (Marlin INT4-IMMA mmvq) — effort, risk, reward
      estimate.
- [x] 2.2 Path B (WMMA attention #2/#3) — explicit warning that
      mitigation #1 is empirically settled negative.
- [x] 2.3 Path C (speculative decoding) — multi-week, biggest
      single potential win.
- [x] 2.4 Path D (Q/K/V mmvq fusion) — quick clean win.
- [x] 2.5 Path E (norm + Q8_1 fusion) — marginal.

## 3. Recommendations

- [x] 3.1 Recommended next-session order: D → A.
- [x] 3.2 Path B explicitly de-prioritised pending non-GQA
      target or 5-10 days of mma.sync PTX work.

## 4. Archive

- [ ] 4.1 Archive after the next contributor confirms the
      navigation aid is useful.
