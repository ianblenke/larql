## ADDED Requirements

### Requirement: a session-level consolidation document SHALL exist for CUDA decode/prefill performance work

`openspec/changes/cuda-decode-perf-results/proposal.md` SHALL serve
as the navigation aid for any future contributor to the CUDA
decode/prefill performance push. It SHALL contain:

- The bench progression checkpoint table (decode ms/tok, tok/s,
  prefill ms) at every branch checkpoint and at the
  llama-cpp-turboquant reference target.
- A categorised catalogue of every shipped branch (wins vs
  documented negative results) with the mechanism for each.
- The per-bucket decode profile breakdown with optimization
  status per bucket.
- Concrete next-step paths (A through E) with effort, risk, and
  reward estimates per path.
- An explicit recommended next-session order.

#### Scenario: navigation aid lets a fresh contributor identify the highest-ROI path quickly

- **WHEN** a contributor opens
  `openspec/changes/cuda-decode-perf-results/proposal.md` cold
- **THEN** they SHALL be able to identify the highest-ROI
  tractable path (Path D for quick wins, Path A for the
  multi-day effort) within ~5 minutes of reading
<!-- test: unbacked -->
