# smallgwn TODO

## H-PLOC efficiency and quality

- [x] Fix H-PLOC `uint32 Morton` correctness/performance issues (tie-break behavior and sort bit range).
- [x] Expand H-PLOC test coverage across unit/integration/facade/query paths.
- [x] Add H-PLOC vs LBVH performance regression gate for topology build ratio.
- [ ] Defer single-kernel top-down wide conversion (paper Section 3.4) for now.
- [ ] Defer explicit stream-synchronization reductions in H-PLOC/preprocess paths for now.

## General backlog

- [ ] Implement SAH optimization pass (post-LBVH / post-collapse), currently missing.
- [ ] Remove Eigen dependency from core library (keep Eigen bridge optional / adapter-only).
- [ ] Add benchmark case comparing LBVH-only vs LBVH+SAH once SAH is implemented.
