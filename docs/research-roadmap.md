# Long-term strengthening plan

Preserve the current champion. A lower training loss is not evidence of stronger play.

1. Freeze baseline source/model/data identities and diagnose losses using full histories.
2. Compare one search, clock or backend change at a time under equal compute/time budgets.
3. Compare teacher-supervised policy/value training with a teacher-only control.
4. Evaluate self-play RL with actual search visit targets, correct side-to-move outcomes,
   censored-game handling, and separate game-group train/validation/test partitions.
5. Select on development data only. Freeze one candidate for independent paired evaluation.
6. Publish negative results and failures. Require rule/time safety and independent review;
   never automatically deploy a candidate.

Keep prior diagnostic positions and mirrored equivalents out of training/held-out reuse.
Freeze seed, resource limits, stop rules, dependencies and all input hashes before each run.
CPU orchestration and arenas are separate from finite GPU training batches.
GPU ownership/exclusion must be freshly verified; utilization alone does not establish ownership.
No paid resources, external data upload or change to live matches is implied by this plan.

Current baseline audit: no independently verified strength gain and no new model promotion.
Native worker-containment tests are engineering validation, not GPU training or playing-strength evidence.
