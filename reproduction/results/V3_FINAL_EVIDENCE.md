# V3 final evidence and publication

Execution commit: `003fdbc29040c33994d99311eb94a80c9041d768`.

Both V3 runs completed 10000 optimizer steps independently with seed 1 and batch 16. The validation rule selected step 10000 for both datasets. Full raw records are in `v3-bridge512-b16-s1-20261016/` and `v3-dacl-b16-s1-20261016/`; when best equals final, the selected test evidence is `test_final_scores.csv` and `test_final_v3.json`.

The final comparison uses `v3_shared_z_fpr_256_v3`, with z candidates 0.1%, 0.25%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, and 100%. Checkpoints, z and thresholds use validation only. V0 is reevaluated at its frozen best-validation checkpoint; no V0 training is repeated. Both V3 full runs finished before either V0 reevaluation began.

V3 combines 448 input, patch-token recentering by the same-layer CLS, and training-only augmentation. V0 retains 392 input and its original preprocessing. These results cannot isolate the contribution of any individual V3 change. Both use the historical 392-to-256 mask path and 256 anomaly-map evaluation. No native-512 appendix is included here.

Bridge has 2211 normal training images, validation 309 normal / 309 anomaly, and test 628 / 628. DACL has 413 normal training candidates, validation 103 / 1284, and test 55 / 920. DACL normal candidates mean no annotated damage label, not independently certified healthy structures. The frozen profiles and fingerprints remain authoritative in `../BENCHMARKS.md` and `../../dataset_profiles/`.

The retention decision must use validation I-AUROC and pixel AP: no decrease larger than 0.5 percentage points on either metric in either dataset, and an improvement of at least 1 percentage point on at least one of those four comparisons. Test results are reporting evidence only. No automatic next experiment is authorized by this report.

Image-stratified paired bootstrap uses 2000 resamples and seed 1, identical image identities and labels. It does not quantify uncertainty across training seeds or bridge/session groups. One training seed supports initial screening only. V0 has z selected at its frozen checkpoint; V3 jointly selects checkpoint and z at five scheduled validation points.

Final numerical tables and retention decision will be generated from completed V0 reevaluations and V3 summaries, without modifying these original records.