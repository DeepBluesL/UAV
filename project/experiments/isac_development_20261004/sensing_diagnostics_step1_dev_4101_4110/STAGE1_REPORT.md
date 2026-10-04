# STEP 1 sensing-controllability diagnostic

This development diagnostic contains 140 episodes: 10 common seeds (4101–4110), two scenarios, and seven feasible path probes. Results include failed paths. Tracker-source noise is aligned by seed, but channel draws can diverge after path-dependent exits, so paired differences are descriptive rather than fully causal counterfactual estimates.

## Task and tracking results

All nominal paths succeeded in 27 s. Their mean first-10 position RMSE ranged from 1.20 to 1.26 m, with individual-seed ranges spanning roughly 0.21–8.49 m. Mean seed-paired RMSE changes versus the goal path were small: 0.00 to +0.05 m, with paired ranges crossing zero or approaching it for every path.

In the crossing scenario, the goal, altitude-plus, and altitude-minus paths all failed at the 100 s deadline. Each averaged 90 safety interventions and had no collision. Both split-altitude paths and both lateral paths succeeded in 22 s with no safety intervention or collision. Their mean first-10 RMSE values were 0.84–0.88 m versus 0.87 m for goal. Seed-paired mean changes versus goal ranged from -0.03 to +0.01 m; the widest paired range was -0.18 to +0.15 m. The 100 s bars therefore represent retained deadline failures, not valid slow arrivals.

| Scenario/path family | Success | Mean time | Mean first-10 RMSE | Mean first-10 rho | Minimum separation |
|---|---:|---:|---:|---:|---:|
| Nominal, all seven paths | 100% | 27 s | 1.20–1.26 m | 0.70–0.73 m² | 20.0 m |
| Crossing, goal and same-direction altitude offsets | 0% | 100 s deadline | 0.83–0.89 m | 0.78 m² | 5.28–5.98 m |
| Crossing, lateral and split-altitude paths | 100% | 22 s | 0.84–0.88 m | 0.78–0.79 m² | 7.46–15.00 m |

Position NEES is reported only as an offline covariance-scale diagnostic. Across path means it was about 4.8–5.1 in crossing and 8.9–10.2 in nominal. Individual episode first-10 means ranged from 2.35 to 17.77 in crossing and 1.84 to 70.39 in nominal. Correlated time steps are not independent chi-square samples, but the high and variable nominal values warrant checking posterior calibration before treating covariance as calibrated uncertainty.

## Delivered-measurement floors

Floor-hit rates are conditional on measurements actually delivered to the EKF. BS supplied 5,770 measurements per component, UAV0 2,761, and UAV1 2,268. Range was at its variance floor in 100% of delivered measurements. Range-rate floor-hit rates were 100% for BS and 99% for each UAV. Azimuth and elevation shared the same raw CRB proxy and hit the floor in 91% of BS, 95% of UAV0, and 96% of UAV1 deliveries.

The sensing model is consequently floor-limited for most delivered samples. Feasible path changes clearly resolve crossing-path task failures through geometry, but this development run does not show a strong or consistent first-10 tracking-RMSE gain. STEP 1 supports continuing with sensing-aware control as a constrained tradeoff study; it does not establish a strong sensing benefit or a causal path-to-estimation improvement.

The figures were regenerated from `episodes.csv` and `source_crb.csv` only. No environment episode was rerun.
