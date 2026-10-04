# STEP 2 sensing-aware planning pilot

This pilot used development seed 4101 only. It is a timing and behavior check, not evidence for selecting a final method. All failures remain in `episodes.csv`. Target truth was used only by the simulator and offline metrics; sensing-aware planners received the public EKF belief.

| Scenario | Method | Success/time | First-10 RMSE | Late RMSE | Communication | Path | Planner ms/step |
|---|---|---:|---:|---:|---:|---:|---:|
| nominal | Goal | yes / 27 s | 0.3147 | 0.2766 | 1.00 | 235.0 m | 0.017 |
| nominal | SA navigation | yes / 27 s | 0.3147 | 0.2766 | 1.00 | 235.0 m | 4.293 |
| nominal | SA sensing 0.2 | yes / 27 s | 0.3147 | 0.2766 | 1.00 | 235.0 m | 31.762 |
| nominal | SA sensing 20 | no / 100 s | 0.2615 | 0.1828 | 1.00 | 520.4 m | 28.797 |
| nominal | MPC navigation | yes / 27 s | 0.3147 | 0.2766 | 1.00 | 235.0 m | 0.344 |
| nominal | MPC sensing 20 | yes / 27 s | 0.3147 | 0.2766 | 1.00 | 235.0 m | 38.380 |
| nominal | MPC sensing 2000 | yes / 27 s | 0.2937 | 0.2413 | 1.00 | 241.6 m | 48.352 |
| crossing | Goal | no / 100 s | 0.2894 | 0.2913 | 0.80 | 100.0 m | 0.016 |
| crossing | SA navigation | yes / 34 s | 0.2623 | 0.2519 | 1.00 | 279.9 m | 4.358 |
| crossing | SA sensing 0.2 | yes / 34 s | 0.2623 | 0.2519 | 1.00 | 280.0 m | 29.534 |
| crossing | SA sensing 20 | no / 100 s | 0.2448 | 0.2253 | 1.00 | 528.0 m | 28.652 |
| crossing | MPC navigation | yes / 25 s | 0.2387 | 0.2159 | 0.90 | 229.0 m | 0.356 |
| crossing | MPC sensing 20 | yes / 25 s | 0.2387 | 0.2159 | 0.90 | 229.0 m | 42.226 |
| crossing | MPC sensing 2000 | yes / 30 s | 0.2607 | 0.2496 | 1.00 | 230.0 m | 38.112 |

Low sensing weights did not change the selected trajectory, while belief forecasting increased decision time substantially. SA weight 20 reduced the fixed-window error but destroyed task completion in both scenarios, showing a real navigation–sensing conflict rather than a useful gain. MPC weight 2000 changed behavior: nominal error improved modestly, but crossing became five seconds slower and tracking error worsened despite communication rising from 0.90 to 1.00.

One seed cannot establish a stable tradeoff. The pilot supports a logarithmic development sweep around each planner's behavioral transition and requires success/safety gates before interpreting lower tracking error as improvement.

The exact grid is in `grid.json`; the immutable pre-run source archive and its SHA-256 are recorded in `manifest.json`. Reproduction must use a new output directory because the runner refuses overwrite.
