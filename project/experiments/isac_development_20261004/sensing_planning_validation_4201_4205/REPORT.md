# STEP 2 validation report

The validation grid was frozen before running reserved seeds 4201–4205. It contains 70 episodes across nominal and crossing scenarios. Every regression remains in the data, and no final evaluation seed was used.

All planners passed the success/safety gate: each SA and MPC variant completed every episode without safety intervention. Goal remained a context reference and failed crossing with 90 interventions per episode.

## SA result and search-budget sensitivity

At the default 16-iteration budget, sensing weight 2 did **not** reproduce the development RMSE gain. Relative to matched navigation SA, crossing first-10 RMSE changed by +0.0007 m, paired 95% interval [-0.0022, +0.0035], while time increased from 26 to 30 s. Nominal changed by +0.0007 m with an interval reaching zero and time increased by 0.4 s.

At 64 iterations, sensing SA had lower mean RMSE than matched navigation SA: -0.0234 m in crossing and -0.0145 m in nominal. Both five-seed intervals included zero. It required roughly 105–106 ms/step, versus 15–16 ms for navigation SA and 30–32 ms for default-budget sensing SA. Increasing search effort therefore did not establish a robust, budget-insensitive active-sensing benefit.

SA weight 2 at the default budget remains in the formal rule comparison because it was selected before validation and is useful as a weak/negative sensing result. It must not be described as a universally beneficial planner.

## MPC validation

MPC sensing weight 2000 preserved 100% success and zero safety intervention. Against matched navigation MPC, nominal first-10 RMSE improved by 0.0227 m, paired interval [0.0158, 0.0307], and late RMSE improved by 0.0387 m. Crossing RMSE improved by 0.0147 m, interval [0.0073, 0.0223], and late RMSE improved by 0.0252 m. Communication rose from 0.90 to 1.00 in crossing, while completion time increased from 25 to 30 s. Computation rose from about 0.35 to 38–49 ms/step.

This is a validated but small navigation–tracking tradeoff. It does not prove that sensing-aware planning is generally superior across tasks, budgets, target motions, or sensor models.

## RL reward scale

Saved development Goal traces show that the existing coefficient 0.2 produces sensing penalties around 0.03–0.06 per step, smaller than ordinary progress and crossing communication/safety terms. Under the proposed `log1p(rho / rho_ref)` form with `rho_ref=1`, coefficient 0.2 remains about 0.028–0.037 per step over the first ten steps; coefficient 2 raises this to about 0.283–0.370.

Accordingly, RL uses 0.2 as the low/legacy-scale coefficient and 2 as a 10× moderate ablation. Planner sensing weights cannot be copied into RL because planner belief costs are horizon-averaged additions to a different native navigation objective. Exact saved-trace magnitudes and provenance are in `selection.json`.

The validation manifest records the actual horizon, iterations, model-evaluation count, controller settings, source snapshot, and reproduction command.
