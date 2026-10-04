# STEP 2 development planning sweep

This frozen development sweep contains 260 episodes: 10 common seeds, nominal and crossing scenarios, one Goal reference, six SA settings, and six MPC settings. Failed settings remain in all summaries. Selection uses actual first-10 tracking RMSE as the primary sensing metric and posterior covariance trace as secondary. A method must first preserve navigation success and avoid increased safety intervention in both scenarios.

## Navigation and safety gate

SA sensing weights 0.2 and 2 retained 100% success and zero safety interventions in both scenarios. Weight 20 succeeded in only 20% of each scenario; weight 200 failed every episode. The joint 20/2 setting achieved 40% nominal and 10% crossing success. Lower error from these failed settings is not considered a valid sensing improvement.

All MPC settings through weight 2000, including joint 2000/200, retained 100% success and zero safety interventions. Weight 20000 failed every episode. Thus only weights 20, 200, 2000, and joint 2000/200 pass the MPC gate.

## Same-family paired results

SA weight 0.2 was effectively inert. SA weight 2 was also identical to navigation SA in nominal, while crossing first-10 RMSE improved by 0.0214 m with a seed-paired 95% bootstrap interval of [0.0041, 0.0422] m. Late RMSE improved by 0.0414 m, interval [0.0130, 0.0732] m. Communication rose from 0.885 to 0.995, while mean crossing time increased from 26.1 to 29.3 s and path length from 236.7 to 261.3 m. Planner time rose from 4.42 to 31.70 ms/step.

MPC weights 20 and 200 did not change nominal behavior; weight 20 was also inert in crossing. Weight 200 improved crossing time by 1 s and communication from 0.90 to 1.00, but its RMSE difference interval crossed zero. Weight 2000 improved nominal first-10 RMSE by 0.0182 m, interval [0.0108, 0.0264] m, and late RMSE by 0.0413 m, interval [0.0305, 0.0532] m. In crossing its mean RMSE improved by 0.0074 m but the interval [-0.0029, 0.0183] m included zero; time increased from 25 to 30 s. Planner time was 37.74–48.80 ms/step versus 0.34–0.36 for navigation MPC. The joint setting did not improve the primary metric more consistently and increased crossing time to 35.8 s.

Posterior covariance generally decreased when sensing behavior changed, but it remains secondary because calibration diagnostics found strong initialization transients. NEES did not consistently improve with lower covariance. The observed gains are small navigation–sensing tradeoffs rather than evidence of a large active-sensing effect.

## Prespecified validation candidates

- **SA sensing weight 2, communication weight 0**: the largest SA setting that passes both navigation gates, with a nonzero paired crossing RMSE improvement.
- **MPC sensing weight 2000, communication weight 0**: passes both gates and gives the clearest nominal actual-RMSE improvement, while validation must test whether the crossing time cost and uncertain crossing RMSE persist.

Validation should compare each candidate to its same-family navigation planner on seeds 4201–4205. For SA it should additionally compare horizon 4 / iterations 16 against horizon 4 / iterations 64 for both navigation and weight 2, treating search budget as a planned sensitivity rather than tuning. No final-test seed has been used.

The original run used default SA horizon 4, 16 iterations, 21 sequence evaluations per action, and default MPC horizon 3 with at most 49 candidate trajectories per action. The source archive was captured before the first episode. Later runner support for explicit search dictionaries is recorded separately in the manifest and does not change this run.
