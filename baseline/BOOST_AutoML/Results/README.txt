Results Structure

This folder stores the experiment outputs used in the paper.

Top-level organization

- `01_Ackley`, `02_Levy`, `03_Rosenbrock` → Synthetic benchmark results
- `2277_15D`, `5636_6D`, `5891_8D`, `5906_16D`, `5964_9D`, `6322_8D`, `6762_6D`, `6794_10D` → HPO-B task results
- `Ablation Study/` → Results for the ablation analyses reported in the paper

Per-task subfolders

Inside each benchmark/task folder, the main method folders are:

- `BOOST/` → Results of the proposed BOOST method
- `Fixed_ker_acq/` → Results of fixed kernel-acquisition combinations
- `Static/` → Results of the static one-shot selection baseline
- `Random/` → Results of the random search baseline
- `Adaptive_ker/` → Results of the Best Utility baseline
- `Adaptive_acq/` → Results of the No-PASt-BO baseline
- `BoTorch_Default/` → Results of the BoTorch default qLogNEI baseline
- `HEBO/` → Results of the HEBO baseline
- `BOOST_FixedHP/` → Results of the BOOST variant with fixed full-data GP hyperparameters
- `Random_KernelAF/` → Results of the Random-KA baseline (uniform random pick from the 16-pair portfolio at every iteration)

What the file names mean

Summary files

- `Objective_recommended_results.xlsx`
  Aggregate summary over seeds for methods that recommend a strategy over time, such as BOOST
- `Objective_Kernel_Acquisition_results.xlsx`
  Aggregate summary over seeds for a fixed or effectively determined kernel-acquisition configuration
  For baselines without an explicit kernel-acquisition pair, method-specific labels are used, such as `TBD_TBD` for the BoTorch default baseline and `HEBO_MACE` for HEBO

Per-seed final evaluated points

- `Objective_seedN_Kernel_Acquisition_final.xlsx`
  Full set of evaluated points for seed `N`, together with metadata such as the final best value
  For adaptive methods such as `BOOST/` and `Adaptive_ker/`, the kernel-acquisition names in this final filename reflect the setting selected at the last iteration
  The chosen strategy may change across iterations, so consult the corresponding log files to see the full selection history

Per-seed logs

- `Objective_recommendation_log_N.xlsx`
  Sequence of kernel-acquisition recommendations for seed `N`
- `Objective_acquisition_log_seed_N.xlsx`
  Acquisition-selection log for seed `N`
- `Objective_kernel_log_seed_N.xlsx`
  Kernel-selection log for seed `N`

Interpretation of method-specific logs

- `BOOST/` typically contains `recommendation_log` files because BOOST recommends both kernel and acquisition functions
- `Adaptive_acq/` typically contains `acquisition_log` files because No-PASt-BO adapts the acquisition function
- `Adaptive_ker/` typically contains `kernel_log` files because Best Utility adapts the kernel while using a fixed acquisition setting
- `Static/` also contains `recommendation_log` files because it performs a one-time strategy selection and then keeps that choice fixed
- `BoTorch_Default/` uses `TBD_TBD` in file names because the kernel, acquisition, and priors are determined by the BoTorch default model and acquisition setup
- `HEBO/` uses `HEBO_MACE` in file names because HEBO uses its own model and MACE acquisition strategy
- `BOOST_FixedHP/` follows the BOOST-style result format and contains `recommendation_log` files
- `Random_KernelAF/` follows the BOOST-style result format and contains `recommendation_log` files (each iteration's randomly drawn (kernel, acquisition) pair)

Fixed-kernel/acquisition results

The `Fixed_ker_acq/` folder contains the full Cartesian set of tested fixed combinations:

- Kernels: `Matern32`, `Matern52`, `RBF`, `RQ`
- Acquisitions: `EI`, `PI`, `UCB`, `PM`

For example:

- `Ackley_Matern32_EI_results.xlsx` → Summary across seeds for the fixed Matérn 3/2 + EI setting
- `Ackley_seed0_Matern32_EI_final.xlsx` → Final evaluated points for seed 0 under that setting

Ablation Study

`Ablation Study/` stores separate experimental variants used in the ablation section.
Each subfolder corresponds to one ablation setting, for example:

- `1_Random_Sampling`
- `2_Ratio_1_1`, `2_Ratio_1_4`, `2_Ratio_2_1`, `2_Ratio_4_1`
- `3_Percentile_0`, `3_Percentile_10`
- `4_Unlimited_max_iter`
- `5_Random_tie_breaking`
- `6_EI`
- `6_Matern32`

These folders follow the same basic naming rules as the main `BOOST/` results:

- summary files ending in `_recommended_results.xlsx`
- per-seed final files ending in `_final.xlsx`
- recommendation logs ending in `_recommendation_log_N.xlsx`

Notes

- In file names, the lower confidence bound acquisition function is labeled `UCB`
- Synthetic benchmarks use names such as `Ackley`, `Levy`, and `Rosenbrock`
- HPO-B tasks use names such as `5891_8D` and `6794_10D`
