Code Structure

tests → Python scripts to evaluate the performance of BOOST and the fixed kernel/acquisition combinations
- Test_Benchmark_Functions.py → Runs tests on synthetic benchmark functions
   • use_boost = True → Runs BOOST
   • use_boost = False → Uses fixed kernel/acquisition combinations
   • fixed_hp_mode = True → Runs the BOOST-FixedHP diagnostic variant
- Test_HPOB.py → Runs tests on HPO-B tasks stored in benchmarks/hpob_data
   • use_boost = True → Runs BOOST
   • use_boost = False → Uses fixed kernel/acquisition combinations
   • fixed_hp_mode = True → Runs the BOOST-FixedHP diagnostic variant
- _class_for_test_BOOST.py → Defines the class to run the BO cycle (with or without BOOST).
   Used by Test_Benchmark_Functions.py and Test_HPOB.py.
   Also extracts full-data GP hyperparameters when fixed_hp_mode is enabled.

benchmarks → Definitions of synthetic benchmark functions and processed HPO-B datasets
- Benchmark_ftn.py → Defines synthetic benchmark functions
- hpob_data/ → Contains processed HPO-B .csv files used in the experiments

core → Core classes and functions for Bayesian Optimization
- BayesianOptimization.py → Implements BO steps and candidate selection
- BOOST.py → Recommends a kernel/acquisition function pair using data-in-hand
- The fixed-hyperparameter path reuses full-data GP hyperparameters inside BOOST's internal reference-set simulations
- kernels_and_acquisitions.py → Defines GP models and enumerates kernel/acquisition options

utils → Utility functions
- Save_results.py → Saves trial summaries, recommendation logs, acquisition logs, kernel logs, and final evaluated points

Baseline Codes (organized by method)

Adaptive_Acquisition (No-PASt-BO)
- nopastbo_synthetic_benchmark.py → No-PASt-BO for synthetic benchmark functions
- nopastbo_hpob.py → No-PASt-BO for HPO-B tasks
- nopast.py → Core Hedge-based adaptive acquisition logic
- nopastbo_benchmarks.py → Synthetic benchmark definitions used by No-PASt-BO
- BO.py → Supporting BO utilities for the adaptive acquisition baseline
- Original No-PASt-BO code: https://github.com/thiago-vasconcelos/no-past-bo

Best_Utility
- Test_Benchmark_Functions_best_utility.py → Best Utility for synthetic benchmark functions
- Test_HPOB_best_utility.py → Best Utility for HPO-B tasks
- _class_for_test_best_utility.py → Defines the BO loop for the Best Utility baseline
- BayesianOptimization_for_Best_Utility.py → BO utilities specialized for the Best Utility baseline

Static
- Test_Benchmark_Functions_static.py → Static method for synthetic benchmark functions
- Test_HPOB_static.py → Static method for HPO-B tasks
- _class_for_test_static.py → Defines the BO loop for the static baseline

Random_Search
- Test_Benchmark_Functions_randomsearch.py → Random Search for synthetic benchmark functions
- Test_HPOB_randomsearch.py → Random Search for HPO-B tasks
- _class_for_test_randomsearch.py → Defines the BO loop for the random-search baseline

BoTorch_Default
- Test_Benchmark_Functions_botorch.py → BoTorch default qLogNEI for synthetic benchmark functions
- Test_HPOB_botorch.py → BoTorch default qLogNEI for HPO-B tasks
- _class_for_test_botorch.py → Defines the BO loop using BoTorch's default GP model and qLogNEI acquisition

HEBO
- Test_Benchmark_Functions_hebo.py → HEBO for synthetic benchmark functions
- Test_HPOB_hebo.py → HEBO for HPO-B tasks
- _class_for_test_hebo.py → Defines the HEBO loop with nearest-neighbor projection to the discrete candidate set

BOOST_FixedHP
- Uses the same test drivers and BO loop as BOOST with `fixed_hp_mode=True`
- Fits GP hyperparameters on the current full data-in-hand, then reuses those values during BOOST's internal reference-set simulations
- Reported as a diagnostic variant for evaluating whether fixed full-data GP hyperparameters explain BOOST's behavior

Random_KernelAF
- Uses the same test drivers and BO loop as BOOST with `random_select_mode=True`
- At each outer BO iteration, draws one (kernel, acquisition) pair uniformly at random from the same 16-pair portfolio used by BOOST, skipping the internal evaluation entirely
- Each trial uses a distinct random seed combined with the outer iteration index so the per-trial sequence of pairs differs across trials
- Reported as a baseline for isolating how much of BOOST's gain comes from the data-driven internal evaluation versus simply rotating through pairs

Ablation Settings for BOOST

The ablation studies in the paper are controlled mainly through `core/BOOST.py`.
The most relevant edit points are:

- Lines 20-21 → Candidate kernel and acquisition sets
  Edit `kernel_candidates` and `acquisition_candidates` to restrict the candidate pool
- Line 61 → Partitioning size ratio for `r_n`
  `ratio_init_boost` controls how large `r_n` is relative to the full data-in-hand
  Since the code uses `n_init_boost = train_x_init.shape[0] // ratio_init_boost`, the parameter is an inverse-style divisor rather than a direct ratio label
  If the full data size is `n` and `|r_n| = n / a`, then `|s_n| = n - |r_n|`, so the implied ratio is `r_n : s_n = 1/a : (1 - 1/a)`
  Examples:
  • ratio `1:2` corresponds to `ratio_init_boost = 3`
  • ratio `2:1` corresponds to `ratio_init_boost = 3/2`
  • In the current implementation, the internal sample size is computed as:
	- `n_init_boost = min(max_init_boost, max(min_init_boost, train_x_init.shape[0] // ratio_init_boost))`
- Line 62 → Maximum number of internal BO iterations
  Edit `max_iter_boost` for the stopping-criteria ablation on the internal loop budget
- Lines 93-96 → Percentile-based stopping target
  Edit `percentile = max(1, round(len(sorted_y) * 0.05))` to change the target percentile
  For example, use percentile `0` to use the global optimum instead
- Lines 108-110 → Partitioning method
  The default uses K-means representative sampling
  Replace this with the commented `torch.randperm(...)[:n_init_boost]` line to use random sampling instead
- Lines 197-202 → Tie-breaking rule
  The default selects the first minimum-iteration result via `min(...)`
  Replace it with the commented random tie-breaking block to choose uniformly among tied candidates
  If you enable random tie-breaking, also add `import random` near the top of `BOOST.py`


Note: Throughout the code and results, the Lower Confidence Bound (LCB) acquisition function is referred to as UCB for convenience, following common usage in BO libraries.
