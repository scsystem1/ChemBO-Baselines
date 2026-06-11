Repository Structure

This repository contains the code and result files for the paper on BOOST.

Top-level contents

- Code/ - Source code used to run the experiments
- Results/ - Saved result files used in the paper
- plots/ - Scripts that regenerate the paper figures from the saved result tables
- run.py - Reviewer-friendly top-level entry point for the main experiments
- test.ipynb - Notebook front-end that builds and runs a `run.py` command from a small set of editable options
- RUN_REVIEWER.md - Practical run instructions for reviewers

Recommended starting points

- To understand the code layout, see `Code/README.txt`
- To understand the saved result layout, see `Results/README.txt`
- To regenerate the paper figures, see `plots/README.md`
- To run the main experiments, see `RUN_REVIEWER.md`

Python version

- The main experiments were run with Python `3.11.11`
- The separate No-PASt-BO environment also uses Python `3.11.11`
- The separate HEBO environment also uses Python `3.11.11`

Main experiment entry points

- `run.py` covers the main methods implemented in this repository:
  BOOST, fixed kernel-acquisition BO, static selection, random search, Best Utility,
  BoTorch default, BOOST-FixedHP, and Random-KA
- `test.ipynb` is a simpler interactive front-end for `run.py`
  Open it from the repository root, where `run.py` is located
- `plots/` contains the scripts that regenerate the main paper figures from the saved `.xlsx` result tables
- `No-PASt-BO` remains separate because it depends on a different package stack (`GPyOpt`)
- `HEBO` remains separate because it depends on a different package stack (`hebo`)

Folder roles at a glance

- `Code/tests/` - BOOST and fixed kernel-acquisition BO; the same test driver also supports BOOST-FixedHP and Random_KernelAF variants
- `Code/Static/` - Static one-shot selection baseline
- `Code/Random_Search/` - Random search baseline
- `Code/Best_Utility/` - Best Utility baseline
- `Code/BoTorch_Default/` - BoTorch default qLogNEI baseline
- `Code/HEBO/` - HEBO baseline
- `Code/Adaptive_Acquisition/` - No-PASt-BO baseline
- `plots/` - Figure-generation scripts for the paper
