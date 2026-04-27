
## Repository Structure

```
UDS-BOPE/
├── acquisition.py              # Acquisition functions (Random, UCB, EI, LLM-EI)
├── model.py                    # Gaussian Process models (standard GP, preference GP)
├── preference_BO.py            # Preference-guided BO for BH datasets
├── run_amide_coupling_bo.py    # Amide coupling BO workflow
├── amide_llm_interface.py      # LLM interface for amide reactions
└── LLM_survey/                 # Survey generation and execution
    ├── run_llm_survey_BH.py    # Buchwald-Hartwig survey
    ├── run_llm_survey_DA.py    # Direct arylation survey
    └── utils.py                # LLM utilities
```

## Usage

### Running Complete BO Workflow

**Buchwald-Hartwig datasets (BH1-5):**
```bash
python preference_BO.py --n_trials 50 --n_start 1 --acc 0.7 --m 100 --beta 2.0
```

**Amide Coupling datasets:**
```bash
python run_amide_coupling_bo.py --n_trials 50 --n_start 1 --acc 0.8 --m 100 --beta 2.0
```

**Arguments:**
- `--n_trials`: Number of independent BO runs (default: 50)
- `--n_start`: Number of initial random observations (default: 1)
- `--acc`: LLM survey accuracy for synthetic preference data (default: 0.7-0.8)
- `--m`: Number of pairwise comparisons (default: 100)
- `--beta`: Exploration parameter (not used in LLM-EI)

### Running LLM Surveys

```bash
python LLM_survey/run_llm_survey_BH.py \
    --questions questions.pkl \
    --exp_database dataset.csv \
    --res_dir results/ \
    --llm_result survey_results.csv \
    --model_type 0
```

**Model types:** 0=Claude 3.5 Sonnet, 1=Claude 3 Sonnet, 2=Claude 3 Haiku

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify Licenses

To check all installed package licenses:
```bash
pip install pip-licenses
pip-licenses --format=markdown
```

For detailed license information, see [DEPENDENCIES.md](DEPENDENCIES.md).

## License

See the [LICENSE](LICENSE) file for details.

