# Dependencies and Licenses

This document lists all third‑party packages used in the project, their licenses, and compliance information.

## Project License

This project is distributed under the **Sanofi Non‑Commercial License** (see the `LICENSE` file provided by Sanofi OSPO).  
All third‑party components listed below remain under their respective upstream licenses.

---

## Detailed Dependency List

### Machine Learning & Scientific Computing

| Package | Version | License | Purpose |
|---------|---------|---------|---------|
| **torch** (PyTorch) | >= 1.9.0 | BSD-3-Clause | Deep learning framework for Gaussian Processes |
| **gpytorch** | >= 1.6.0 | MIT | Gaussian Process library built on PyTorch |
| **botorch** | >= 0.6.0 | MIT | Bayesian optimization library for preference modeling |
| **numpy** | >= 1.20.0 | BSD-3-Clause | Numerical computing and array operations |
| **scipy** | >= 1.7.0 | BSD-3-Clause | Scientific computing (distributions, optimization) |
| **pandas** | >= 1.3.0 | BSD-3-Clause | Data manipulation and analysis |
| **scikit-learn** | >= 1.0.0 | BSD-3-Clause | Machine learning utilities |

### AWS & LLM Integration

| Package | Version | License | Purpose |
|---------|---------|---------|---------|
| **boto3** | >= 1.20.0 | Apache-2.0 | AWS SDK for Python (Bedrock access) |
| **botocore** | >= 1.23.0 | Apache-2.0 | Low-level AWS service interface |
| **litellm** | >= 0.1.0 | MIT | Unified LLM API interface |

### Utilities

| Package | Version | License | Purpose |
|---------|---------|---------|---------|
| **tqdm** | >= 4.62.0 | MIT / MPL-2.0 | Progress bars for iterative processes |

### Python Standard Library (No External License)

Built-in modules used: `os`, `sys`, `argparse`, `pickle`, `json`, `time`, `warnings`, `pathlib`, `copy`, `concurrent.futures`

---

## Installation & Verification

### Install All Dependencies
```bash
pip install -r requirements.txt
``
