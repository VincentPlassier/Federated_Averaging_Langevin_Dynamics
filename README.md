# Federated Averaging Langevin Dynamics: Towards a unified theory and new algorithms

This repository contains the code to reproduce the experiments in the paper *Federated Averaging Langevin Dynamics: Towards a unified theory and new algorithms* by Vincent Plassier, Alain Durmus and Eric Moulines.

## Requirements

We use provide a `requirements.txt` file that can be used to create a conda
environment to run the code in this repo:
```bash
$ conda create --name <env> --file requirements.txt
```

Example set-up using `pip`:
```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Abstract

This paper focuses on Bayesian inference in a federated learning context (FL). While several distributed MCMC algorithms have been proposed, few consider the specific limitations of FL such as communication bottlenecks and statistical heterogeneity. Recently, Federated Averaging Langevin Dynamics (FALD) was introduced, which extends the Federated Averaging algorithm to Bayesian inference. We obtain a novel tight non-asymptotic upper bound on the Wasserstein distance to the global posterior for FALD. This bound highlights the effects of statistical heterogeneity, which causes a drift in the local updates that negatively affects convergence. We propose a new algorithm VR-FALD* that uses control variates to correct the client drift. We establish non-asymptotic bounds showing that VR-FALD* is not affected by statistical heterogeneity. Finally, we illustrate our results on several FL benchmarks for Bayesian inference.

## File Structure

```
├── README.md
├── requirements.txt
└── src
    ├── algo_dl
    │   ├── fald_base.py
    │   ├── fald.py
    │   ├── fsgld.py
    │   ├── sgld.py
    │   └── vr_fald_star.py
    ├── algo_np
    │   ├── fald_base_np.py
    │   ├── fald_np.py
    │   └── vr_fald_star_np.py
    ├── compute_mse.py
    ├── log_regression_predictive.py
    ├── log_regression.py
    ├── models
    │   ├── lenet5.py
    │   ├── logistic.py
    │   └── resnet.py
    ├── run_deepens_scores.py
    ├── run_fald.py
    ├── run_fald_scores.py
    ├── run_fsgld.py
    ├── run_fsgld_scores.py
    ├── run_sgd.py
    ├── run_sgd_scores.py
    ├── run_sgld.py
    ├── toy-mse.py
    └── utils
        ├── fed_dataset.py
        ├── learning_rates.py
        ├── load_dataset.py
        ├── metrics.py
        ├── sgld_tools.py
        ├── tools_dl.py
        ├── toy_tools_data.py
        ├── toy_tools_func.py
        └── uncertainties_tools.py
```
