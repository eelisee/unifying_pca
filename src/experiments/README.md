Experiment: compare OLS, PCR, Hybrid

This folder contains a small script `compare_methods.py` to run experiments comparing
ordinary least squares (OLS), principal components regression (PCR), and a
hybrid alternating estimator that minimizes prediction + reconstruction loss.

Basic usage:

  python -m src.experiments.compare_methods --mode synthetic

To run on the NBA dataset (CSV must be present):

  python -m src.experiments.compare_methods --mode nba --csv data/nba/nba_draft_combine_all_years.csv

Required Python packages (suggested):

- numpy
- scikit-learn
- matplotlib

You can install them in a virtualenv or conda environment. The script is lightweight
and intended for exploratory runs and extensions.
