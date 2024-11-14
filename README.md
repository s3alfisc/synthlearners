# `synthlearners`: Scalable Synthetic Control Methods

synthetic control methods powered by the [`pyensmallen`](https://github.com/apoorvalal/pyensmallen) library for fast optimisation.

Supports
- (Simplex/Lasso/Ridge/Linear/Matching) unit weights, (time weights forthcoming)
  - with multiple treated units, either performed to match aggregate outcomes (default) or individual outcomes (Abadie and L'Hour 2021)
- jacknife confidence intervals (permutation tests forthcoming)
- visualisations
  - raw outcome time series with treated average and synthetic control
  - event study plot (treatment effect over time)

```
pip install git+https://github.com/apoorvalal/synthlearners/
```

Contributions welcome!
