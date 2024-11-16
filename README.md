# `synthlearners`: Scalable Synthetic Control Methods in Python

synthetic control methods powered by the [`pyensmallen`](https://github.com/apoorvalal/pyensmallen) library for fast optimisation.
Check out the `notebooks` directory for synthetic and real data examples. 


## installation

```
pip install git+https://github.com/apoorvalal/synthlearners/
```

or git clone and run `uv pip install -e .` and make changes. 

## features 

features are indicated by 
- [ ] pending; good first PR; contributions welcome 
- [x] done

### weights
  - [x] unit weights [`/solvers.py`] 
    - [x] simplex (Abadie, Diamond, Hainmueller 2010, 2015)
    - [x] lasso (Hollingsworth and Wing 2021)
    - [x] ridge (Imbens and Doudchenko 2016, Arkhangelsky et al 2021)
    - [x] matching (Imai and Kim 2019)
    - [x] support intercept term (Ferman and Pinto 2021)
    - [ ] entropy weights (Hainmueller 2012)    
  - [x] with multiple treated units, match aggregate outcomes (default) or individual outcomes (Abadie and L'Hour 2021)
  - [ ] time weights
    - [ ] L2 weights (Arkhangelsky et al 2021)
    - [ ] time-distance penalised weights (Imbens et al 2024) 
  - [ ] sklearn outcome models (Ben-Michael et al 2021)
    - [ ] matrix completion (Athey et al 2021)   
  - [ ] two-way kernel ridge weights (Ben-Michael et al 2023)

### inference
- [x] jacknife confidence intervals (multiple treated units) [Arkhangelsky et al 2021)
- [x] permutation test (single treated unit)
- [ ] conformal inference (Chernozhukov et al 2021)

### visualisations
  - [x] raw outcome time series with treated average and synthetic control
  - [x] event study plot (treatment effect over time)
  - [x] weight distributions 



Contributions welcome!
