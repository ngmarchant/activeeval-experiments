# Experiments for "A general framework for label efficient online evaluation with asymptotic guarantees"

This folder contains files required to reproduce the experiments for the 
following paper:

> N. G. Marchant and  B. I. P. Rubinstein. (2020) "A general framework for 
label efficient online evaluation with asymptotic guarantees".

## Datasets

All seven data sets are included in the `datasets` directory in HDF5 format. 
Each data set may contain the arrays:
* `features`: feature vectors
* `labels`: ground truth labels
* `scores`: real-valued classifier scores, e.g. distance from the decision 
  boundary for an SVM
* `probs`: classifier probabilities, i.e. estimates of p(y|x)
* `preds`: predicted labels according to the classifier

## Dependencies

The main dependency is the `activeeval` Python package, which implements 
our method as well as static importance sampling and passive sampling. 
The source code for `activeeval` is included under the `activeeval` directory. 
It can be installed by running:

```bash
$ pip install activeeval
```

Other dependencies include:
* `numpy`
* `pandas`
* `scipy`
* `matplotlib`
* `seaborn`
* `tables` (for reading HDF files)
* `oasis` (implementation of the OASIS evaluation method)
* `pathos` (for parallelizing experiments)

## Scripts

The experiments can be reproduced by running the `run-*.py` scripts in the 
root directory, e.g. 

```bash
$ python run_f1-score.py
```

Each script runs the experiments for a different target measure: `accuracy`, 
`f1-score` or `pr-curve` (precision-recall curve). Multiple combinations of 
datasets and methods are tested for each measure, with the evaluation for 
each combination being repeated 1000 times (to assess convergence 
statistically). The repeated evaluations are parallelized over 20 workers 
using the `pathos` library. The number of workers can be changed by modifying 
the `n_processes` variable in each script.


### Output
Results are saved in the `results` directory. The results for each target 
measure are stored in a different subdirectory, e.g. results for `f1-score` 
are stored at `results/f1-score`. Various diagnostic plots are saved under 
this subdirectory in PDf format, as well as the complete sampling history 
in compressed npz files.
