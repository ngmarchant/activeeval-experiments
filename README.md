# Experiments for "A general framework for label efficient online evaluation with asymptotic guarantees"

This repository contains files required to reproduce the experiments for the 
following paper:

> Neil G. Marchant and Benjamin I. P. Rubinstein. 2021. 
  Needle in a Haystack: Label-Efficient Evaluation under Extreme Class 
  Imbalance. 
  In _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and 
  Data Mining (KDD ’21)_, 
  ACM, New York, NY, USA, 11 pages. 
  DOI: [10.1145/3447548.3467435](https://doi.org/10.1145/3447548.3467435).
  arXiv: [2006.06963](https://arxiv.org/abs/2006.06963).

## Datasets

All seven data sets are included in the `datasets` directory in HDF5 format. 
Each data set may contain the following variables:

* `features`: feature vectors
* `labels`: ground truth labels
* `scores`: real-valued classifier scores, e.g. distance from the decision 
  boundary for an SVM
* `probs`: classifier probabilities, i.e. estimates of p(y|x)
* `preds`: predicted labels according to the classifier

## Dependencies

The main dependency is the [ActiveEval](https://github.com/ngmarchant/activeeval) 
Python package, which implements our method as well as static importance 
sampling and passive sampling. 
Other dependencies include:

* `numpy`
* `pandas`
* `scipy`
* `matplotlib`
* `seaborn`
* `tables` (for reading HDF files)
* `oasis` (implementation of the OASIS evaluation method)
* `pathos` (for parallelizing experiments)

We recommend managing dependencies in a Python virtual environment using 
[Pipenv](https://pipenv.pypa.io). The included `Pipfile` and `Pipfile.lock` 
files specify all dependencies and their versions. Assuming Pipenv is 
installed, you can initialize the virtual environment by running the 
following command from the root of this repository:

```bash
$ pipenv sync
```

## Scripts

The experiments can be reproduced by executing the `run-*.py` scripts in the 
root of this repository. First, ensure the virtual environment has been 
initialized according to the instructions under "Dependencies". Then enter 
the virtual environment by executing the following from the root of 
the repository:

```bash
pipenv shell
```

Now you're ready to run the scripts. To run the experiments for F1-score, 
execute the following:

```bash
(activeeval-experiments) $ python run_f1-score.py
```

Each script runs the experiments for a different target measure: `accuracy`, 
`f1-score` or `pr-curve` (precision-recall curve). Multiple datasets and 
methods are evaluated for each measure, with the evaluation for 
each measure/dataset/method being repeated 100 times. The repeated 
evaluations are parallelized across 4 workers using the `pathos` library. 
The number of workers can be changed by modifying the `n_processes` variable 
in each script.


### Output
Results are saved in the `results` directory. The results for each target 
measure are stored in a different subdirectory, e.g. results for `f1-score` 
are stored at `results/f1-score`. Various diagnostic plots are saved under 
this subdirectory in PDf format, as well as the complete sampling history 
in compressed npz files.
