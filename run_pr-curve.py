import os
import numpy as np
import pandas as pd

from experiments import (AISHExperiment, AISHShallowExperiment, 
                         PassiveExperiment, AISIExperiment, 
                         StratifiedExperiment, OASISExperiment, ISExperiment, 
                         load_pool, result_to_dataframe, compute_true_measure)

# Avoid issues with default TkAgg backend and multithreading
import matplotlib
matplotlib.use('Agg')

import plotting
from plotting import plot_convergence, plot_results

from activeeval.pools import HierarchicalStratifiedPool
from activeeval.measures import PrecisionRecallCurve

init_wd = os.getcwd()

# Run experiments on these datasets
h5_paths = ['datasets/abt-buy-svm.h5',
            'datasets/dblp-acm-svm-small.h5']

# Specify custom y-axis limits for datasets above (excluding .h5 extension)
mse_ylims = {}

# Non-default method names to show in plots. It's best if these are short.
map_expt_name = {'AISHExperiment': 'Ours-8',
                 'AISHShallowExperiment': 'Ours-1',
                 'OASISExperiment': 'OASIS',
                 'PassiveExperiment': 'Passive',
                 'ISExperiment': 'IS',
                 'StratifiedExperiment': 'Stratified'}

# Data set names to show in plots.
map_data_name = {'abt-buy-svm': 'abt-buy',
                 'amazon-googleproducts-svm': 'amzn-goog',
                 'dblp-acm-svm-small': 'dblp-acm',
                 'restaurant-svm': 'restaurant',
                 'safedriver-xgb': 'safedriver',
                 'creditcard-lr': 'creditcard',
                 'tweets100k-svm': 'tweets100k'}

n_queries = [50]*100
n_repeats = 100
n_processes = 4
tree_depth = 8
n_children = 2
max_error = 1
compute_kl_div = True
deterministic = True
em_tol = 1e-6
em_max_iter = 1000
expt_types = [AISHExperiment, AISHShallowExperiment, ISExperiment, 
              StratifiedExperiment, PassiveExperiment]
working_dir = "results/pr-curve"

os.makedirs(working_dir, exist_ok=True)
os.chdir(working_dir)

# List for storing the dataframe of results for each data set
all_dfs = []

for h5_path in h5_paths:
    # Since we changed directory, use initial working directory as reference
    h5_path = os.path.join(init_wd, h5_path)
    print("Working on experiments for dataset at '{}'".format(h5_path))

    _, labels, scores, probs, preds, dataname = load_pool(h5_path)
    
    prior = np.c_[1 - probs, probs]
    true_label_dist = np.c_[1-labels, labels]
    labels = np.asarray([0,1])
    
    # Set the target measure
    thresholds = np.linspace(scores.min(), scores.max(), num=1024)
    measure = PrecisionRecallCurve(scores, thresholds)

    # Evaluate the (unknown) true value of the target measure
    true_measure = compute_true_measure(measure, true_label_dist, labels)

    # Partition required for OASISExperiment, AISHShallowExperiment and 
    # AISHExperiment
    pool = HierarchicalStratifiedPool(scores, tree_depth, n_children, 
                                      bins='linear')

    # Run experiments, storing only the results needed for the plots in memory 
    # (in a list of dataframes)
    expts = []
    expts_dfs = []
    for expt_type in expt_types:
        # Use non-default name for experiment if specified
        name = map_expt_name.get(expt_type.__name__, None)
        
        expt = expt_type(pool, true_label_dist, n_queries, n_repeats, measure, 
                         labels=labels, dataname=dataname, scores=scores, 
                         compute_kl_div=compute_kl_div, prior=prior,
                         prior_strength=pool.n_blocks, 
                         deterministic=deterministic, em_tol=em_tol, 
                         em_max_iter=em_max_iter, name=name)

        print("--> Getting results for experiment '{}'".format(expt.name))
        # Compute or load saved results from disk
        if not expt.complete(queried_only=True):
            expt.run_all(n_processes, queried_only=True, print_interval=20)

        # Extract results for plotting
        result_df = result_to_dataframe(expt, true_measure, 
                                        n_processes=n_processes)
        expts_dfs.append(result_df)

    # Save convergence plots for the dataset
    expts_df = pd.concat(expts_dfs)
    expts_df['dataname'] = map_data_name.get(dataname, dataname)
    all_dfs.append(expts_df)
    fig = plot_convergence(expts_df, mse_ylim=mse_ylims.get(dataname))
    fig.savefig(dataname + '_pr-curve_stacked.pdf', bbox_inches='tight', dpi=72)

all_df = pd.concat(all_dfs)

fig = plot_results(all_df, label_budget=2000)
fig.savefig('mse_2000-queries_pr-curve_barplot.pdf', bbox_inches='tight', dpi=72)

fig = plot_results(all_df, label_budget=5000)
fig.savefig('mse_5000-queries_pr-curve_barplot.pdf', bbox_inches='tight', dpi=72)
