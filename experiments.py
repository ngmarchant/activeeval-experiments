import warnings
import os
import time

import numpy as np

# Typing
from typing import Tuple, Union, Iterable, Optional, NamedTuple, Iterator, KeysView, List
from numpy import ndarray
from activeeval.pools import BasePool, BasePartitionedPool, BaseHierarchicalPool
from activeeval.measures import BaseMeasure
from activeeval.measures import FMeasure

from abc import ABC, abstractmethod

# Pre- and post-processing
import tables                            # reading HDF5 files
from scipy.special import expit, logit   # for converting scores/probabilities
from scipy.stats import entropy          # for computing KL divergence
import pandas as pd

# Evaluation methods
import oasis
from activeeval.proposals import (Passive, StaticVarMin, PartitionedStochasticOE, PartitionedDeterministicOE,
                                  PartitionedAdaptiveVarMin, PartitionedIndepOE,
                                  HierarchicalStochasticOE, HierarchicalDeterministicOE, AdaptiveBaseProposal,
                                  AdaptiveVarMin, compute_optimal_proposal)
from activeeval.evaluator import Evaluator
from activeeval.estimators import StratifiedEstimator

from pathos.pools import ProcessPool     # parallelization
from functools import partial
from itertools import repeat


def load_pool(path: str, load_features: bool = False) -> Tuple:
    """Load pool from HDF file

    Parameters
    ----------
    path : str
        Path to HDF file.

    load_features : boolean, optional (default: False)
        Whether to load the feature vectors associated with each instance.

    Returns
    -------
    features : numpy.ndarray, shape=(n_items,n_features) or None
        Feature vectors for each item, if available.

    labels : numpy.ndarray, shape=(n_items,) or None
        Ground truth labels for each item, if available.

    scores : numpy.ndarray, shape=(n_items,)
        Real-valued classifier scores for each item. If not included in 
        the HDF file, these are derived from the probabilities by applying 
        the logit function.

    probs : numpy.ndarray, shape=(n_items,)
        Estimate of the probability p(y=1|x) for each item x. If not 
        included in the HDF file, these are derived from the scores by 
        applying the expit function.

    preds : numpy.ndarray, shape=(n_items,)
        Predicted label for each item. If not included in the HDF file, 
        these are derived from the scores by thresholding at zero.

    dataset_name : str
        Name of the data set (extracted from the filename).
    """
    features = None
    labels = None
    scores = None
    probs = None
    preds = None

    h5_file = tables.open_file(path, mode='r')
    if load_features and hasattr(h5_file.root, "features"):
        features = h5_file.root.features[:, :]
    if hasattr(h5_file.root, "labels"):
        labels = h5_file.root.labels[:]
    if hasattr(h5_file.root, "scores"):
        scores = h5_file.root.scores[:]
    if hasattr(h5_file.root, "probs"):
        probs = h5_file.root.probs[:]
    if hasattr(h5_file.root, "preds"):
        preds = h5_file.root.preds[:]
    h5_file.close()

    if path.lower().find('svm') != -1:
        # Remove calibrated probabilities from SVM which were expensive to compute
        probs = None

    if probs is None and scores is None:
        raise RuntimeError('probs and scores both missing')

    if probs is None:
        warnings.warn('converting scores into probabilities', UserWarning)
        probs = expit(scores)

    if scores is None:
        warnings.warn('converting probabilities into scores', UserWarning)
        scores = logit(probs)

    if preds is None:
        warnings.warn('making predictions from scores using default threshold of zero', UserWarning)
        preds = (scores >= 0) * 1

    dataset_name = os.path.splitext(os.path.basename(path))[0]

    return features, labels, scores, probs, preds, dataset_name


def compute_true_measure(measure: BaseMeasure, true_label_dist: ndarray, labels: Optional[ndarray] = None):
    """Compute the measure given the (unknown) oracle response
    
    Parameters
    ----------
    measure : instance of activeeval.measures.BaseMeasure
        Evaluation measure to estimate

    true_label_dist : numpy.ndarray, shape (n_instances, n_classes)
        Ground truth label distribution p(y|x) for each instance x in the pool.
    
    labels : array-like, shape (n_classes,)
        The set of class labels, i.e. the support of the oracle response
        :math:`p(y|x)`. If None, the labels are assumed to be integers in
        the set `{0, 1, ..., n_classes - 1}`, where the number of classes
        `n_classes` is inferred from `response_est`.
    """
    if labels is None:
        labels = np.arange(true_label_dist.shape[1])
    
    n_instances, n_classes = true_label_dist.shape
    idx = np.repeat(np.arange(n_instances), n_classes)
    y = np.tile(labels, n_instances)
    # Compute loss: 0th axis corresponds to (idx, y) combination. 1st axis corresponds to dimensions of the risk.
    # Later can be reshaped to (n_instances, n_classes, measure.n_dim_risk)
    loss = measure.loss(idx, y)

    # Reshape to match 0th axis of loss
    true_label_dist = true_label_dist.ravel()

    # Evaluate the risk. Works when loss is an ndarray or spmatrix.
    true_risk = loss.T.dot(true_label_dist / n_instances)

    true_measure = measure.g(true_risk)

    return true_measure


RunResult = NamedTuple('RunResult', [('estimate_history', ndarray), ('queried_oracle', Optional[ndarray]),
                                     ('kl_div_history', Optional[ndarray])])

class LabelCache:
    """Container for storing labels of instances, as received from the oracle
    """
    def __init__(self, deterministic: bool = True):
        self.deterministic = deterministic
        self._map = {}

    def __getitem__(self, instance: Union[str, int]) -> Union[List[int], int]:
        return self._map[instance]

    def __len__(self) -> int:
        return len(self._map)

    def __str__(self) -> str:
        return str(self._map)

    def __contains__(self, instance_id) -> bool:
        return self._map.__contains__(instance_id)

    def __iter__(self):
        for instance in self._map.items():
            yield instance

    def clear(self) -> None:
        """Clear the cache
        """
        self._map.clear()

    def instances(self) -> KeysView:
        """A view on the instances in the cache
        """
        return self._map.keys()

    def update(self, instance_id, label) -> None:
        """Update cache with a label
        """
        if self.deterministic:
            self._map[instance_id] = label
        else:
            if instance_id in self._map:
                self._map[instance_id].append(label)
            else:
                self._map[instance_id] = [label]


class BaseExperiment(ABC):
    """Base class for an experiment

    Parameters
    ----------
    pool : an instance of activeeval.pools.BasePool
        A pool of items.

    true_label_dist : numpy.ndarray, shape (n_instances, n_classes)
        Ground truth label distribution p(y|x) for each instance x in the pool.

    n_queries : int or array-like
        Number of label queries to make in each evaluation run

    n_repeats : int
        Number of times to repeat the evaluation

    measure : instance of activeeval.measures.BaseMeasure
        Target evaluation measure

    labels : array-like, shape (n_classes,) or None, (default=None)
        The set of class labels, i.e. the support of the oracle response
        :math:`p(y|x)`. If None, the labels are assumed to be integers in
        the set `{0, 1, ..., n_classes - 1}`, where the number of classes
        `n_classes` is inferred from `response_est`.

    dataname : str or None, optional
        Name of dataset. Defaults to 'unknown-dataset'.

    name : str or None, optional
        Name of experiment.

    compute_kl_div : bool, optional
        Whether to compute the KL-divergence in each round. Defaults to False.

    deterministic_oracle : bool, optional
        Whether the oracle is deterministic. Defaults to True.

    Attributes
    ----------
    optimal_pmf : numpy.ndarray, shape (n_instances,)
        Asymptotically optimal (variance-minimizing) proposal distribution
        for importance sampling.

    path : str
        Path to directory where experimental data is stored.
    """
    def __init__(self, pool: BasePool, true_label_dist: ndarray, n_queries: Union[int, Iterable, ndarray], 
                 n_repeats: int, measure: BaseMeasure, labels: Union[Iterable, ndarray, None] = None, 
                 dataname: Optional[str] = None, name: Optional[str] = None, compute_kl_div: bool = False, 
                 deterministic_oracle: bool = True, **kwargs) -> None:
        self.pool = pool
        self.true_label_dist = true_label_dist
        if labels is None:
            self.labels = np.arange(self.true_label_dist.shape[1])
        else:
            self.labels = np.asarray(labels)
        # Store original n_queries in private variable. Non-adaptive methods may collapse to one round.
        self._n_queries = n_queries
        self.n_repeats = int(n_repeats)
        self.measure = measure
        self.dataname = dataname if dataname is not None else 'unknown-dataset'
        self.name = name
        self.deterministic_oracle = deterministic_oracle
        self.compute_kl_div = compute_kl_div
        self.optimal_pmf = compute_optimal_proposal(self.pool, self.labels, self.true_label_dist, self.measure)
        self.evaluator = self._setup_evaluator(**kwargs)

    @property
    def n_queries(self) -> Union[int, Iterable, ndarray]:
        if isinstance(self._n_queries, (int, np.integer)):
            return repeat(1, self.n_queries)
        else:
            return self._n_queries

    @property
    @abstractmethod
    def path(self) -> str:
        pass

    def _file(self, seed: int, queried_only: bool) -> str:
        """Returns path to npz file for an evaluation run"""
        queried_str = '_queried_only' if queried_only else ''
        filename = 'seed-{}_queries-{}_rounds-{}{}.npz'.format(seed, np.sum(self.n_queries), np.size(self.n_queries), queried_str)
        return os.path.join(self.path, filename)

    @abstractmethod
    def _setup_evaluator(self, **kwargs):
        pass

    def _run(self, seed: int, queried_only: bool = False) -> RunResult:
        """Do a single evaluation run and save the result to disk

        Parameters
        ----------
        seed : int
            Random seed.
        
        queried_only : bool, optional (default=True)
            Whether to only return samples for which the oracle was queried. 
        
        Returns
        -------
        RunResult
        """
        random_state = np.random.RandomState(seed)
        self.evaluator.proposal.random_state = random_state
        self.evaluator.reset()

        label_cache = LabelCache(self.deterministic_oracle)
        # Record whether oracle was queried for each sample (oracle will not be
        # queried if it is deterministic and the instance has been labelled
        # before)
        queried_oracle = []
        kl_div_history = [] if self.compute_kl_div else None

        for n in self.n_queries:
            # Query labels for instances in this round. Since we want to count
            # the number of queries to the oracle, which is not necessarily
            # equal to the number of samples, the code here is a bit
            # long-winded.
            query_ctr = 0
            instance_ids = []
            weights = []
            labels = []
            while query_ctr < n:
                # Sample instances to label in bulk (may not use all of them,
                # if target number of queries is met)
                this_instance_ids, this_weights = self.evaluator.query(n)
                # Convert to arrays so that we can use zip below even if n == 1
                this_instance_ids, this_weights = np.atleast_1d(this_instance_ids, this_weights)

                # Query labels from oracle
                this_labels = []
                for instance_id in this_instance_ids:
                    if self.deterministic_oracle and (instance_id in label_cache):
                        label = label_cache[instance_id]
                        queried_oracle.append(False)
                    else:
                        p = self.true_label_dist[instance_id]
                        label = random_state.choice(self.labels, p=p) # get from "oracle"
                        label_cache.update(instance_id, label)
                        queried_oracle.append(True)
                        query_ctr += 1
                    this_labels.append(label)

                    if query_ctr == n:
                        # Made enough queries to oracle for this round
                        break

                # Append results to lists
                this_instance_ids = this_instance_ids[:len(this_labels)]
                this_weights = this_weights[:len(this_labels)]
                instance_ids.extend(this_instance_ids.tolist())
                weights.extend(this_weights.tolist())
                labels.extend(this_labels)

            # Update evaluator using results from this round
            self.evaluator.update(instance_ids, labels, weights)

            # Update KL divergence for this round
            if self.compute_kl_div:
                if not isinstance(self.evaluator.proposal, AdaptiveBaseProposal) and kl_div_history:
                    # KL divergence is constant, so just repeat initial value
                    kl_div_history.append(kl_div_history[-1])
                else:
                    this_kl_div = entropy(self.optimal_pmf, self.evaluator.proposal.get_pmf())
                    kl_div_history.append(this_kl_div)
        
        estimate_history = self.evaluator.estimate_history
        if queried_only:
            estimate_history = [est for q, est in zip(queried_oracle, estimate_history) if q]
            queried_oracle = None
        else:
            queried_oracle = np.array(queried_oracle)
        estimate_history = np.array(estimate_history)
        if kl_div_history is not None:
            kl_div_history = np.array(kl_div_history)
        return RunResult(estimate_history, queried_oracle, kl_div_history=kl_div_history)

    @staticmethod
    def _save(result: RunResult, file: str) -> None:
        """Save run result to a compressed npz file

        Parameters
        ----------
        result : RunResult
            Result of a single run

        file : str
            File where the data will be saved. An ``.npz`` extension is
            appended automatically if not already present.
        """
        arrays_to_save = {'estimate_history': result.estimate_history,
                          'queried_oracle': result.queried_oracle}
        if result.kl_div_history is not None:
            arrays_to_save['kl_div_history'] = result.kl_div_history
        dirname = os.path.dirname(file)
        os.makedirs(dirname, exist_ok = True)
        np.savez_compressed(file, **arrays_to_save, allow_pickle=True)

    def _run_and_save(self, seed: int, queried_only: bool = False) -> None:
        file = self._file(seed, queried_only)
        if not os.path.isfile(file):
            result = self._run(seed, queried_only)
            self._save(result, file)

    def result(self, seed: int, queried_only: bool = False) -> RunResult:
        """Get result for an evaluation run

        The result is loaded from disk if available. Otherwise it is computed 
        from scratch and saved to disk.

        Parameters
        ----------
        seed : int
            Random seed.
        
        queried_only : bool, optional (default=False)
            Whether to only return samples for which the oracle was queried.

        Returns
        -------
        an instance of RunResult
        """
        file = self._file(seed, queried_only)
        try:
            # Read from disk
            npz_obj = np.load(file, allow_pickle=True)
            result = RunResult(npz_obj['estimate_history'], npz_obj.get('queried_oracle', None),
                               npz_obj.get('kl_div_history', None))
        except FileNotFoundError:
            try:
                # Try extracting from result with all samples
                file = self._file(seed, False)
                npz_obj = np.load(file, allow_pickle=True)
                estimate_history = npz_obj['estimate_history']
                queried_oracle = npz_obj['queried_oracle']
                result = RunResult(estimate_history[queried_oracle], None,
                                   npz_obj.get('kl_div_history', None))
            except FileNotFoundError:
                file = self._file(seed, queried_only)
                result = self._run(seed, queried_only)
                self._save(result, file)
        return result

    def results_iter(self) -> Iterator[RunResult]:
        """Return an iterator over the results"""
        for seed in range(self.n_repeats):
            yield self.result(seed)

    def complete(self, queried_only: bool = False) -> bool:
        """Checks whether all experiments have been run and saved to disk
        
        Parameters
        ----------
        queried_only : bool, optional (default=False)
            Whether to only save samples for which the oracle was queried.
        """
        for seed in range(self.n_repeats):
            files = [self._file(seed, queried_only)]
            if queried_only:
                files.append(self._file(seed, not queried_only))
            if not any([os.path.isfile(file) for file in files]):
                return False
        return True

    def run_all(self, n_processes: int, queried_only: bool = False, print_interval: int = 5) -> None:
        """Repeat the evaluation multiple times

        Parameters
        ----------
        n_processes : int
            Number of parallel processes to use
        
        queried_only : bool, optional (default=False)
            Whether to only save samples for which the oracle was queried.

        print_interval : float
            How often (in seconds) to print progress
        """
        pool = ProcessPool(n_processes)
        print("Running experiment {} with {} repeats".format(self.name, self.n_repeats))
        results = pool.amap(partial(self._run_and_save, queried_only=queried_only), range(self.n_repeats))
        while not results.ready():
            print("Waiting for", results._number_left, "tasks to complete...")
            time.sleep(print_interval)
        results.get()


class OASISExperiment(BaseExperiment):
    def __init__(self, pool: BasePartitionedPool, true_label_dist: ndarray, n_queries: Union[int, Iterable, ndarray],
                 n_repeats: int, measure: FMeasure, labels: Union[Iterable, ndarray, None] = None, 
                 dataname: Optional[str] = None, name: Optional[str] = None,
                 compute_kl_div: bool = False, **kwargs) -> None:
        if not isinstance(measure, FMeasure):
            raise TypeError("`measure` must be an instance of activeeval.measures.FMeasure")
        super().__init__(pool, true_label_dist, n_queries, n_repeats, measure, labels=labels, dataname=dataname,
                         name=name, compute_kl_div=compute_kl_div, deterministic_oracle=True, **kwargs)
        self.name = 'OASIS' if name is None else name

    @property
    def n_queries(self) -> Union[int, Iterable, ndarray]:
        return np.ones(np.sum(self._n_queries).item(), dtype=int)

    @property
    def path(self) -> str:
        return (self.dataname + 
                '_oasis_{}-blocks'.format(self.pool.n_blocks) + 
                '_deterministic' if self.deterministic_oracle else '_non-deterministic')

    def _setup_evaluator(self, **kwargs):
        alpha = 1 / (1 + self.measure.beta ** 2)
        prior_strength = kwargs.get('prior_strength', None)
        probs = kwargs['prior'][:, 1]
        strata = oasis.Strata(self.pool.block_assignments)
        preds = self.measure.y_pred
        oracle = lambda idx: np.argmax(self.true_label_dist[idx])
        return oasis.OASISSampler(alpha, preds, probs, oracle, strata=strata,
                                  max_iter=2000000, proba=True, epsilon=0.001,
                                  prior_strength=prior_strength)

    def _run(self, seed: int, queried_only: bool = False) -> RunResult:
        n_queries = np.sum(self.n_queries).item()
        np.random.seed(seed)
        self.evaluator.reset()
        if self.compute_kl_div:
            kl_div_history = []
            for _ in range(n_queries):
                if self.deterministic_oracle:
                    self.evaluator.sample_distinct(1)
                else:
                    self.evaluator.sample(1)
                this_pmf = (self.evaluator.inst_pmf_ / self.evaluator.strata.sizes_)[self.pool.block_assignments]
                this_kl_div = entropy(self.optimal_pmf, this_pmf)
                kl_div_history.append(this_kl_div)
            kl_div_history = np.array(kl_div_history)
        else:
            if self.deterministic_oracle:
                self.evaluator.sample_distinct(n_queries)
            else:
                self.evaluator.sample(n_queries)
            kl_div_history = None
        estimate_history = self.evaluator.estimate_[:, np.newaxis]
        if queried_only:
            queried_oracle = None
            if self.deterministic_oracle:
                estimate_history = estimate_history[self.evaluator.queried_oracle_]
        else:
            if self.deterministic_oracle:
                queried_oracle = self.evaluator.queried_oracle_
            else:
                queried_oracle = np.full_like(self.evaluator.queried_oracle_, True)
        return RunResult(estimate_history, queried_oracle, kl_div_history=kl_div_history)


class PassiveExperiment(BaseExperiment):
    def __init__(self, pool: BasePool, true_label_dist: ndarray, n_queries: Union[int, Iterable, ndarray], 
                 n_repeats: int, measure: BaseMeasure, labels: Union[Iterable, ndarray, None] = None, 
                 dataname: Optional[str] = None, name: Optional[str] = None, compute_kl_div: bool = False, 
                 deterministic_oracle: bool = True, **kwargs) -> None:
        super().__init__(pool, true_label_dist, n_queries, n_repeats, measure, labels=labels, dataname=dataname,
                         name=name, compute_kl_div=compute_kl_div, deterministic_oracle=deterministic_oracle, **kwargs)
        self.name = 'Passive' if name is None else name

    @property
    def n_queries(self) -> Union[int, Iterable, ndarray]:
        # More efficient to query all labels in one round. Makes no difference
        # since proposal is static.
        return np.atleast_1d(np.sum(self._n_queries))

    @property
    def path(self) -> str:
        return (self.dataname + '_passive' + 
                '_deterministic' if self.deterministic_oracle else '_non-deterministic')

    def _setup_evaluator(self, **kwargs) -> Evaluator:
        proposal = Passive(self.pool)
        return Evaluator(self.pool, self.measure, proposal, estimator=None)


class AISIExperiment(BaseExperiment):
    def __init__(self, pool: BasePartitionedPool, true_label_dist: ndarray, n_queries: Union[int, Iterable, ndarray],
                 n_repeats: int, measure: BaseMeasure, labels: Union[Iterable, ndarray, None] = None, 
                 dataname: Optional[str] = None, name: Optional[str] = None, compute_kl_div: bool = False, 
                 deterministic_oracle: bool = True, **kwargs) -> None:
        super().__init__(pool, true_label_dist, n_queries, n_repeats, measure, labels=labels, dataname=dataname,
                         name=name, compute_kl_div=compute_kl_div, deterministic_oracle=deterministic_oracle, **kwargs)
        self.name = 'AIS-I' if name is None else name

    @property
    def path(self) -> str:
        oracle_estimator = self.evaluator.proposal.oracle_estimator
        return (self.dataname +
                '_ais-i_{}-blocks_{}-prior-strength_' \
                '{}-smoothing-constant'.format(self.pool.n_blocks,
                                               oracle_estimator.prior_strength,
                                               oracle_estimator.smoothing_constant) + 
                '_deterministic' if self.deterministic_oracle else '_non-deterministic')

    def _setup_evaluator(self, **kwargs) -> Evaluator:
        allowed_kwargs = ['prior_strength', 'smoothing_constant', 'prior']
        kwargs = {kwarg: v for kwarg, v in kwargs.items() if kwarg in allowed_kwargs}
        oracle_estimator = PartitionedIndepOE(self.pool, self.labels, **kwargs)
        proposal = PartitionedAdaptiveVarMin(self.pool, self.measure, oracle_estimator)
        return Evaluator(self.pool, self.measure, proposal, estimator=None)


class AISHShallowExperiment(BaseExperiment):
    def __init__(self, pool: BasePartitionedPool, true_label_dist: ndarray, n_queries: Union[int, Iterable, ndarray],
                 n_repeats: int, measure: BaseMeasure, labels: Union[Iterable, ndarray, None] = None, 
                 dataname: Optional[str] = None, name: Optional[str] = None, compute_kl_div: bool = False, 
                 deterministic_oracle: bool = True, **kwargs) -> None:
        super().__init__(pool, true_label_dist, n_queries, n_repeats, measure, labels=labels, dataname=dataname,
                         name=name, compute_kl_div=compute_kl_div, deterministic_oracle=deterministic_oracle, **kwargs)
        self.name = 'AIS-H-Shallow' if name is None else name

    @property
    def path(self) -> str:
        oracle_estimator = self.evaluator.proposal.oracle_estimator
        return (self.dataname +
                '_ais-h-shallow_{}-blocks_{}-prior-strength_' \
                '{}-smoothing-constant'.format(self.pool.n_blocks,
                                               oracle_estimator.prior_strength,
                                               oracle_estimator.smoothing_constant) + 
                '_deterministic' if self.deterministic_oracle else '_non-deterministic')

    def _setup_evaluator(self, **kwargs) -> Evaluator:
        allowed_kwargs = ['prior_strength', 'smoothing_constant', 'prior']

        if self.deterministic_oracle:
            allowed_kwargs.extend(['em_tol', 'em_max_iter'])
            kwargs = {kwarg: v for kwarg, v in kwargs.items() if kwarg in allowed_kwargs}
            oracle_estimator = PartitionedDeterministicOE(self.pool, self.labels, **kwargs)
        else:
            kwargs = {kwarg: v for kwarg, v in kwargs.items() if kwarg in allowed_kwargs}
            oracle_estimator = PartitionedStochasticOE(self.pool, self.labels, **kwargs)
        proposal = AdaptiveVarMin(self.pool, self.measure, oracle_estimator)
        return Evaluator(self.pool, self.measure, proposal, estimator=None)


class ISExperiment(BaseExperiment):
    def __init__(self, pool: BasePool, true_label_dist: ndarray, n_queries: Union[int, Iterable, ndarray], 
                 n_repeats: int, measure: BaseMeasure, labels: Union[Iterable, ndarray, None] = None, 
                 dataname: Union[str, None] = None, name: Union[str, None] = None, compute_kl_div: bool = False, 
                 deterministic_oracle: bool = True, **kwargs) -> None:
        super().__init__(pool, true_label_dist, n_queries, n_repeats, measure, labels=labels, dataname=dataname,
                         name=name, compute_kl_div=compute_kl_div, deterministic_oracle=deterministic_oracle, **kwargs)
        self.name = 'IS' if name is None else name

    @property
    def path(self) -> str:
        return (self.dataname + '_is' + 
                '_deterministic' if self.deterministic_oracle else '_non-deterministic')

    def _setup_evaluator(self, **kwargs) -> Evaluator:
        epsilon = kwargs.get('epsilon', 1e-9)
        response_est = kwargs['prior']
        proposal = StaticVarMin(self.pool, self.measure, response_est, labels=self.labels, epsilon=epsilon, 
                                deterministic=self.deterministic_oracle)
        return Evaluator(self.pool, self.measure, proposal, estimator=None)


class StratifiedExperiment(BaseExperiment):
    def __init__(self, pool: BasePartitionedPool, true_label_dist: ndarray, n_queries: Union[int, Iterable, ndarray],
                 n_repeats: int, measure: BaseMeasure, labels: Union[Iterable, ndarray, None] = None, 
                 dataname: Optional[str] = None, name: Optional[str] = None, compute_kl_div: bool = False, 
                 deterministic_oracle: bool = True, **kwargs) -> None:
        if compute_kl_div and deterministic_oracle:
            warnings.warn("cannot compute kl divergence when oracle is deterministic")
            compute_kl_div = False
        super().__init__(pool, true_label_dist, n_queries, n_repeats, measure, labels=labels, dataname=dataname,
                         name=name, compute_kl_div=compute_kl_div, deterministic_oracle=deterministic_oracle, **kwargs)
        self.name = 'Stratified' if name is None else name

    @property
    def path(self) -> str:
        return (self.dataname + 
                '_stratified_{}-blocks'.format(self.pool.n_blocks) + 
                '_deterministic' if self.deterministic_oracle else '_non-deterministic')

    def _setup_evaluator(self, **kwargs) -> Evaluator:
        estimator = StratifiedEstimator(self.measure, self.pool)
        # Perform sampling without replacement for a deterministic oracle
        proposal = Passive(self.pool, replace=not self.deterministic_oracle)
        return Evaluator(self.pool, self.measure, proposal, estimator=estimator)


class AISHExperiment(BaseExperiment):
    def __init__(self, pool: BaseHierarchicalPool, true_label_dist: ndarray, n_queries: Union[int, Iterable, ndarray],
                 n_repeats: int, measure: BaseMeasure, labels: Union[Iterable, ndarray, None] = None, 
                 dataname: Optional[str] = None, name: Optional[str] = None, compute_kl_div: bool = False, 
                 deterministic_oracle: bool = True, **kwargs) -> None:
        super().__init__(pool, true_label_dist, n_queries, n_repeats, measure, labels=labels, dataname=dataname,
                         name=name, compute_kl_div=compute_kl_div, deterministic_oracle=deterministic_oracle, **kwargs)
        self.name = 'AIS-H' if name is None else name

    @property
    def path(self) -> str:
        oracle_estimator = self.evaluator.proposal.oracle_estimator
        return (self.dataname + 
                '_ais-h_{}-depth_{}-size_{}-blocks_' \
                '{}-prior-strength_{}-smoothing_constant_' \
                '{}-tree-prior-type'.format(self.pool.tree.depth(),
                                            self.pool.tree.size(),
                                            self.pool.n_blocks,
                                            oracle_estimator.prior_strength,
                                            oracle_estimator.smoothing_constant,
                                            oracle_estimator.tree_prior_type) + 
                '_deterministic' if self.deterministic_oracle else '_non-deterministic')

    def _setup_evaluator(self, **kwargs) -> Evaluator:
        allowed_kwargs = ['prior_strength', 'smoothing_constant', 'tree_prior_type', 'prior']
        if self.deterministic_oracle:
            allowed_kwargs.extend(['em_tol', 'em_max_iter'])
            kwargs = {kwarg: v for kwarg, v in kwargs.items() if kwarg in allowed_kwargs}
            oracle_estimator = HierarchicalDeterministicOE(self.pool, self.labels, **kwargs)
        else:
            kwargs = {kwarg: v for kwarg, v in kwargs.items() if kwarg in allowed_kwargs}
            oracle_estimator = HierarchicalStochasticOE(self.pool, self.labels, **kwargs)

        proposal = AdaptiveVarMin(self.pool, self.measure, oracle_estimator)
        return Evaluator(self.pool, self.measure, proposal, estimator=None)


def result_to_dataframe(expt: BaseExperiment, true_measure, count_queries: bool = True, n_processes: int = 1):
    """Summarizes the result of an experiment. 
    
    Parameters
    ----------
    expt : an instance of BaseExperiment
        Experiment
    
    true_measure : float or numpy.ndarray, shape=(n_dim_g,)
        Population value of measure
    
    count_queries : bool
        Whether to count queries (True) or samples (False)

    n_processes : int
        Number of parallel processes to use

    Returns
    -------
    df : pandas.DataFrame
        A dataframe containing the following columns: 
            'method': name of the evaluation method, 
            'query': cumulative number of queries,
            'sq_err': sum-squared-error of the estimate,
            'kl_div': kl-divergence from optimal proposal to approximation.
    """
    # Ensure scalar is represented as a length 1 array
    true_measure = np.atleast_1d(true_measure)

    def get_summary(seed: int):
        result = expt.result(seed, queried_only=count_queries)
        est = result.estimate_history
        if est.ndim == 1: # measure is a scalar
            est = est[:, np.newaxis]
        sq_err_history = np.add.reduce((est - true_measure[np.newaxis, :]) ** 2, axis=1)
        query_ids = np.arange(sq_err_history.size)
        kl_div_history = result.kl_div_history
        return sq_err_history, query_ids, kl_div_history

    pool = ProcessPool(n_processes)
    results = pool.amap(get_summary, range(expt.n_repeats))
    while not results.ready():
        time.sleep(5); print(".", end=' ')
    print()
    results = results.get()

    sq_err_histories = []
    queries = []
    kl_div_histories = []
    for sq_err_history, query_ids, kl_div_history in results:
        sq_err_histories.append(sq_err_history)
        queries.append(query_ids)
        if kl_div_history is not None:
            kl_div_histories.append(kl_div_history)

    # Index queries starting from 1 (not 0) in dataframe
    query = np.concatenate(queries).squeeze() + 1
    sq_err = np.concatenate(sq_err_histories).squeeze()

    df = pd.DataFrame({'method': expt.name, 'query': query, 'sq_err': sq_err})

    if len(kl_div_histories) > 0:
        kl_div_histories = np.array(kl_div_histories)
        if (kl_div_histories.shape[1]) != np.sum(expt.n_queries):
            kl_div_histories = np.repeat(kl_div_histories, expt.n_queries, axis=1)
        df['kl_div'] = kl_div_histories.ravel()

    return df
