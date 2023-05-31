import os
import sys
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

sys.path.append(".")

import demes
import moments
import dadi
import momi as momi2
import timeit
from scipy.optimize import approx_fprime

from momi3.MOMI import Momi
from momi3.utils import update, Parallel_runtime

import autograd.numpy as np
from autograd import grad as auto_grad
from collections import namedtuple
from copy import deepcopy

Option_keys = [
    'model_name',
    'method',
    'n',
    'n_sampled_demes',
    'sequence_length',
    'recombination_rate',
    'mutation_rate',
    'sim_seed',
    'CPU',
    'CPU_count',
    'GPU',
    'n_devices',
    'jsfs_density',
    'grad_params'
]

Method_keys = [
    'esfs',
    'loglik_compilation_time',
    'loglik_time',
    'loglik_with_gradient_compilation_time',
    'loglik_with_gradient_time',
    'likelihood_value',
    'gradient_values'
]

Return = namedtuple(
    'Return', Option_keys + Method_keys
)


class momi2_timeit:
    def __init__(self, sampled_demes, sample_sizes, jsfs, params, model, num_replicates):
        PD = {key: params[key].num for key in params}
        train_keys = params._train_keys
        momi2_model = lambda theta_train: model(theta_train, train_keys=train_keys, PD=PD)
        num_sample = dict(zip(sampled_demes, sample_sizes))
        self.train_keys = train_keys
        self.sampled_demes = sampled_demes
        self.sample_sizes = sample_sizes
        self.theta_train = np.array(params._theta_train)
        self.num_replicates = num_replicates
        m2m = momi2_model(params._theta_train)

        def momi2_model_func(x):
            mdl = momi2_model(x)._get_demo(num_sample)
            return mdl

        def loglik(theta_train, sfs):
            return momi2.likelihood._composite_log_likelihood(sfs, momi2_model_func(theta_train))
        self._loglik = loglik

        def _get_esfs(sfs):
            m2m.set_data(sfs, length=1)
            m2m.set_mut_rate(1.0)
            return m2m.expected_sfs()
        self._get_esfs = _get_esfs

        def grad(theta_train, sfs):
            return auto_grad(lambda x: momi2.likelihood._composite_log_likelihood(
                sfs, momi2_model_func(x))
            )(theta_train)
        self._grad = grad

    def esfs(self, jsfs):
        coords = dict(zip(self.sampled_demes, jsfs.nonzero()))
        n = dict(zip(self.sampled_demes, self.sample_sizes))
        sfs = self.get_data(jsfs)
        myd = self._get_esfs(sfs)
        b = []
        for i in range(jsfs.nnz):
            ind = tuple((n[j] - coords[j][i], coords[j][i]) for j in self.sampled_demes)
            b.append(myd[ind])
        return b

    def loglik(self, jsfs):
        sfs = self.get_data(jsfs)
        return self._loglik(self.theta_train, sfs)

    def grad(self, jsfs):
        sfs = self.get_data(jsfs)
        g = self._grad(self.theta_train, sfs)
        return dict(zip(self.train_keys, [float(i) for i in g]))

    def time_loglik(self, jsfs):
        sfs = self.get_data(jsfs)
        f = lambda: self._loglik(self.theta_train, sfs)
        return Parallel_runtime(f, num_replicates=self.num_replicates, n_jobs=self.num_replicates)

    def time_grad(self, jsfs):
        sfs = self.get_data(jsfs)
        f = lambda: self._grad(self.theta_train, sfs)
        return Parallel_runtime(f, num_replicates=self.num_replicates, n_jobs=self.num_replicates)

    def get_data(self, jsfs):
        data = np.array(jsfs.data)
        coords = np.array(jsfs.coords.T)
        n = [i - 1 for i in jsfs.shape]
        P = len(n)
        config_list = {
            tuple((n[j] - coord[j], coord[j]) for j in range(P)): val for coord, val in zip(coords, data)
        }
        sfs = momi2.site_freq_spectrum(self.sampled_demes, [config_list])
        return sfs


class dadi_timeit:
    def __init__(self, sampled_demes, sample_sizes, jsfs, params, num_replicates):
        demo_dict = params.demo_dict
        self.EPS = 1e-5
        self.demo = params.demo_graph
        self.num_replicates = num_replicates
        ndemes = len(sampled_demes)
        n = sample_sizes[0]
        if ndemes == 5:
            pts = 30
        elif ndemes == 4:
            pts = 80
        elif ndemes == 3:
            pts = 50
        elif ndemes == 2:
            pts = 50
        pts = max(pts, n)

        self.pts = pts  # min(max_pts, dadi.RAM_to_pts(1, ndemes))

        keys = list(params._theta_train_path_dict())
        self.theta_train = np.array(list(params._theta_train_path_dict().values()))

        def get_esfs(demo):
            esfs = dadi.Spectrum.from_demes(
                demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes, pts=self.pts
            )
            return esfs * 4 * demo.demes[0].epochs[0].start_size

        def loglik(theta_train, jsfs_flatten, demo_dict=demo_dict):
            demo_dict = deepcopy(demo_dict)
            theta_train_dict = dict(zip(keys, theta_train))

            for paths, val in theta_train_dict.items():
                for path in paths:
                    update(demo_dict, path, float(val))

            demo = demes.Builder.fromdict(demo_dict).resolve()
            esfs = get_esfs(demo)
            esfs = np.array(esfs).flatten()[1:-1]
            data = jsfs_flatten[1:-1]
            esfs /= esfs.sum()
            esfs = np.clip(1e-32, np.inf, esfs)
            return (data * np.log(esfs)).sum()

        self._get_esfs = get_esfs
        self._loglik = loglik

        def grad(theta_train, jsfs_flatten):
            # return [loglik(theta_train, jsfs_flatten) for _ in range(2 * len(theta_train))]
            return approx_fprime(
                theta_train, loglik, self.EPS, jsfs_flatten
            )
        self._grad = grad

    def esfs(self, jsfs):
        return list(self._get_esfs(self.demo)[tuple(jsfs.coords)])

    def loglik(self, jsfs):
        jsfs_flatten = self.flatten_jsfs(jsfs)
        return self._loglik(self.theta_train, jsfs_flatten)

    def grad(self, jsfs):
        jsfs_flatten = self.flatten_jsfs(jsfs)
        return self._grad(self.theta_train, jsfs_flatten)

    def time_loglik(self, jsfs):
        jsfs_flatten = self.flatten_jsfs(jsfs)
        f = lambda: self._loglik(self.theta_train, jsfs_flatten)
        run_time = timeit.repeat(f, repeat=self.num_replicates, number=1)
        return run_time

    def time_grad(self, jsfs):
        jsfs_flatten = self.flatten_jsfs(jsfs)
        f = lambda: self._grad(self.theta_train, jsfs_flatten)
        run_time = timeit.repeat(f, repeat=self.num_replicates, number=1)
        return run_time

    def flatten_jsfs(self, jsfs):
        return jsfs.todense().flatten()


class dadi_just_esfs:
    def __init__(self, sampled_demes, sample_sizes, params):
        self.demo = params.demo_graph
        self.sampled_demes = sampled_demes
        self.sample_sizes = sample_sizes

    def esfs(self, jsfs, pts):
        esfs = dadi.Spectrum.from_demes(
            self.demo, sampled_demes=self.sampled_demes, sample_sizes=self.sample_sizes, pts=pts
        )
        esfs = esfs * 4 * self.demo.demes[0].epochs[0].start_size

        return list(esfs[tuple(jsfs.coords)])


class moments_timeit:
    def __init__(self, sampled_demes, sample_sizes, jsfs, params, num_replicates):
        demo_dict = params.demo_dict
        self.EPS = 0.1
        self.demo = params.demo_graph
        self.num_replicates = num_replicates

        keys = list(params._theta_train_path_dict())
        self.theta_train = np.array(list(params._theta_train_path_dict().values()))

        def get_esfs(demo):
            esfs = moments.Spectrum.from_demes(
                demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes
            )
            return esfs * 4 * demo.demes[0].epochs[0].start_size

        def loglik(theta_train, jsfs_flatten, demo_dict=demo_dict):
            demo_dict = deepcopy(demo_dict)
            theta_train_dict = dict(zip(keys, theta_train))

            for paths, val in theta_train_dict.items():
                for path in paths:
                    update(demo_dict, path, float(val))

            demo = demes.Builder.fromdict(demo_dict).resolve()
            esfs = get_esfs(demo)
            esfs = np.array(esfs).flatten()[1:-1]
            data = jsfs_flatten[1:-1]
            esfs /= esfs.sum()
            esfs = np.clip(1e-32, np.inf, esfs)
            return (data * np.log(esfs)).sum()

        self._get_esfs = get_esfs
        self._loglik = loglik

        def grad(theta_train, jsfs_flatten):
            # return [loglik(theta_train, jsfs_flatten) for _ in range(2 * len(theta_train))]
            return approx_fprime(
                theta_train, loglik, self.EPS, jsfs_flatten
            )
        self._grad = grad

    def esfs(self, jsfs):
        return list(self._get_esfs(self.demo)[tuple(jsfs.coords)])

    def loglik(self, jsfs):
        jsfs_flatten = self.flatten_jsfs(jsfs)
        return self._loglik(self.theta_train, jsfs_flatten)

    def grad(self, jsfs):
        jsfs_flatten = self.flatten_jsfs(jsfs)
        return self._grad(self.theta_train, jsfs_flatten)

    def time_loglik(self, jsfs):
        jsfs_flatten = self.flatten_jsfs(jsfs)
        f = lambda: self._loglik(self.theta_train, jsfs_flatten)
        return Parallel_runtime(f, num_replicates=self.num_replicates, n_jobs=self.num_replicates)

    def time_grad(self, jsfs):
        jsfs_flatten = self.flatten_jsfs(jsfs)
        f = lambda: self._grad(self.theta_train, jsfs_flatten)
        return Parallel_runtime(f, num_replicates=self.num_replicates, n_jobs=self.num_replicates)

    def flatten_jsfs(self, jsfs):
        return jsfs.todense().flatten()


class momi_timeit:
    def __init__(self, sampled_demes, sample_sizes, jsfs, params, batch_size, num_replicates, bound_sample_args=None):
        if batch_size is not None:
            low_memory = True
        else:
            low_memory = False

        demo = params.demo_graph
        self.params = params
        self.momi = Momi(demo, sampled_demes, sample_sizes, jitted=True, batch_size=batch_size, low_memory=low_memory)

        if bound_sample_args is not None:
            bounds = self.momi.bound_sampler(self.params, **bound_sample_args)
            self.momi = self.momi.bound(bounds)

        self.batch_size = batch_size
        self.num_replicates = num_replicates

    def time(self, f):
        return timeit.repeat(
            f,
            number=1,
            repeat=self.num_replicates
        )

    def get_esfs(self, jsfs):
        num_deriveds = {}
        for i, pop in enumerate(self.momi.sampled_demes):
            num_deriveds[pop] = np.array(jsfs.coords[i])
        theta_dict = self.params._theta_path_dict
        return self.momi._JAX_functions.esfs(theta_dict, num_deriveds, None).tolist()

    def time_loglik(self, jsfs):
        v, compilation_time, run_time = self.momi._time_loglik(
            self.params, jsfs, repeat=self.num_replicates, average=False
        )
        return float(v), compilation_time, run_time

    def time_loglik_with_grad(self, jsfs):
        (v, g), compilation_time, run_time = self.momi._time_loglik_with_gradient(
            self.params, jsfs, repeat=self.num_replicates, average=False
        )
        g = {i: float(g[i]) for i in g}
        return float(v), g, compilation_time, run_time


def get_momi_times(sampled_demes, sample_sizes, jsfs, params, batch_size, num_replicates, loglik_with_grad=True, bound_sample_args=None):

    momit = momi_timeit(sampled_demes, sample_sizes, jsfs, params, batch_size, num_replicates, bound_sample_args)

    esfs = momit.get_esfs(jsfs)

    # Get compilation time for value and grad
    if loglik_with_grad:
        v, g, g_compilation_time, g_run_time = momit.time_loglik_with_grad(jsfs)
    else:
        g = None
        g_compilation_time = None
        g_run_time = None

    v, compilation_time, run_time = momit.time_loglik(jsfs)

    momi3_ret = dict(
        esfs=esfs,
        loglik_compilation_time=compilation_time,
        loglik_time=run_time,
        loglik_with_gradient_compilation_time=g_compilation_time,
        loglik_with_gradient_time=g_run_time,
        likelihood_value=v,
        gradient_values=g,
    )

    return momi3_ret


def get_momi2_times(sampled_demes, sample_sizes, jsfs, params, momi2_model, num_replicates, loglik_with_grad=True):
    momi2t = momi2_timeit(sampled_demes, sample_sizes, jsfs, params, momi2_model, num_replicates)

    esfs = momi2t.esfs(jsfs)

    # Get grad times
    times = momi2t.time_loglik(jsfs)

    if loglik_with_grad:
        grad_times = momi2t.time_grad(jsfs)
        grad_with_loglik_time = [i + j for i, j in zip(times, grad_times)]
        g = momi2t.grad(jsfs)
    else:
        grad_with_loglik_time = None
        g = None

    # Get values
    v = momi2t.loglik(jsfs)

    momi2_ret = dict(
        esfs=esfs,
        loglik_compilation_time=None,
        loglik_time=times,
        loglik_with_gradient_compilation_time=None,
        loglik_with_gradient_time=grad_with_loglik_time,
        likelihood_value=v,
        gradient_values=g,
    )

    return momi2_ret


def get_moments_times(sampled_demes, sample_sizes, jsfs, params, num_replicates, loglik_with_grad=True):
    momentst = moments_timeit(sampled_demes, sample_sizes, jsfs, params, num_replicates)

    # Get esfs
    esfs = momentst.esfs(jsfs)

    times = momentst.time_loglik(jsfs)

    if loglik_with_grad:
        grad_times = momentst.time_grad(jsfs)
        grad_with_loglik_time = [grad_times[i] + times[i] for i in range(len(times))]
    else:
        grad_with_loglik_time = None

    # Get values
    v = momentst.loglik(jsfs)

    moments_ret = dict(
        esfs=esfs,
        loglik_compilation_time=None,
        loglik_time=times,
        loglik_with_gradient_compilation_time=None,
        loglik_with_gradient_time=grad_with_loglik_time,
        likelihood_value=v,
        gradient_values=None,
    )

    return moments_ret


# def get_dadi_esfs(sampled_demes, sample_sizes, jsfs, params, pts, num_replicates):
#     dadist = dadi_timeit(sampled_demes, sample_sizes, jsfs, params, num_replicates)

#     esfs = dadist.esfs(jsfs, pts)

#     dadi_ret = dict(
#         esfs=esfs,
#         loglik_compilation_time=None,
#         loglik_time=None,
#         loglik_with_gradient_compilation_time=None,
#         loglik_with_gradient_time=None,
#         likelihood_value=None,
#         gradient_values=None,
#     )

#     return dadi_ret


def get_dadi_times(sampled_demes, sample_sizes, jsfs, params, num_replicates, loglik_with_grad=True):
    dadist = dadi_timeit(sampled_demes, sample_sizes, jsfs, params, num_replicates)

    # Get esfs

    def get_esfs_dadi(demo, sampled_demes, sample_sizes, pts):
        esfs = dadi.Spectrum.from_demes(
            demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes, pts=pts
        )
        return esfs * 4 * demo.demes[0].epochs[0].start_size

    esfs = dadist.esfs(jsfs)

    times = dadist.time_loglik(jsfs)

    if loglik_with_grad:
        grad_times = dadist.time_grad(jsfs)
        grad_with_loglik_time = [grad_times[i] + times[i] for i in range(len(times))]
    else:
        grad_with_loglik_time = None

    # Get values
    v = dadist.loglik(jsfs)

    moments_ret = dict(
        esfs=esfs,
        loglik_compilation_time=None,
        loglik_time=times,
        loglik_with_gradient_compilation_time=None,
        loglik_with_gradient_time=grad_with_loglik_time,
        likelihood_value=v,
        gradient_values=None,
    )

    return moments_ret
