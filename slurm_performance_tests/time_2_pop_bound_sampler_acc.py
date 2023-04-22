import sys
import os

import jax

from momi3.utils import get_CPU_name, get_GPU_name, msprime_chromosome_simulator, downsample_jsfs
from momi3.MOMI import Momi
from momi3.Params import Params

from timeit_methods import momi_timeit, get_momi2_times

from joblib import cpu_count
from collections import namedtuple
from timeit import default_timer as time

import pickle
import timeit
import demes
import hashlib

import momi as momi2
import autograd.numpy as np
import pandas as pd


def get_esfs(momi, params, jsfs):
    num_deriveds = {}
    for i, pop in enumerate(momi.sampled_demes):
        num_deriveds[pop] = np.array(jsfs.coords[i])
    theta_dict = params._theta_path_dict
    return momi._JAX_functions.esfs(theta_dict, num_deriveds, momi.batch_size).tolist()


Option_keys = [
    'model_name',
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
    'jsfs_density'
]

Method_keys = ['TV', 'GGV']

Return2pop = namedtuple(
    'Return2pop', Option_keys + Method_keys
)

key_params = [
    'eta_0',
    'eta_1',
    'eta_2',
    'eta_3',
    'pi_0',
    'pi_1',
    'tau_1',
    'tau_2',
    'tau_3',
    'tau_4',
]

key_params_range = {
    'eta_0': (2000, 5000),
    'eta_1': (7000, 20000),
    'eta_2': (100, 5000),
    'eta_3': (6000, 13000),
    'pi_0': (0.001, 0.1),
    'pi_1': (0.001, 0.1),
    'tau_1': (100, 500),
    'tau_2': (700, 1500),
    'tau_3': (1700, 2300),
    'tau_4': (7000, 14000)
}

bound_sample_args = {'size': 100000, 'seed': 108, 'quantile': 0.99}

EPS = 0.1
NUM_REPLICATES = 10
RANDOM_SEED = 108
BATCH_SIZE = None
RR = 1e-8  # Recombination Rate
MR = 1e-8  # Mutation Rate
CPU = get_CPU_name()
GPU = get_GPU_name()
PTS = 11


def get_demo():
	return demes.load('../timing_tests_yaml/2_pop_adm.yaml')


def get_params():

    demo = get_demo()

    momi = Momi(
        demo,
        sampled_demes=demo.metadata['sampled_demes'],
        sample_sizes=len(demo.metadata['sampled_demes']) * [1]
    )

    # Init params and set params to train
    params = Params(momi)
    params.set_train_all(True)
    return params


if __name__ == "__main__":
    # python time_2_pop_bound_sampler_acc.py 15 1e8 /tmp/
    args = sys.argv[1:]
    n = int(args[0])
    sequence_length = int(float(args[1]))
    save_path = args[2]

    model_name = 'pop2adm2'

    demo = get_demo()
    params = get_params()
    sampled_demes = demo.metadata['sampled_demes']
    sample_sizes = len(sampled_demes) * [n]

    jsfs = msprime_chromosome_simulator(
        demo=demo,
        sampled_demes=sampled_demes,
        sample_sizes=sample_sizes,
        sequence_length=sequence_length,
        recombination_rate=RR,
        mutation_rate=MR,
        seed=RANDOM_SEED
    )

    print(jsfs.nnz)

    opts = dict(
        model_name=model_name,
        n=sample_sizes[0],
        n_sampled_demes=len(sampled_demes),
        sequence_length=sequence_length,
        recombination_rate=RR,
        mutation_rate=MR,
        sim_seed=RANDOM_SEED,
        CPU=CPU,
        CPU_count=cpu_count(),
        GPU=GPU,
        n_devices=jax.device_count(),
        jsfs_density=jsfs.density
    )

    # if bound_sample_args is None:
    #     BATCH_SIZE = min(jsfs.nnz + 3, 18000)
    # else:
    #     bound_sample_args['scale'] = scale
    if n > 51:
        batch_size = min(2500, jsfs.nnz + 3)
        low_memory = True
    else:
        batch_size = None
        low_memory = False
    momi = Momi(
        demo, sampled_demes, sample_sizes, batch_size=batch_size, low_memory=low_memory, jitted=True
    )
    momi_b = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    bounds = momi_b.bound_sampler(params, **bound_sample_args)
    momi_b = momi_b.bound(bounds)

    grad_grid_values = {}
    esfs_TV = {}

    for param_key in key_params_range:
        grad_grid_values[param_key] = {}
        esfs_TV[param_key] = {}

        st, en = key_params_range[param_key]
        lins = [float(i) for i in np.linspace(st, en, PTS)]

        orig_val = params[param_key].num
        for val in lins:
            params.set(param_key, val)

            esfs = np.array(get_esfs(momi, params, jsfs))
            esfs_b = np.array(get_esfs(momi_b, params, jsfs))
            esfs /= esfs.sum()
            esfs_b /= esfs_b.sum()
            TV = np.sum(np.abs(esfs - esfs_b)) / 2

            v, g = momi.loglik_with_gradient(params, jsfs)
            gr = g[param_key]
            v, g = momi_b.loglik_with_gradient(params, jsfs)
            gr_b = g[param_key]

            grad_grid_values[param_key][val] = {'exact': gr, 'bound_sampler': gr_b}
            esfs_TV[param_key][val] = TV

        params.set(param_key, orig_val)

    ret = {'TV': esfs_TV, 'GGV': grad_grid_values}

    my_tuple = tuple(sorted(opts.items()))
    file_hash = hashlib.sha256(str(my_tuple).encode()).hexdigest()
    save_path = os.path.join(save_path, f"{model_name}_{file_hash}.pickle")

    ret = ret | opts
    ret = Return2pop(**ret)

    print(save_path)
    with open(save_path, 'wb') as f:
    	pickle.dump(ret, f)
