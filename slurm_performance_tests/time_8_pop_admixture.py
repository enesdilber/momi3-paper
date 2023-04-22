import sys
import os

import jax

from momi3.utils import get_CPU_name, get_GPU_name, msprime_chromosome_simulator
from momi3.MOMI import Momi
from momi3.Params import Params

from timeit_methods import get_momi_times, get_momi2_times, get_moments_times, Option_keys, Method_keys

from joblib import cpu_count
from collections import namedtuple
from timeit import default_timer as time

import pickle
import timeit
import demes
import hashlib

import momi as momi2
import autograd.numpy as np


Option_keys = [
    'bound_sampler'
] + Option_keys

Method_keys = Method_keys

Return8 = namedtuple(
    'Return8', Option_keys + Method_keys
)

EPS = 0.1
NUM_REPLICATES = 10
RANDOM_SEED = 108
RR = 1e-8  # Recombination Rate
MR = 1e-8  # Mutation Rate
CPU = get_CPU_name()
GPU = get_GPU_name()


def momi2_model(theta_train, train_keys, PD):
    PD.update(dict(zip(train_keys, theta_train)))
    g_NEA = np.log(PD['eta_6'] / PD['eta_5']) / (PD['tau_11'] - PD['tau_6'])

    # Momi 2 model building
    model1 = momi2.DemographicModel(N_e=PD["eta_0"], muts_per_gen=None)

    for i in range(16):
        if i == 5:
            pass
        else:
            eta = f"eta_{i}"
            model1.add_size_param(eta, PD[eta])

    for i in range(13):
        tau = f"tau_{i}"
        model1.add_size_param(tau, PD[tau])

    model1.add_growth_param("g_NEA", g_NEA)
    
    model1.add_pulse_param("pi_2", PD["pi_2"])
    model1.add_pulse_param("pi_1", PD["pi_1"])
    model1.add_pulse_param("pi_0", PD["pi_0"])
   
    model1.add_leaf('Mbuti', N='eta_8')
    model1.add_leaf('Han', N='eta_11')
    model1.add_leaf('Sardinian', N='eta_15')
    model1.add_leaf('Loschbour', N='eta_3')
    model1.add_leaf('LBK', N='eta_14')
    model1.add_leaf('MA1', N="eta_12")
    model1.add_leaf('UstIshim', N='eta_10')
    model1.add_leaf('Neanderthal', N='eta_7')

    model1.set_size('BasalEurasian', N='eta_9', t='tau_0')
    
    model1.move_lineages("Sardinian", "Loschbour", t='tau_1', p='pi_2')
    model1.move_lineages("Sardinian", "LBK", t="tau_2")
    
    model1.set_size('LBK', N='eta_13', t='tau_2')
    
    model1.move_lineages("LBK", "BasalEurasian", t="tau_3", p='pi_1')
    model1.move_lineages("LBK", "Loschbour", t='tau_4')
    model1.move_lineages("MA1", "Loschbour", t='tau_5')

    model1.set_size('Neanderthal', N='eta_6', t='tau_6', g='g_NEA')

    model1.move_lineages("Han", "Loschbour", t="tau_7")
    
    model1.set_size('Loschbour', N='eta_2', t='tau_7')
    
    model1.move_lineages("UstIshim", "Loschbour", t="tau_8")
    model1.move_lineages("Loschbour", "Neanderthal", t="tau_9", p='pi_0')
    model1.move_lineages("BasalEurasian", "Loschbour", t="tau_10")
    model1.move_lineages("Mbuti", "Loschbour", t="tau_11")

    model1.set_size('Neanderthal', N='eta_4', t='tau_11')
    model1.set_size('Loschbour', N='eta_1', t='tau_11')

    model1.move_lineages("Neanderthal", "Loschbour", t="tau_12")

    model1.set_size('Loschbour', N='eta_0', t='tau_12')

    # momi2.DemographyPlot(model1, ['p1', 'p2', 'p3', 'p4', 'p5'])

    model1._mem_chunk_size = 5000

    return model1


def get_demo():
	return demes.load('../timing_tests_yaml/8_pop_3_admix.yaml')


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
    # python tests/time_5_pop_admixture.py <method> <sample size> <number of positions> <number of replications> <save folder>
    # e.g. python tests/time_8_pop_admixture.py momi2 4 100 10 /tmp/
    args = sys.argv[1:]
    method = args[0]
    extra_n = int(args[1])
    sequence_length = int(float(args[2]))
    save_path = args[3]

    bound_sample_args = None

    try:
        if args[4] == "bound_sampler":
            bound_sample_args = {'size': 100000, 'seed': 108, 'quantile': 0.99}
    except:
        pass

    model_name = 'momi2OOA'

    demo = get_demo()
    params = get_params()
    sampled_demes = demo.metadata['sampled_demes']
    sample_sizes = demo.metadata['sample_sizes']
    for i in range(3):
        sample_sizes[i] += extra_n

    jsfs = msprime_chromosome_simulator(
        demo=demo,
        sampled_demes=sampled_demes,
        sample_sizes=sample_sizes,
        sequence_length=sequence_length,
        recombination_rate=RR,
        mutation_rate=MR,
        seed=RANDOM_SEED
    )

    if method in ['momi2', 'moments']:
        GPU = None

    opts = dict(
        model_name=model_name,
        method=method,
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
        grad_params=tuple(params._train_keys),
        jsfs_density=jsfs.density
    )

    if method == "momi3":
        BATCH_SIZE = None
        if bound_sample_args is None:
            if extra_n > 18:
                BATCH_SIZE = min(18000, jsfs.nnz + 3)
        else:
            if GPU is not None:
                if extra_n > 90:
                    BATCH_SIZE = min(20000, jsfs.nnz + 3)
        ret = get_momi_times(
            sampled_demes,
            sample_sizes,
            jsfs,
            params,
            BATCH_SIZE,
            NUM_REPLICATES,
            loglik_with_grad=True,
            bound_sample_args=bound_sample_args
        )
    elif method == "momi2":
        ret = get_momi2_times(sampled_demes, sample_sizes, jsfs, params, momi2_model, NUM_REPLICATES, loglik_with_grad=True)
    else:
    	raise ValueError(f"Unknown {method=}")

    if bound_sample_args is None:
        opts['bound_sampler'] = False
    else:
        opts['bound_sampler'] = True

    my_tuple = tuple(sorted(opts.items()))
    file_hash = hashlib.sha256(str(my_tuple).encode()).hexdigest()
    save_path = os.path.join(save_path, f"{model_name}_{file_hash}.pickle")

    ret = ret | opts
    ret = Return8(**ret)

    print(save_path)
    with open(save_path, 'wb') as f:
    	pickle.dump(ret, f)
