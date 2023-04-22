import sys
import os

import jax

from momi3.utils import get_CPU_name, get_GPU_name, msprime_chromosome_simulator
from momi3.MOMI import Momi
from momi3.Params import Params

from timeit_methods import get_momi_times, get_momi2_times, get_moments_times, Return

from joblib import cpu_count

import pickle
import timeit
import demes
import hashlib

import momi as momi2
import autograd.numpy as np

EPS = 0.1
NUM_REPLICATES = 10
RANDOM_SEED = 108
BATCH_SIZE = None
RR = 1e-8  # Recombination Rate
MR = 1e-8  # Mutation Rate
CPU = get_CPU_name()
GPU = get_GPU_name()


def momi2_model(theta_train, train_keys, PD):
    PD.update(dict(zip(train_keys, theta_train)))
    g_chb = np.log(PD['eta_8'] / PD['eta_7']) / (PD['tau_2'])
    g_ceu = np.log(PD['eta_6'] / PD['eta_5']) / (PD['tau_2'])

    # Momi 2 model building
    model1 = momi2.DemographicModel(N_e=PD["eta_0"], muts_per_gen=None)

    for i in range(9):
        if i in [5, 7]:
            pass
        else:
            eta = f"eta_{i}"
            model1.add_size_param(eta, PD[eta])

    for i in range(8):
        tau = f"tau_{i}"
        model1.add_size_param(tau, PD[tau])

    model1.add_growth_param("g_chb", g_chb)
    model1.add_growth_param("g_ceu", g_ceu)

    model1.add_pulse_param("pi_1", PD["pi_1"])
    model1.add_pulse_param("pi_0", PD["pi_0"])

    model1.add_leaf('YRI', N='eta_1')
    model1.add_leaf('CEU', N='eta_6', g='g_ceu')
    model1.add_leaf('CHB', N='eta_8', g='g_chb')
    model1.add_leaf('ArchaicAFR', N="eta_3")
    model1.add_leaf('Neanderthal', N='eta_2')

    model1.move_lineages("CEU", "Neanderthal", t="tau_1", p='pi_1')
    model1.move_lineages("CHB", "CEU", t='tau_2', N="eta_4")
    model1.move_lineages("CEU", "YRI", t='tau_3', N="eta_1")
    model1.move_lineages("Neanderthal", "YRI", t="tau_4", p='pi_0')
    model1.set_size('YRI', N='eta_0', t='tau_5')
    model1.move_lineages("ArchaicAFR", "YRI", t="tau_6", N="eta_0")
    model1.move_lineages("Neanderthal", "YRI", t="tau_7", N="eta_0")

    # momi2.DemographyPlot(model1, ['p1', 'p2', 'p3', 'p4', 'p5'])

    model1._mem_chunk_size = 10000

    return model1


def get_demo():
	return demes.load('../timing_tests_yaml/5_pop_2_admix.yaml')


def get_params():

    demo = get_demo()

    momi = Momi(
        demo,
        sampled_demes=demo.metadata['sampled_demes'],
        sample_sizes=5 * [1]
    )

    # Init params and set params to train
    params = Params(momi)
    params.set_train_all_etas(True)
    params.set_train_all_pis(True)
    params.set_train_all_taus(True)
    params.set_train('tau_0', False)

    return params


if __name__ == "__main__":
    # python tests/time_5_pop_admixture.py <method> <sample size> <number of positions> <number of replications> <save folder>
    # e.g. python tests/time_5_pop_admixture.py momi2 4 100 10 /tmp/
    args = sys.argv[1:]
    method = args[0]
    n = int(args[1])
    sequence_length = int(float(args[2]))
    save_path = args[3]
    time = int(timeit.default_timer())

    model_name = 'demes5pulses2'

    demo = get_demo()
    params = get_params()
    sampled_demes = demo.metadata['sampled_demes']
    sample_sizes = 5 * [n]

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
        n=n,
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

    my_tuple = tuple(sorted(opts.items()))
    file_hash = hashlib.sha256(str(my_tuple).encode()).hexdigest()
    save_path = os.path.join(save_path, f"{model_name}_{file_hash}.pickle")

    if method == "momi3":
    	ret = get_momi_times(sampled_demes, sample_sizes, jsfs, params, BATCH_SIZE, NUM_REPLICATES, loglik_with_grad=True)
    elif method == "momi2":
    	ret = get_momi2_times(sampled_demes, sample_sizes, jsfs, params, momi2_model, NUM_REPLICATES, loglik_with_grad=True)
    elif method == "moments":
    	ret = get_moments_times(sampled_demes, sample_sizes, jsfs, params, NUM_REPLICATES, loglik_with_grad=True)
    else:
    	raise ValueError(f"Unknown {method=}")

    ret = ret | opts
    ret = Return(**ret)

    print(save_path)

    with open(save_path, 'wb') as f:
    	pickle.dump(ret, f)
