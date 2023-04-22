import sys
import os

import jax

from momi3.utils import get_CPU_name, get_GPU_name, msprime_chromosome_simulator
from momi3.MOMI import Momi
from momi3.Params import Params

from timeit_methods import get_momi_times, get_moments_times, get_dadi_esfs, Return

from joblib import cpu_count

import pickle
import demes
import hashlib

EPS = 0.1
NUM_REPLICATES = 10
RANDOM_SEED = 108
RR = 1e-8  # Recombination Rate
MR = 1e-8  # Mutation Rate
GRID_POINTS = 250
BATCH_SIZE = 1500
CPU = get_CPU_name()
GPU = get_GPU_name()


def get_demo_2():
    return demes.load('../timing_tests_yaml/2_pop_mig.yaml')


def get_demo_3():
    return demes.load('../timing_tests_yaml/3_pop_mig.yaml')


def get_demo_4():
    return demes.load('../timing_tests_yaml/4_pop_mig.yaml')


def get_demo_5():
    return demes.load('../timing_tests_yaml/5_pop_mig.yaml')


def get_params(sampled_demes, demo):

    momi = Momi(
        demo,
        sampled_demes=sampled_demes,
        sample_sizes=[1 for _ in sampled_demes]
    )

    # Init params and set params to train
    params = Params(momi)
    params.set_train_all_rhos(True)
    params.set_train_all_etas(True)
    params.set_train_all_taus(True)
    params.set_train('tau_0', False)

    return params


if __name__ == "__main__":
    # python tests/time_5_pop_admixture.py <method> <number of demes> <sample size> <number of positions> <number of replications> <save folder>
    # e.g. python tests/time_IWM_constant.py momi3 3 4 100 10 /tmp/
    args = sys.argv[1:]
    method = args[0]
    ndemes = int(args[1])
    n = int(args[2])
    sequence_length = int(float(args[3]))
    save_path = args[4]

    model_name = "IWM_constant"

    if ndemes == 2:
        demo = get_demo_2()
    elif ndemes == 3:
        demo = get_demo_3()
    elif ndemes == 4:
        demo = get_demo_4()
    elif ndemes == 5:
        demo = get_demo_5()
    else:
        raise ValueError('ndemes should be in [2, 3, 4, 5]')

    sampled_demes = demo.metadata["sampled_demes"]

    params = get_params(sampled_demes, demo)

    sample_sizes = ndemes * [n]
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
    elif method == "moments":
        ret = get_moments_times(sampled_demes, sample_sizes, jsfs, params, NUM_REPLICATES, loglik_with_grad=True)
    elif method == "dadi":
        if len(sampled_demes) > 3:
            raise ValueError('Cant handle more than 3 pops')
        else:
            ret = get_dadi_esfs(sampled_demes, sample_sizes, jsfs, params, pts=GRID_POINTS)
    else:
        raise ValueError(f"Unknown {method=}")

    ret = ret | opts
    ret = Return(**ret)

    with open(save_path, 'wb') as f:
        pickle.dump(ret, f)
