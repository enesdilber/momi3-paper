import sys
import os

import jax

from momi3.utils import get_CPU_name, get_GPU_name, msprime_simulator
from momi3.MOMI import Momi
from momi3.Params import Params

from timeit_methods import get_momi_times, Return

import pickle
import demes
import hashlib

EPS = 0.1
NUM_REPLICATES = 5
RANDOM_SEED = 108
BATCH_SIZE = 100000
CPU = get_CPU_name()
GPU = get_GPU_name()


def get_params(sampled_demes, demo):

    momi = Momi(
        demo,
        sampled_demes=sampled_demes,
        sample_sizes=[1 for _ in sampled_demes]
    )

    # Init params and set params to train
    params = Params(momi)
    # params.set_train_all_etas(True)
    # params.set_train_all_rhos(True)
    # params.set_train_all_taus(True)
    # params.set_train_all_pis(True)

    return params


def get_demo():
    demo = demes.load("../yaml_files/jacobson.yml")
    demo = demo.in_generations()
    return demo


if __name__ == "__main__":
    # python time_jacobson.py <method> <sample size> <number of positions> <number of replications> <save folder>
    # e.g. python time_jacobson.py momi3 5 2 /tmp/
    args = sys.argv[1:]
    method = args[0]
    n = int(args[1])
    npos = int(args[2])
    save_path = args[3]

    model_name = 'jacobson'

    demo = get_demo()
    sampled_demes = ['YRI', 'CEU', 'CHB', 'Papuan', 'Nea1', 'NeaA', 'Den2', 'Den1', 'DenA']
    params = get_params(sampled_demes, demo)
    sample_sizes = 9 * [n]
    jsfs = msprime_simulator(demo, sampled_demes, sample_sizes, npos, seed=RANDOM_SEED)

    opts = dict(
        model_name=model_name,
        method=method,
        n=n,
        n_sampled_demes=len(sampled_demes),
        npos=npos,
        sim_seed=RANDOM_SEED,
        CPU=CPU,
        GPU=GPU,
        n_devices=jax.device_count(),
        grad_params=tuple(params._train_keys),
        jsfs_density=jsfs.density
    )

    my_tuple = tuple(sorted(opts.items()))
    file_hash = hashlib.sha256(str(my_tuple).encode()).hexdigest()
    save_path = os.path.join(save_path, f"{model_name}_{file_hash}.pickle")

    if method == "momi3":
        ret = get_momi_times(sampled_demes, sample_sizes, jsfs, params, BATCH_SIZE, NUM_REPLICATES, loglik_with_grad=False)
    else:
        raise ValueError(f"Unknown {method=}")

    ret = ret | opts
    ret = Return(**ret)

    with open(save_path, 'wb') as f:
        pickle.dump(ret, f)
