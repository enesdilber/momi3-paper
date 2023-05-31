import sys
import os

import jax

import momi as momi2

from momi3.utils import get_CPU_name, get_GPU_name, msprime_chromosome_simulator
from momi3.MOMI import Momi
from momi3.Params import Params

from timeit_methods import get_dadi_times, get_momi_times, get_momi2_times, get_moments_times, Return

from joblib import cpu_count

import pickle
import demes
import hashlib
import dadi

EPS = 0.1
NUM_REPLICATES = 10
RANDOM_SEED = 108
BATCH_SIZE = None
RR = 1e-8  # Recombination Rate
MR = 1e-8  # Mutation Rate
CPU = get_CPU_name()
GPU = get_GPU_name()


def get_demo_2():

    size = 5000
    t = 5000

    D = demes.Builder()
    D.add_deme("ANC", epochs=[dict(start_size=size, end_time=t)])
    D.add_deme("A", epochs=[dict(start_size=size)], ancestors=["ANC"])
    D.add_deme("B", epochs=[dict(start_size=size)], ancestors=["ANC"])

    demo = D.resolve()
    return demo


def get_demo_3():

    size = 5000
    t = 5000

    D = demes.Builder()
    D.add_deme("ANC", epochs=[dict(start_size=size, end_time=t)])
    D.add_deme("A", epochs=[dict(start_size=size)], ancestors=["ANC"])
    D.add_deme("B", epochs=[dict(start_size=size)], ancestors=["ANC"])
    D.add_deme("C", epochs=[dict(start_size=size)], ancestors=["ANC"])

    demo = D.resolve()
    return demo


def get_demo_4():

    size = 5000
    t = 5000

    D = demes.Builder()
    D.add_deme("ANC", epochs=[dict(start_size=size, end_time=t)])
    D.add_deme("A", epochs=[dict(start_size=size)], ancestors=["ANC"])
    D.add_deme("B", epochs=[dict(start_size=size)], ancestors=["ANC"])
    D.add_deme("C", epochs=[dict(start_size=size)], ancestors=["ANC"])
    D.add_deme("D", epochs=[dict(start_size=size)], ancestors=["ANC"])

    demo = D.resolve()
    return demo


def get_demo_5():

    size = 5000
    t = 5000

    D = demes.Builder()
    D.add_deme("ANC", epochs=[dict(start_size=size, end_time=t)])
    D.add_deme("A", epochs=[dict(start_size=size)], ancestors=["ANC"])
    D.add_deme("B", epochs=[dict(start_size=size)], ancestors=["ANC"])
    D.add_deme("C", epochs=[dict(start_size=size)], ancestors=["ANC"])
    D.add_deme("D", epochs=[dict(start_size=size)], ancestors=["ANC"])
    D.add_deme("E", epochs=[dict(start_size=size)], ancestors=["ANC"])

    demo = D.resolve()
    return demo


def get_model_2(theta_train, train_keys, PD):
    PD.update(dict(zip(train_keys, theta_train)))

    model1 = momi2.DemographicModel(N_e=PD['eta_0'], muts_per_gen=1)
    model1.add_leaf("A", N=PD['eta_1'])
    model1.add_leaf("B", N=PD['eta_2'])
    model1.move_lineages("B", "A", t=PD['tau_1'], N=PD['eta_0'])
    model1._mem_chunk_size = 10000
    return model1


def get_model_3(theta_train, train_keys, PD):
    PD.update(dict(zip(train_keys, theta_train)))

    model1 = momi2.DemographicModel(N_e=PD['eta_0'], muts_per_gen=1)
    model1.add_leaf("A", N=PD['eta_1'])
    model1.add_leaf("B", N=PD['eta_2'])
    model1.add_leaf("C", N=PD['eta_3'])
    model1.move_lineages("B", "A", t=PD['tau_1'], N=PD['eta_0'])
    model1.move_lineages("C", "A", t=PD['tau_1'], N=PD['eta_0'])
    model1._mem_chunk_size = 10000
    return model1


def get_model_4(theta_train, train_keys, PD):
    PD.update(dict(zip(train_keys, theta_train)))

    model1 = momi2.DemographicModel(N_e=PD['eta_0'], muts_per_gen=1)
    model1.add_leaf("A", N=PD['eta_1'])
    model1.add_leaf("B", N=PD['eta_2'])
    model1.add_leaf("C", N=PD['eta_3'])
    model1.add_leaf("D", N=PD['eta_4'])
    model1.move_lineages("B", "A", t=PD['tau_1'], N=PD['eta_0'])
    model1.move_lineages("C", "A", t=PD['tau_1'], N=PD['eta_0'])
    model1.move_lineages("D", "A", t=PD['tau_1'], N=PD['eta_0'])
    model1._mem_chunk_size = 10000
    return model1


def get_model_5(theta_train, train_keys, PD):
    PD.update(dict(zip(train_keys, theta_train)))

    model1 = momi2.DemographicModel(N_e=PD['eta_0'], muts_per_gen=1)
    model1.add_leaf("A", N=PD['eta_1'])
    model1.add_leaf("B", N=PD['eta_2'])
    model1.add_leaf("C", N=PD['eta_3'])
    model1.add_leaf("D", N=PD['eta_4'])
    model1.add_leaf("E", N=PD['eta_5'])
    model1.move_lineages("B", "A", t=PD['tau_1'], N=PD['eta_0'])
    model1.move_lineages("C", "A", t=PD['tau_1'], N=PD['eta_0'])
    model1.move_lineages("D", "A", t=PD['tau_1'], N=PD['eta_0'])
    model1.move_lineages("E", "A", t=PD['tau_1'], N=PD['eta_0'])
    return model1


def get_params(sampled_demes, demo):

    momi = Momi(
        demo,
        sampled_demes=sampled_demes,
        sample_sizes=[1 for _ in sampled_demes]
    )

    # Init params and set params to train
    params = Params(momi)
    # params.set_train_all_etas(True)
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

    model_name = "constant"

    if ndemes == 2:
        demo = get_demo_2()
        momi2_model = get_model_2
        sampled_demes = ["A", "B"]
    elif ndemes == 3:
        demo = get_demo_3()
        momi2_model = get_model_3
        sampled_demes = ["A", "B", "C"]
    elif ndemes == 4:
        demo = get_demo_4()
        momi2_model = get_model_4
        sampled_demes = ["A", "B", "C", "D"]
    elif ndemes == 5:
        demo = get_demo_5()
        momi2_model = get_model_5
        sampled_demes = ["A", "B", "C", "D", "E"]
    else:
        raise ValueError('ndemes should be in [2, 3, 4, 5]')

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

    print(jsfs.nnz)

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
    elif method == "momi2":
        ret = get_momi2_times(sampled_demes, sample_sizes, jsfs, params, momi2_model, NUM_REPLICATES, loglik_with_grad=True)
    elif method == 'dadi':
        dadi.cuda_enabled(True)
        ret = get_dadi_esfs(sampled_demes, sample_sizes, jsfs, params, NUM_REPLICATES, loglik_with_grad=False)

    else:
        raise ValueError(f"Unknown {method=}")

    ret = ret | opts
    ret = Return(**ret)

    with open(save_path, 'wb') as f:
        pickle.dump(ret, f)
