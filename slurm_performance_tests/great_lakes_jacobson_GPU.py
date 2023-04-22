import sys
sys.path.append("/nfs/turbo/lsa-jonth/eneswork/pyslurm")
import numpy as np
from pyslurm import Slurm
from math import ceil

import sys
import os

import jax

from momi3.utils import get_CPU_name, get_GPU_name, msprime_simulator
from momi3.MOMI import Momi
from momi3.Params import Params
from momi3.Data import get_X_batches, get_sfs_batches
from momi3.Experimental import Experimental
import sparse

import numpy as np

from time import time

#from timeit_methods import get_momi_times, Return

import pickle
import demes
import hashlib

slurm = Slurm(
    user='enes',
    path='/scratch/stats_dept_root/stats_dept/enes/',
    account='jonth0'
)

TIME = "0-0:60:00"
# srun = slurm.batch(
#     f'#time={time}'
# )

srun = slurm.batch(
    f'#time={TIME}',
    "#mem-per-cpu=None",
    "#nodes=1",
    "#cpus-per-task=1",
    "#mem=45G",
    "#gpus-per-node=1",
    "#partition=spgpu",
    '#job-name="GPU_tests"',
    "module load cudnn",
    "module load cuda"
)

# For esfs use batch_size=250 with 45G

EPS = 0.1
NUM_REPLICATES = 5
RANDOM_SEED = 108
BATCH_SIZE = 100000
CPU = get_CPU_name()
GPU = get_GPU_name()


def get_model(n, min_mig=1e2, jitted=True):
    demo = demes.load("../jacobson_data/jacobson.yml")
    sampled_demes = demo.metadata["sampled_demes"]
    sample_sizes = 4 * [n] + 5 * [1]
    demo_asdict = demo.asdict()
    for mig in demo_asdict['migrations']:
        mig['end_time'] = np.clip(mig['end_time'], min_mig, np.inf)
    return Momi(demo, sampled_demes, sample_sizes, jitted=jitted)


def sim_data(momi, sequence_length=int(1e8), recombination_rate=1e-8, mutation_rate=1e-8, seed=108):

    out_name = f"jsfs_{sequence_length}_{recombination_rate}_{mutation_rate}_{seed}_{'-'.join([str(i) for i in momi.sample_sizes])}.npz"
    out_path = os.path.join('../jacobson_data', out_name)

    if os.path.exists(out_path):
        jsfs = sparse.load_npz(out_path)
    else:
        jsfs = momi.simulate_chromosome(
            sequence_length=sequence_length,
            seed=seed,
            recombination_rate=recombination_rate,
            mutation_rate=mutation_rate
        )
        sparse.save_npz(out_path, jsfs)

    return jsfs, out_name


if __name__ == "__main__":

    args = sys.argv[1:]

    send_jobs = False
    for arg in args:
        if arg == '--send_great_lakes_jobs':
            # send great lakes jobs
            send_jobs = True
        else:
            key, value = arg.split('=')
            if key == 'n':
                n = int(value)
            if key == 'batch':
                batch = int(value)
            if key == 'batch_size':
                batch_size = int(value)
            if key == 'sequence_length':
                sequence_length = int(float(value))

    n = 4
    momi = get_model(n, jitted=False)
    jsfs, out_name = sim_data(momi, sequence_length=sequence_length)
    params = momi._default_params

    entries = jsfs.coords
    n_entries = entries.shape[1]

    if send_jobs:
        for batch in range(ceil(n_entries / batch_size)):
            job = f'python great_lakes_jacobson_GPU.py n={n} batch={batch} batch_size={batch_size} sequence_length={sequence_length}'
            jobid = srun.run(job)
            print(f"{jobid}: Sent -- {job}")
    else:
        entries = jsfs.coords
        n_entries = entries.shape[1]

        deriveds = {pop: entries[i] for i, pop in enumerate(momi.sampled_demes)}
        X = get_X_batches(
            momi.sampled_demes,
            momi.sample_sizes,
            tuple(set(momi._T._leaves)),
            deriveds=tuple(tuple(deriveds[pop]) for pop in momi.sampled_demes),
            batch_size=batch_size,
            add_etbl_vecs=True
        )

        sfs = get_sfs_batches(jsfs.data, batch_size=batch_size)[0]

        X_batch = {pop: X[pop][0][batch][None, :, :] for pop in X}
        X_size = X_batch[tuple(X_batch)[0]].shape[1] - 3  # -3 for etbl

        st, en = batch * X_size, (batch + 1) * X_size
        sfs_batch = sfs[st:en]
        sfs_batch = np.pad(sfs_batch, [0, X_size - len(sfs_batch)])

        auxd = momi._auxd
        demo = params.demo_graph
        _f = momi._JAX_functions._f
        esfs_tensor_prod = momi._JAX_functions.esfs_tensor_prod
        esfs_map = momi._JAX_functions.esfs_map

        def fun(theta_train_dict, theta_nuisance_dict, X_batch, auxd, demo, _f, esfs_tensor_prod):
            theta_dict = theta_train_dict | theta_nuisance_dict
            return esfs_map(theta_dict, X_batch, auxd, demo, _f, esfs_tensor_prod).sum()

        def f():
            return momi._JAX_functions.loglik_batch(
                params._theta_train_dict,
                params._theta_nuisance_dict,
                X_batch,
                sfs_batch,
                auxd,
                demo,
                _f,
                esfs_tensor_prod,
                esfs_map,
            )
        # t1 = time()
        x = f()
        # t2 = time()
        # x = f()
        # t3 = time()
        # x = f()
        # t4 = time()

        # print(t2 - t1)
        # print(t3 - t2)
        # print(t4 - t3)

        print(x)

        # momiX = Experimental(momi, params)
        # x = momiX.numerical_FIM_uncert(X_batch, sfs_batch)
        # print(x)

        # val = jax.grad(fun)(params._theta_train_dict, params._theta_nuisance_dict, X_batch, auxd, demo, _f, esfs_tensor_prod)
        out_path = f'../jacobson_data/loglik_{batch}_{out_name}'[:-4]
        np.save(out_path, x)
