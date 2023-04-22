import sys
sys.path.append("../../../pyslurm")
import os

from momi3.MOMI import Momi
from momi3.Params import Params
from momi3.utils import bootstrap_sample
from momi3.optimizers import optax_for_momi

import pickle
import demes
import optax
import tskit

import pandas as pd
import numpy as np
import jax.numpy as jnp

from scipy import optimize
from time import sleep

model_name = 'arc5'


def get_lr_vector(theta_train_dict, lr, transformed=True):
    train_keys = tuple(theta_train_dict)
    lr_vec = []
    for i in train_keys:
        if i.find('eta') != -1:
            if transformed:
                lr_vec.append(lr * 10000)
            else:
                lr_vec.append(lr * 10**5)
        elif i.find('pi') != -1:
                lr_vec.append(lr)
        elif i.find('tau') != -1:
            if transformed:
                lr_vec.append(lr * 100)
            else:
                lr_vec.append(lr * 10**4)
    
    return jnp.array(lr_vec)


def get_srun(proc):
    from pyslurm import Slurm
    slurm = Slurm(
        user='enes',
        path='/scratch/stats_dept_root/stats_dept/enes/',
        account='jonth0'
    )

    time = "0-12:30:00"
    # srun = slurm.batch(
    #     f'#time={time}'
    # )

    srun_GPU = slurm.batch(
        f'#time={time}',
        "#mem-per-cpu=None",
        "#nodes=1",
        "#cpus-per-task=1",
        "#mem=16G",
        "#gpus-per-node=1",
        "#partition=spgpu",
        '#job-name="GPU_bs_acc"',
        "module load cudnn",
        "module load cuda"
    )

    mem = 1000
    time = "0-10:00:00"
    srun_CPU = slurm.batch(
        f'#mem-per-cpu={mem}',
        '#cpus-per-task=5',
        f'#time={time}',
        '#job-name="CPU_bs_acc"'
    )

    if proc == 'CPU':
        srun = srun_CPU
    else:
        srun = srun_GPU

    return srun


def get_demo():
    demo = demes.load('arc5_pulse_inferred.yaml')
    ddict = demo.asdict()
    dem = ddict['demes']
    pul = ddict['pulses']
    dem[7]['ancestors'] = ['ancestral']
    dem[7]['start_time'] = 12500
    dem = [dem[0], dem[1], dem[2], dem[3], dem[4], dem[7], dem[8]]
    pul = [pul[0]]
    ddict['demes'] = dem
    ddict['pulses'] = pul
    demo = demes.Builder.fromdict(ddict).resolve()
    return demo


def get_params(momi):
    params = Params(momi)

    params.set_train('tau_5', True)
    params.set_train('pi_0', True)
    params.set_train('eta_1', True)

    return params


def send_job(arg_d, mode='run'):
    srun = get_srun(arg_d['proc'])
    arg_d['mode'] = mode
    x = ''
    for key, val in arg_d.items():
        x += f' {key}={val}'
    test = f'python bs_example_demo.py' + x
    jobid = srun.run(test)
    print(f"{jobid}: Sent -- {test}")
    sleep(0.05)


if __name__ == "__main__":
    # For sending GL_jobs: python bootstrap.py out=/tmp/ mode=send_jobs njobs=50 proc=GPU or (CPU)
    # For sending bootstrap iter: python out=/tmp/ mode=run
    args = sys.argv[1:]

    arg_d = {
        'mode': 'run',  # run, runmore, send_job_bootstrap, send_job_lr
        'seed': 'None',  # seed for bootstrap
        'niter': '300',
        'lr': '0.0001',
        'bound_sampler': 'False',
        'proc': 'CPU',
        'njobs': 'None',
        'q': '0.999'
    }

    for arg in args:
        k, v = arg.split('=')
        if k in arg_d:
            arg_d[k] = v
        else:
            raise KeyError(f"Unknown key={k}")

    mode = arg_d['mode']

    if mode == 'send_jobs':
        for i in range(int(arg_d['njobs'])):
            arg_d['seed'] = f'{i}'
            send_job(arg_d)

    else:
        seed = int(arg_d['seed'])
        niter = int(arg_d['niter'])
        lr = float(arg_d['lr'])
        bound_sampler = arg_d['bound_sampler'] == 'True'
        q = float(arg_d['q'])

        ts = tskit.load('../../../Unified_genome/hgdp_tgp_sgdp_high_cov_ancients_chr20_p.dated.trees')
        deme_ids = {'Yoruba': 64, 'French': 16, 'Vindija': 214}
        sampled_demes = tuple(deme_ids)

        samples = [ts.samples(deme_ids[pop]) for pop in sampled_demes]
        jsfs = ts.allele_frequency_spectrum(samples, polarised=True, span_normalise=False)
        sample_sizes = [i - 1 for i in jsfs.shape]

        demo = get_demo()
        momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
        params = get_params(momi)

        if bound_sampler:
            bounds = momi.bound_sampler(params, 10000, min_lineages=2, seed=108, quantile=q)
            momi = momi.bound(bounds)

        theta_train_dict = params.theta_train_dict()
        lr_vec = get_lr_vector(theta_train_dict, lr, transformed=False)

        x = momi._bootstrap_sample(jsfs, seed=seed)
        optimizer = optax.adabelief(learning_rate=lr_vec)
        theta_train_dict, opt_state, history = optax_for_momi(optimizer, momi, params, x, niter, transformed=False)
        ind = np.argmin(history['LLs'])

        if bound_sampler:
            name = f"seed_{seed}_bound_{bound_sampler}_q={q}.csv"
        else:
            name = f"seed_{seed}_bound_{bound_sampler}.csv"

        with open('bootstrap_acc/' + name, 'wb') as f:
            pickle.dump(history, f)
