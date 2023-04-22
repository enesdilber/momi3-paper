import sys
sys.path.append("/nfs/turbo/lsa-jonth/eneswork/pyslurm")
import os

from momi3.MOMI import Momi
from momi3.Params import Params
from momi3.utils import bootstrap_sample
from momi3.optimizers import optax_for_momi

import pickle
import demes
import optax

import numpy as np
import jax.numpy as jnp

from scipy import optimize
from time import sleep

model_name = 'arc5'


def get_lr_vector(theta_train_dict, arg_d):
    train_keys = tuple(theta_train_dict)
    lr_vec = []
    for i in train_keys:
        if i.find('eta') != -1:
            lr_vec.append(float(arg_d['lr_eta']))
        elif i.find('pi') != -1:
            lr_vec.append(float(arg_d['lr_pi']))
        elif i.find('rho') != -1:
            lr_vec.append(float(arg_d['lr_rho']))
        elif i.find('tau') != -1:
            lr_vec.append(float(arg_d['lr_tau']))
        else:
            raise ValueError('')

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
        '#job-name="GPU_arc"',
        "module load cudnn",
        "module load cuda"
    )

    mem = 3000
    time = "0-10:00:00"
    srun_CPU = slurm.batch(
        f'#mem-per-cpu={mem}',
        '#cpus-per-task=20',
        f'#time={time}',
        '#job-name="CPU_arc"'
    )

    if proc == 'CPU':
        srun = srun_CPU
    else:
        srun = srun_GPU

    return srun


def get_demo(model):
    demo = demes.load('arc5_pop_size_inferred.yaml')
    if model == 'vanilla':
        pass
    elif model == 'pulse':
        ddict = demo.asdict()
        dbuilder = demes.Builder.fromdict(ddict)
        dbuilder.add_pulse(sources=['NeanderthalGHOST'], dest='OOA', proportions=[0.05], time=2500)
        dbuilder.add_pulse(sources=['DenisovanGHOST'], dest='Papuan', proportions=[0.025], time=1000)
        demo = dbuilder.resolve()
    else:
        raise ValueError(f'Unknown {model=}')
    return demo


def get_params(momi, train_pars):
    params = Params(momi)

    if train_pars.find('eta') != -1:
        params.set_train_all_etas(True)
        params.set_train('eta_0', False)

    if train_pars.find('rho') != -1:
        params.set_train_all_rhos(True)

    if train_pars.find('tau') != -1:
        params.set_train_all_taus(True)

    if train_pars.find('pi') != -1:
        params.set_train_all_pis(True)

    return params


def send_job(arg_d, mode='run'):
    srun = get_srun(arg_d['proc'])
    arg_d['mode'] = mode
    x = ''
    for key, val in arg_d.items():
        x += f' {key}={val}'
    test = f'python train.py' + x
    jobid = srun.run(test)
    print(f"{jobid}: Sent -- {test}")
    sleep(0.05)


if __name__ == "__main__":
    # For sending GL_jobs: python bootstrap.py out=/tmp/ mode=send_jobs njobs=50 proc=GPU or (CPU)
    # For sending bootstrap iter: python out=/tmp/ mode=run
    args = sys.argv[1:]

    arg_d = {
        'mode': 'run',  # run, runmore, send_job_bootstrap, send_job_lr
        'model': 'vanilla',  # vanilla, pulse
        'data': 'jsfs_UNIF_Yoruba_French_Papuan_Vindija_Denisovan_11979_108.npy',
        'seed': 'None',  # seed for bootstrap
        'nSNPs': 'all',
        'niter': '500',
        'lr_pi': '0.05',
        'lr_eta': '5.0',
        'lr_rho': '0.1',
        'lr_tau': '0.1',
        'bound_sampler': 'False',
        'train_pars': 'eta',  # 'eta,pi,rho,tau'
        'transformed': 'True',
        'proc': 'CPU',
        'njobs': 'None',
        'file_name': 'None',
        'q': '0.99'
    }

    for arg in args:
        k, v = arg.split('=')
        if k in arg_d:
            arg_d[k] = v
        else:
            raise KeyError(f"Unknown key={k}")

    mode = arg_d['mode']
    data_path = arg_d['data']
    out_dir = data_path.split('.')[0]

    if os.path.exists(out_dir):
        pass
    else:
        os.mkdir(out_dir)

    if mode == 'send_job_bootstrap':
        for i in range(int(arg_d['njobs'])):
            arg_d['seed'] = f'{i}'
            send_job(arg_d)

    elif mode == 'send_job_lr':
        lrs = [0.001, 0.01, 0.05, 0.1, 0.5, 1., 10., 100]
        for lr in lrs:
            arg_d['lr'] = f'{lr}'
            send_job(arg_d)

    elif mode[:8] == 'send_job':
        if mode[-4:] == 'more':
            mode = 'runmore'
        else:
            mode = 'run'
        send_job(arg_d, mode)

    else:
        print(arg_d)
        # prms
        model = arg_d['model']
        seed = arg_d['seed']
        nSNPs = arg_d['nSNPs']
        niter = int(arg_d['niter'])
        bound_sampler = arg_d['bound_sampler'] == 'True'
        train_pars = arg_d['train_pars']
        transformed = arg_d['transformed'] == 'True'
        file_name = arg_d['file_name']
        q = float(arg_d['q'])

        if file_name == 'None':
            file_name = []
            for key, val in arg_d.items():
                if key in ['data', 'mode', 'file_name']:
                    pass
                else:
                    file_name.append(key + '=' + val)
            file_name = '__'.join(file_name) + '.pickle'
        else:
            pass
        out_path = os.path.join(out_dir, file_name)

        # DATA
        jsfs = np.load(data_path)
        if seed == 'None':
            pass
        else:
            seed = int(seed)
            if nSNPs == 'all':
                nSNPs = None
            else:
                nSNPs = int(nSNPs)
            jsfs = bootstrap_sample(jsfs, n_SNPs=nSNPs, seed=seed)

        # Model
        demo = get_demo(model)
        sampled_demes = demo.metadata['sampled_demes']
        sample_sizes = [i - 1 for i in jsfs.shape]
        momi = Momi(demo, sampled_demes, sample_sizes, jitted=True, batch_size=152000, low_memory=True)
        params = get_params(momi, train_pars)
        if bound_sampler:
            bounds = momi.bound_sampler(params, 10000, min_lineages=2, seed=108, quantile=0.999)
            momi = momi.bound(bounds)

        print(momi.loglik(params, jsfs))
        print(jsfs.std())

        theta_train_dict = params.theta_train_dict(transformed)
        lr_vec = get_lr_vector(theta_train_dict, arg_d)
        optimizer = optax.adabelief(learning_rate=lr_vec)
        if mode == 'runmore':
            with open(out_path, 'rb') as f:
                ret = pickle.load(f)
            theta_train_dict = ret['ttd']
            opt_state = ret['opt_state']
            history = ret['history']
        elif mode == 'run':
            theta_train = jnp.array(list(theta_train_dict.values()))
            opt_state = optimizer.init(theta_train)
            history = dict(LLs=[], ttds=[])
        else:
            raise ValueError(f'Unknown {mode=}')

        print(jsfs.shape)
        print(jsfs.sum())
        theta_train_dict, opt_state, history = optax_for_momi(
            optimizer,
            momi,
            params,
            jsfs,
            niter=niter,
            transformed=transformed,
            theta_train_dict=theta_train_dict,
            opt_state=opt_state,
            history=history
        )
        ret = {'ttd': theta_train_dict, 'opt_state': opt_state, 'history': history}

        with open(out_path, 'wb') as f:
            pickle.dump(ret, f)

        print(f'saved: {out_path}')
