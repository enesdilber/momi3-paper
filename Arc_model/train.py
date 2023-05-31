import sys
sys.path.append("/nfs/turbo/lsa-jonth/eneswork/pyslurm")
import os

from momi3.MOMI import Momi
from momi3.Params import Params
from momi3.utils import bootstrap_sample, tqdm
from momi3.optimizers import optax_for_momi

import pickle
import demes
import optax
import tskit
import moments

import numpy as np
import jax.numpy as jnp
import numdifftools as nd

from scipy import optimize
from time import sleep
from time import time

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
        "#mem=48G",
        "#gpus-per-node=1",
        "#partition=spgpu",
        '#job-name="GPU_arc"',
        "module load cudnn",
        "module load cuda"
    )

    mem = 1000
    time = "0-10:00:00"
    srun_CPU = slurm.batch(
        f'#mem-per-cpu={mem}',
        '#cpus-per-task=16',
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
    elif model == 'inferred':
        demo = demes.load('arc5_pulse_inferred.yaml')
    elif model == 'inferred_plus_migration':
        demo = demes.load('arc5_pulse_inferred.yaml')
        dd = demo.asdict()
        b = demes.Builder.fromdict(dd)
        b.add_migration(source='AMH', dest='NeanderthalGHOST', rate=0.001)
        b.add_migration(source='NeanderthalGHOST', dest='AMH', rate=0.01)
        demo = b.resolve()
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


def get_sfs(SEED, sampled_demes, sample_sizes, TS):
    deme_ids = {'Yoruba': 64, 'French': 16, 'Papuan': 175, 'Vindija': 214, 'Denisovan': 213}
    np.random.seed(SEED)
    for ts in TS:
        samples = []
        for pop in sampled_demes:
            deme_id = deme_ids[pop]
            samp = ts.samples(deme_id)
            n = sample_sizes[pop]
            n_deme = min(len(samp), n)
            samples.append(np.random.choice(samp, size=n_deme, replace=False))
        AFS = ts.allele_frequency_spectrum(samples, polarised=True, span_normalise=False)

        try:
            X += AFS
        except:
            X = AFS
    
    return X


def f_migration_grid(hess=False):
    demo = get_demo('inferred_plus_migration')
    jsfs = np.load('jsfs_UNIF_Yoruba_French_Papuan_Vindija_Denisovan_3419145_108.npy')
    sampled_demes = ('Yoruba', 'French', 'Papuan', 'Vindija', 'Denisovan')
    sample_sizes = [i - 1 for i in jsfs.shape]

    momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
    params = Params(momi)
    bounds = momi.bound_sampler(params, 10000, min_lineages=4, quantile=0.99)
    momi = momi.bound(bounds)

    bs_jsfs = momi._bootstrap_sample(jsfs, int(1e6), seed=108)

    if hess:
        params.set_train('rho_0', True)
        params.set_train('rho_1', True)
        for i in tqdm([1, 2]):
            print(momi.loglik_with_gradient(params, bs_jsfs))

        def f(rho):
            print(rho)
            params.set('rho_0', rho[0])
            params.set('rho_1', rho[1])
            return momi.loglik(params, bs_jsfs)

        rho = [1.2508442281550083e-05, 2.2409018563948744e-05]
        H = nd.Hessian(f, step=1e-6)(rho)
        print(H)
    else:
        mle_rho0 = 1.2508442281550083e-05
        mle_rho1 = 2.2409018563948744e-05
        s_rho0 = 1.06904497e-06
        s_rho1 = 3.71390676e-07

        Z = 7
        rho0s = np.linspace(mle_rho0 - Z * s_rho0, mle_rho0 + Z * s_rho0, 11)
        rho1s = np.linspace(mle_rho1 - Z * s_rho1, mle_rho1 + Z * s_rho1, 11)

        Xs = []
        for rho0 in tqdm(rho0s):
            xs = []
            for rho1 in tqdm(rho1s):
                params.set('rho_0', float(rho0))
                params.set('rho_1', float(rho1))
                x = momi.loglik(params, bs_jsfs)
                print(x)
                xs.append(float(x))
            Xs.append(xs)

        print(Xs)


def f_small_sample_bootstrap():
    loc = lambda i: f"../../../Unified_genome/hgdp_tgp_sgdp_high_cov_ancients_chr{i}_p.dated.trees"
    TS = []
    for i in range(1, 22):
        try:
            ts = tskit.load(loc(i))
            TS.append(ts)
        except:
            pass

    sampled_demes = ('Yoruba', 'French', 'Papuan', 'Vindija', 'Denisovan')
    sample_sizes = {'Yoruba': 6, 'French': 6, 'Papuan': 6, 'Vindija': 2, 'Denisovan': 2}

    demo = demes.load('arc5_pulse_inferred.yaml')
    ss = [sample_sizes[pop] for pop in sampled_demes]

    momi = Momi(demo, sampled_demes, ss, jitted=True, batch_size=3090)
    params = momi._default_params
    params.set_train_all_etas(True)
    params.set_train_all_pis(True)
    params.set_train('eta_0', False) # Ancestral pop size is not inferred
    
    bs_iter = 100
    niter = 800

    histories = []

    for i in range(bs_iter):
        jsfs = get_sfs(i, sampled_demes, sample_sizes, TS)

        transformed = False
        theta_train_dict = params.theta_train_dict(transformed)
        lr_vec = get_lr_vector(theta_train_dict, {'lr_eta': 1., 'lr_pi': 0.001})
        optimizer = optax.adabelief(learning_rate=lr_vec, b1=0.3, b2=0.1)

        theta_train_dict, opt_state, history = optax_for_momi(
            optimizer,
            momi,
            params,
            jsfs,
            niter=niter,
            transformed=transformed,
            theta_train_dict=theta_train_dict,
        )
        
        histories.append(history)

        with open('bs_small_sample2.pickle', 'wb') as f:
            pickle.dump(histories, f)


def moments5(jsfs):
    # Model
    demo = get_demo('pulse')
    sampled_demes = demo.metadata['sampled_demes']
    sample_sizes = [i - 1 for i in jsfs.shape]
    esfs = moments.Spectrum.from_demes(
        demo, sampled_demes=sampled_demes, sample_sizes=sample_sizes
    )
    print(esfs)


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
        'q': '0.99',
        'batch_size': 'None',
        'hess': 'False'
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
        file_name = arg_d['file_name']
        for i in range(int(arg_d['njobs'])):
            arg_d['seed'] = f'{i}'
            name, _ = file_name.split('.pickle')
            arg_d['file_name'] = f"{name}{i}.pickle"
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

    elif mode == 'small_sample':
        f_small_sample_bootstrap()
    elif mode == 'small_sample_job':
        srun = get_srun('GPU')
        test = f'python train.py mode=small_sample'
        jobid = srun.run(test)
        print(f"{jobid}: Sent -- {test}")

    elif mode == 'moments5':
        jsfs = np.load(data_path)
        moments5(jsfs)
    elif mode == 'moments5_job':
        srun = get_srun('CPU')
        test = f'python train.py mode=moments5'
        jobid = srun.run(test)
        print(f"{jobid}: Sent -- {test}")

    elif mode == 'migration_grid':
        f_migration_grid(arg_d['hess'] == 'True')
    elif mode == 'migration_grid_job':
        srun = get_srun('GPU')
        hess = arg_d['hess']
        test = f'python train.py mode=migration_grid hess={hess}'
        jobid = srun.run(test)
        print(f"{jobid}: Sent -- {test}")
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
        if arg_d['batch_size'] == 'None':
            momi = Momi(demo, sampled_demes, sample_sizes, jitted=True)
        else:
            batch_size = int(arg_d['batch_size'])
            momi = Momi(demo, sampled_demes, sample_sizes, jitted=True, batch_size=batch_size, low_memory=True)

        params = get_params(momi, train_pars)
        if bound_sampler:
            bounds = momi.bound_sampler(params, 10000, min_lineages=2, seed=108, quantile=0.999)
            momi = momi.bound(bounds)

        print(momi.loglik(params, jsfs))
        print(jsfs.std())

        theta_train_dict = params.theta_train_dict(transformed)
        lr_vec = get_lr_vector(theta_train_dict, arg_d)
        optimizer = optax.adabelief(learning_rate=lr_vec, b1=0.3, b2=0.1)
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

        run_left = niter
        every_x_iter = 100

        while(run_left > 0):
            theta_train_dict, opt_state, history = optax_for_momi(
                optimizer,
                momi,
                params,
                jsfs,
                niter=every_x_iter,
                transformed=transformed,
                theta_train_dict=theta_train_dict,
                opt_state=opt_state,
                history=history
            )
            ret = {'ttd': theta_train_dict, 'opt_state': opt_state, 'history': history}

            with open(out_path, 'wb') as f:
                pickle.dump(ret, f)

            run_left = run_left - every_x_iter

            print(theta_train_dict)

        print(f'saved: {out_path}')
