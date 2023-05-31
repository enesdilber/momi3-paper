import sys
sys.path.append("/nfs/turbo/lsa-jonth/eneswork/pyslurm")
import numpy as np
from pyslurm import Slurm
from time import sleep


# gpu (V100 double precision): max gpus = 3
# spgpu (A40 single precision): max gpus = 8

slurm = Slurm(
    user='enes',
    path='/scratch/stats_dept_root/stats_dept/enes/',
    account='jonth0'
)

time = "0-2:30:00"
# srun = slurm.batch(
#     f'#time={time}'
# )

srun = slurm.batch(
    f'#time={time}',
    "#mem-per-cpu=None",
    "#nodes=1",
    "#cpus-per-task=1",
    "#mem=16G",
    "#gpus-per-node=1",
    "#partition=gpu",
    '#job-name="GPU_tests"',
    "module load cudnn",
    "module load cuda"
)

save_folder = '/nfs/turbo/lsa-jonth/eneswork/jthlab/momi3-paper/slurm_performance_tests/timing_tests_out/'

SEQ_LENs = [1e8, 1e7, 1e6]

# GPU TESTS


def pop5admix():
    tests = []
    sample_sizes = [25, 15, 10, 5, 3]
    methods = ["momi3"]

    for sample_size in sample_sizes:
        for seqlen in SEQ_LENs:
            for method in methods:
                tests.append(f'python time_5_pop_admixture.py {method} {sample_size} {seqlen} {save_folder}')
    return tests


def pop8admix():
    tests = []
    extra_ns = [94]
    methods = ["momi3_bounds"]

    for extra_n in extra_ns:
        for seqlen in SEQ_LENs:
            for method in methods:
                if (extra_n > 20) & (method != "momi3_bounds"):
                    pass
                else:
                    if method == "momi3_bounds":
                        method = "momi3"
                        bound_sampler = "bound_sampler"
                    else:
                        bound_sampler = ""
                    tests.append(f'python time_8_pop_admixture.py {method} {extra_n} {seqlen} {save_folder} {bound_sampler}')
    return tests


def IWM():
    tests = []
    Ndemes = [5, 4, 3, 2]
    sample_sizes = [5]
    methods = ['dadi']  # ["momi3"]
    for sample_size in sample_sizes:
        for seqlen in SEQ_LENs:
            for ndemes in Ndemes:
                for method in methods:
                    tests.append(f'python time_mig_OOA.py {method} {ndemes} {sample_size} {seqlen} {save_folder}')
    return tests


def IWM_n_large():
    tests = []
    Ndemes = [3, 2]
    sample_sizes = [20, 15, 10]
    methods = ["momi3"]
    for sample_size in sample_sizes:
        for seqlen in SEQ_LENs:
            for ndemes in Ndemes:
                for method in methods:
                    tests.append(f'python time_mig_OOA.py {method} {ndemes} {sample_size} {seqlen} {save_folder}')
    return tests


def constant_n_large():
    tests = []
    Ndemes = [3, 2]
    sample_sizes = [200, 100]
    methods = ["momi3"]
    for sample_size in sample_sizes:
        for seqlen in SEQ_LENs:
            for ndemes in Ndemes:
                for method in methods:
                    tests.append(f'python time_no_mig_no_adm.py {method} {ndemes} {sample_size} {seqlen} {save_folder}')
    return tests


def constant():
    tests = []
    Ndemes = [5, 4, 3, 2]
    sample_sizes = [50, 25, 10, 5]
    methods = ["momi3"]
    for sample_size in sample_sizes:
        for seqlen in SEQ_LENs:
            for ndemes in Ndemes:
                for method in methods:
                    tests.append(f'python time_no_mig_no_adm.py {method} {ndemes} {sample_size} {seqlen} {save_folder}')
    return tests


# tests = [f'python time_jacobson.py momi3 5 200 {save_folder}']


# tests = ['python time_mig_OOA.py momi3 3 20 100000000.0 /nfs/turbo/lsa-jonth/eneswork/jthlab/momi3-paper/slurm_performance_tests/timing_tests_out/']#constant() + constant_n_large() + IWM()
tests = IWM()  # + pop5admix() + IWM() + IWM_n_large() + constant() + constant_n_large()

# tests = [
#     #'python time_5_pop_admixture.py momi3 25 100000000.0 /nfs/turbo/lsa-jonth/eneswork/jthlab/momi3-paper/slurm_performance_tests/timing_tests_out/',
#     'python time_mig_OOA.py momi3 3 20 100000000.0 /nfs/turbo/lsa-jonth/eneswork/jthlab/momi3-paper/slurm_performance_tests/timing_tests_out/'
# ]

jobids = []
for test in tests:
    jobids.append(srun.run(test))
    print(f"{jobids[-1]}: Sent -- {test}")
    sleep(0.05)

jobids = set(jobids)

while True:
    finished_jobs = jobids.difference(set(slurm.queue()["JOBID"]))
    jobids = jobids.difference(finished_jobs)
    for jobid in finished_jobs:
        print(f"{jobid}: {slurm.my_job_stats(jobid)['State']}")
    if len(jobids) == 0:
        break
    sleep(20)
