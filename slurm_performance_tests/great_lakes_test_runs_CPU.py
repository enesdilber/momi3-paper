import sys
sys.path.append("/nfs/turbo/lsa-jonth/eneswork/pyslurm")
import numpy as np
from pyslurm import Slurm
from time import sleep

slurm = Slurm(
    user='enes',
    path='/scratch/stats_dept_root/stats_dept/enes/',
    account='jonth0'
)

mem = '100G'
time = "0-10:00:00"
srun = slurm.batch(
    f'#mem-per-cpu={mem}',
    '#cpus-per-task=1',
    f'#time={time}',
    '#job-name="CPU_tests"'
)

save_folder = '/nfs/turbo/lsa-jonth/eneswork/jthlab/momi3-paper/slurm_performance_tests/timing_tests_out/'

SEQ_LENs = [1e8, 1e7, 1e6]

# CPU TESTS


def pop5admix():
    tests = []
    sample_sizes = [25, 15, 10, 5, 3]
    methods = ["momi2"]

    for sample_size in sample_sizes:
        for seqlen in SEQ_LENs:
            for method in methods:
                if (method == "moments") & (sample_size > 5):
                    pass
                else:
                    tests.append(f'python time_5_pop_admixture.py {method} {sample_size} {seqlen} {save_folder}')
    return tests


def pop2bound():
    save_folder = '/nfs/turbo/lsa-jonth/eneswork/jthlab/momi3-paper/slurm_performance_tests/timing_bound_sampler/'
    tests = []
    sample_sizes = [100]
    seqlen = SEQ_LENs[-1]

    for sample_size in sample_sizes:
        tests.append(f'python time_2_pop_bound_sampler_acc.py {sample_size} {seqlen} {save_folder}')
    return tests


def pop8admix():
    tests = []
    extra_ns = [94, 19, 14, 9, 4, 0]
    methods = ["momi2", "momi3", "momi3_bounds"]

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
    methods = ["moments", "dadi"]
    for sample_size in sample_sizes:
        for seqlen in SEQ_LENs:
            for ndemes in Ndemes:
                for method in methods:
                    if (method == "dadi") & (ndemes > 3):
                        pass
                    else:
                        tests.append(f'python time_mig_OOA.py {method} {ndemes} {sample_size} {seqlen} {save_folder}')
    return tests


def IWM_n_large():
    tests = []
    Ndemes = [3, 2]
    sample_sizes = [20, 15, 10]
    methods = ["moments", "dadi"]
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
    methods = ["momi2"]
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
    methods = ["momi2"]
    for sample_size in sample_sizes:
        for seqlen in SEQ_LENs:
            for ndemes in Ndemes:
                for method in methods:
                    if (method == "moments") & (sample_size > 5) & (ndemes > 3):
                        pass
                    else:
                        tests.append(f'python time_no_mig_no_adm.py {method} {ndemes} {sample_size} {seqlen} {save_folder}')
    return tests


tests = ['python time_8_pop_admixture.py momi3 19 100000000.0 /nfs/turbo/lsa-jonth/eneswork/jthlab/momi3-paper/slurm_performance_tests/timing_tests_out/']  # GK()
tests = pop5admix() + constant() + constant_n_large()
jobids = []
for test in tests:
    jobids.append(srun.run(test))
    print(f"{jobids[-1]}: Sent -- {test}")
    sleep(0.05)

jobids = set(jobids)

while True:
    sleep(20)
    finished_jobs = jobids.difference(set(slurm.queue()["JOBID"]))
    jobids = jobids.difference(finished_jobs)
    for jobid in finished_jobs:
        print(f"{jobid}: {slurm.my_job_stats(jobid)['State']}")
    if len(jobids) == 0:
        break
