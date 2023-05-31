import tskit

import numpy as np
import sparse

from ast import literal_eval
from tqdm import trange

loc = lambda i: f"/mnt/turbo/eneswork/Unified_genome/hgdp_tgp_sgdp_high_cov_ancients_chr{i}_p.dated.trees"
NAME = "UNIF"

deme_ids = {'Yoruba': 64, 'French': 16, 'Papuan': 175, 'Vindija': 214, 'Denisovan': 213, 'CEU': 54}
sampled_demes = ('Yoruba', 'French', 'Papuan', 'Vindija', 'Denisovan', 'CEU')
sample_sizes = {'Yoruba': 300, 'French': 300, 'Papuan': 300, 'Vindija': 300, 'Denisovan': 300, 'CEU': 300}
SEED = 108

p = len(sampled_demes)

np.random.seed(SEED)

for i in trange(23):#3):
	try:
		ts = tskit.load(loc(i))
		tspops = ts.populations()

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
	except:
		pass

size = np.prod(X.shape)
x = '_'.join(sampled_demes)
X = sparse.COO(X)
sparse.save_npz(f'jsfs_{NAME}_{x}_{size}_{SEED}', X)
