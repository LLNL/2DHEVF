# init.py
import signac
import numpy as np
import itertools

project = signac.init_project('volume-fraction-sweep', workspace='/p/lscratchh/miguel/heat_exchanger/VF_2D/')


mu_list = np.array([0.08, 0.04, 0.02, 0.01])
enthalpy_scale = np.array([0.04])
alphabar_arr = np.array([3e-4])

import itertools
for mu, es, alphabar, filter in list(itertools.product(mu_list, enthalpy_scale, alphabar_arr, filter_arr)):
    sp = { 'enthalpy_scale' : es,
            'mu' : mu,
            'alphabar' : alphabar,
            'filter' : filter
            }
    job = project.open_job(sp)
    job.init()
