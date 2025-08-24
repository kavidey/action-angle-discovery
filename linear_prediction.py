# %%
import pickle
import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from multiprocessing import Pool
from scipy.integrate import solve_ivp

import rebound as rb
import celmech as cm
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
from celmech.poisson_series import PoissonSeriesHamiltonian 
from celmech.disturbing_function import list_secular_terms
from celmech.poisson_series import PoissonSeries
from celmech.rk_integrator import RKIntegrator

from hadden_theory.test_particle_secular_hamiltonian import SyntheticSecularTheory, TestParticleSecularHamiltonian
from hadden_theory import test_particle_secular_hamiltonian
# hack to make pickle load work
import sys
sys.modules['test_particle_secular_hamiltonian'] = test_particle_secular_hamiltonian
# %%
integration_path = Path("integrations") / "ecc_inc_no_integration"
integration_path.mkdir(parents=True, exist_ok=True)
# %%
try:
	with open("hadden_theory/solar_system_synthetic_solution.bin","rb") as fi:
		solar_system_synthetic_theory=pickle.load(fi)
except Exception as e:
	print("Cannot find solar_system_synthetic_solution.bin. Please run OuterSolarSystemSyntheticSecularSolution.ipynb first")
	raise Exception()
# %%
outer_only = True
print(f"Outer Planets Only: {outer_only}")

truncate_dictionary = lambda d,tol: {key:val for key,val in d.items() if np.abs(val)>tol}
simpler_secular_theory = SyntheticSecularTheory(
	solar_system_synthetic_theory.masses,
	solar_system_synthetic_theory.semi_major_axes,
	solar_system_synthetic_theory.omega_vector,
	[truncate_dictionary(x_d,1e-3) for x_d in solar_system_synthetic_theory.x_dicts],
	[truncate_dictionary(y_d,1e-3) for y_d in solar_system_synthetic_theory.y_dicts]
)

merged_df = pd.read_csv("merged_elements.csv")

def ecc_inc_prediction(r):
	idx, row = r
	sim = rb.Simulation('planets-epoch-2460200.5.bin')
	sim.add(
		a=row['a'],
		e=row['e'],
		inc=row['Incl.'] * np.pi / 180,
		Omega=row['Node'] * np.pi / 180,
		omega=row['Peri.'] * np.pi / 180,
		primary=sim.particles[0]
	)
	sim.particles[5].m
	for i in range(4): # this removes the four terrestrial planets
		sim.remove(index=1)
	sim.move_to_com()
	cm.nbody_simulation_utilities.align_simulation(sim)
	sim.integrator = 'whfast'
	sim.dt = np.min([p.P for p in sim.particles[1:]]) / 25.
	sim.ri_whfast.safe_mode = 0
	a = sim.particles[5].a
	e = sim.particles[5].e
	inc = sim.particles[5].inc
	omega = sim.particles[5].omega
	Omega = sim.particles[5].Omega


	X = np.sqrt(2*(1-np.sqrt(1-e**2))) * np.exp(1j * omega)
	Y = 2*(1-e**2)**(0.25) * np.sin(0.5 * inc) * np.exp(1j * Omega)
	tp_h = TestParticleSecularHamiltonian(row['propa'], simpler_secular_theory)

	# leading order Hamiltonian
	h2_series = tp_h.H2_poisson_series()

	# list of 4th order terms
	sec_terms = list_secular_terms(4,4)
	h4_series = PoissonSeries(2,tp_h.synthetic_secular_theory.N_freq)
	for k,nu in sec_terms:
		for i in range(tp_h.synthetic_secular_theory.N_planets):
			h4_series+=tp_h.DFTerm_poisson_series(i,k,nu)

	# Strip terms that only depend on angles and not x,y,\bar{x},\bar{y}
	angle_only_term  = lambda term: (np.all(term.k==0) and np.all(term.kbar==0))
	h4_series_reduced = PoissonSeries.from_PSTerms([term for term in h4_series.terms if not angle_only_term(term)])

	# Hamiltonian
	h_tot = h2_series + h4_series_reduced
	ham_ps = PoissonSeriesHamiltonian(h_tot)

	u0 = X - np.sum(list(tp_h.F_e.values()))
	v0 = Y - np.sum(list(tp_h.F_inc.values()))

	np.savez(integration_path / f"integration_results_{row["Des'n"]}", u=u0, v = v0)
# %%
with Pool(40) as p:
	table = p.map(ecc_inc_prediction, merged_df.iterrows())