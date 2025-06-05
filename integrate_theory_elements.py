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
integration_path = Path("/home/lshen/action-angle-discovery/integrations") / ("new_ecc_inc_integrations_1")
integration_path.mkdir(parents=True, exist_ok=True)

simulation_path = "/home/lshen/action-angle-discovery/integrations/simulation_files/"
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

df = pd.read_fwf('/home/lshen/action-angle-discovery/MPCORB.DAT', colspecs=[[0,7], [8,14], [15,19], [20,25], [26,35], [36,46], [47, 57], [58,68], [69,81], [82, 91], [92, 103]])
df = df[df['Epoch'] == 'K239D']

df.infer_objects()
for c in ['a', 'e', 'Incl.', 'Node', 'Peri.', 'M']:
	df[c] = pd.to_numeric(df[c])

labels = pd.read_fwf('/home/lshen/action-angle-discovery/proper_catalog24.dat', colspecs=[[0,10], [10,18], [19,28], [29,37], [38, 46], [47,55], [56,66], [67,78], [79,85], [86, 89], [90, 97]], header=None, index_col=False, names=['propa', 'da', 'prope', 'de', 'propsini', 'dsini', 'g', 's', 'H', 'NumOpps', "Des'n"])

merged_df = pd.merge(df, labels, on="Des'n", how="inner")

def ecc_inc_prediction(r):
	idx, row = r
	Tfin_approx = 538000 * 2*np.pi
	sim = rb.Simulation('planets-epoch-2460200.5.bin')
	sim.add(
		a=row['a'],
		e=row['e'],
		inc=row['Incl.'] * np.pi / 180,
		Omega=row['Node'] * np.pi / 180,
		omega=row['Peri.'] * np.pi / 180,
		M=row['M'],
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
	def run_simulation(sim, Tfin_approx, Nout=100):
		total_steps = np.ceil(Tfin_approx / sim.dt)
		Tfin = total_steps * sim.dt + sim.dt

		sim.save_to_file(simulation_path + f"simulation_{row["Des'n"]}.sa", step=int(np.floor(total_steps / Nout)), delete_file=True)
		sim.integrate(Tfin, exact_finish_time=0)

		results = cm.nbody_simulation_utilities.get_simarchive_integration_results(
			simulation_path + f"simulation_{row["Des'n"]}.sa", coordinates='heliocentric'
		)
		return sim, results, Tfin

	while True:
		sim, results, Tfin = run_simulation(sim, Tfin_approx)

		tp_h = TestParticleSecularHamiltonian(np.mean(results['a'][-1]), simpler_secular_theory)
		minTsec = np.min(2*np.pi / np.abs((tp_h.g0,tp_h.s0)))
		dt = 0.1 * minTsec
		NTsec = Tfin / minTsec
		steps = int(Tfin / dt)
		times = np.arange(steps) * dt

		if NTsec >= 10:
			break
		else:
			print("NTsec too low. Doubling integration time.")
			Tfin_approx *= 2


	results['X'] = np.sqrt(2*(1-np.sqrt(1-results['e']**2))) * np.exp(1j * results['pomega'])
	results['Y'] = (1-results['e']**2)**(0.25) * np.sin(0.5 * results['inc'] ) * np.exp(1j * results['Omega'])
	x_numerical = results['X'][-1]
	y_numerical = 2*results['Y'][-1]
	x_linear,y_linear = tp_h.linear_theory_solution(x_numerical[0],y_numerical[0],results['time'])

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

	u0 = x_numerical[0] - np.sum(list(tp_h.F_e.values()))
	v0 = y_numerical[0] - np.sum(list(tp_h.F_inc.values()))
	_RT2 = np.sqrt(2)
	X = np.zeros(2*(h_tot.M+h_tot.N))
	X[:2] = -1 * _RT2 * np.imag(np.array([u0,v0]))
	X[h_tot.M+h_tot.N:h_tot.M+h_tot.N+2] = _RT2 * np.real(np.array([u0,v0]))

	soln_h = solve_ivp(
		lambda t,y: ham_ps.flow(y),
		(0,np.max(times)),
		X,
		method="Radau",
		t_eval=times,
		jac = lambda t,y: ham_ps.jacobian(y)
	)

	u_soln = (soln_h.y[h_tot.M+h_tot.N] - 1j * soln_h.y[0]) / np.sqrt(2)
	F_e_soln = np.sum([amp*np.exp(1j*(m @ soln_h.y[h_tot.N:h_tot.M+h_tot.N])) for m,amp in tp_h.F_e.items()],axis=0)    
	x_soln = u_soln+F_e_soln

	v_soln = (soln_h.y[1+h_tot.M+h_tot.N] - 1j * soln_h.y[1]) / np.sqrt(2)
	F_inc_soln = np.sum([amp*np.exp(1j*(m @ soln_h.y[h_tot.N:h_tot.M+h_tot.N])) for m,amp in tp_h.F_inc.items()],axis=0)    
	y_soln = v_soln+F_inc_soln

	u = u0
	y = soln_h.y
	t = soln_h.t

	np.savez(integration_path / f"integration_results_{row["Des'n"]}", u=u, y=y, t=t)
# %%
with Pool(40) as p:
	table = p.map(ecc_inc_prediction, merged_df.iterrows())