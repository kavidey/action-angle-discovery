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
integration_path = Path("action-angle-discovery/integrations") / ("ecc_inc_integrations")
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

df = pd.read_fwf('/home/lshen/action-angle-discovery/MPCORB.DAT', colspecs=[[0,7], [8,14], [15,19], [20,25], [26,35], [36,46], [47, 57], [58,68], [69,81], [82, 91], [92, 103]])
df = df[df['Epoch'] == 'K239D']

df.infer_objects()
for c in ['a', 'e', 'Incl.', 'Node', 'Peri.', 'M']:
	df[c] = pd.to_numeric(df[c])

labels = pd.read_fwf('/home/lshen/action-angle-discovery/proper_catalog24.dat', colspecs=[[0,10], [10,18], [19,28], [29,37], [38, 46], [47,55], [56,66], [67,78], [79,85], [86, 89], [90, 97]], header=None, index_col=False, names=['propa', 'da', 'prope', 'de', 'propsini', 'dsini', 'g', 's', 'H', 'NumOpps', "Des'n"])

merged_df = pd.merge(df, labels, on="Des'n", how="inner")

sim = rb.Simulation()
date = "2023-09-13 12:00"
sim.add("Sun", date=date)
if not outer_only:
    sim.add("Mercury", date=date)
    sim.add("Venus", date=date)
    sim.add("Earth", date=date)
    sim.add("Mars", date=date)
sim.add("Jupiter", date=date)
sim.add("Saturn", date=date)
sim.add("Uranus", date=date)
sim.add("Neptune", date=date)
sim.save_to_file(str(integration_path / "planets.bin"))
# %%
def ecc_inc_prediction(r):
    idx, row = r
    sim = rb.Simulation(str(integration_path / 'planets.bin'))
    sim.add(a=row['a'], e=row['e'], inc=row['Incl.']*np.pi/180, Omega=row['Node']*np.pi/180, omega=row['Peri.']*np.pi/180, M=row['M'], primary=sim.particles[0])
    sim.move_to_com()
    p = sim.particles[-1]  # assuming this is Vesta
    e = p.e
    inc = p.inc
    pomega = p.pomega
    Omega = p.Omega
    a = p.a

    X = np.sqrt(2 * (1 - np.sqrt(1 - e**2))) * np.exp(1j * pomega)
    Y = (1 - e**2)**0.25 * np.sin(0.5 * inc) * np.exp(1j * Omega)
    x_numerical = X
    y_numerical = 2 * Y
    tp_h = TestParticleSecularHamiltonian(a, simpler_secular_theory)
    t_array = np.linspace(0, 1e6 * 2*np.pi, 1000)
    x_linear, y_linear = tp_h.linear_theory_solution(x_numerical, y_numerical, t_array)

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

    dt = 0.1 * np.min(2*np.pi / np.abs((tp_h.g0,tp_h.s0)))

    u0 = x_numerical - np.sum(list(tp_h.F_e.values()))
    v0 = y_numerical - np.sum(list(tp_h.F_inc.values()))
    _RT2 = np.sqrt(2)
    X = np.zeros(2*(h_tot.M+h_tot.N))
    X[:2] = -1 * _RT2 * np.imag(np.array([u0,v0]))
    X[h_tot.M+h_tot.N:h_tot.M+h_tot.N+2] = _RT2 * np.real(np.array([u0,v0]))

    steps = 200
    times = np.arange(steps) * dt

    Tfin = steps * dt
    Nout = 200
    t_eval = np.linspace(0,Tfin,Nout)
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

    t = soln_h.t
    y = soln_h.y

    np.savez(integration_path / f"integration_results_{row["Des'n"]}", t, y)
# %%
with Pool(40) as p:
	table = p.map(ecc_inc_prediction, merged_df.iterrows())