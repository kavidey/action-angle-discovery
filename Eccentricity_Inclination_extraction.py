import celmech as cm
import rebound as rb
import numpy as np
from test_particle_secular_hamiltonian import SyntheticSecularTheory, TestParticleSecularHamiltonian
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
from matplotlib import pyplot as plt
from celmech.poisson_series import PoissonSeriesHamiltonian 
import pickle
from celmech.disturbing_function import list_secular_terms
from celmech.poisson_series import PoissonSeries
from celmech.rk_integrator import RKIntegrator
from scipy.integrate import solve_ivp
import os
import pandas as pd
from multiprocessing import Pool

integration_path = "/home/kdey/action-angle-discovery/integrations/outer_planets/"
file_names = os.listdir(integration_path)
file_names = [x for x in file_names if x != 'planets.bin']

with open("solar_system_synthetic_solution.bin","rb") as fi:
    solar_system_synthetic_theory=pickle.load(fi)

truncate_dictionary = lambda d,tol: {key:val for key,val in d.items() if np.abs(val)>tol}
simpler_secular_theory = SyntheticSecularTheory(
    solar_system_synthetic_theory.masses,
    solar_system_synthetic_theory.semi_major_axes,
    solar_system_synthetic_theory.omega_vector,
    [truncate_dictionary(x_d,1e-3) for x_d in solar_system_synthetic_theory.x_dicts],
    [truncate_dictionary(y_d,1e-3) for y_d in solar_system_synthetic_theory.y_dicts]
)

def ecc_inc_prediction(filename):
    results = cm.nbody_simulation_utilities.get_simarchive_integration_results(integration_path + filename,coordinates='heliocentric')
    results['X'] = np.sqrt(2*(1-np.sqrt(1-results['e']**2))) * np.exp(1j * results['pomega'])
    results['Y'] = (1-results['e']**2)**(0.25) * np.sin(0.5 * results['inc'] ) * np.exp(1j * results['Omega'])
    tp_h = TestParticleSecularHamiltonian(np.mean(results['a'][-1]),simpler_secular_theory)
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
    print("{} terms total".format(len(h_tot.terms)))
    ham_ps = PoissonSeriesHamiltonian(h_tot)

    dt = 0.1 * np.min(2*np.pi / np.abs((tp_h.g0,tp_h.s0)))

    integrator = RKIntegrator(
    ham_ps.flow,
	lambda x: (ham_ps.flow(x), ham_ps.jacobian(x)),
    Ndim=2*(ham_ps.N+ham_ps.M),
    dt=dt,
    rtol=1e-4,
    atol=1e-7,
    rk_method='GL6',
    rk_root_method='Newton',
    max_iter=10
    ) 

    u0 = x_numerical[0] - np.sum(list(tp_h.F_e.values()))
    v0 = y_numerical[0] - np.sum(list(tp_h.F_inc.values()))
    _RT2 = np.sqrt(2)
    X = np.zeros(2*(h_tot.M+h_tot.N))
    X[:2] = -1 * _RT2 * np.imag(np.array([u0,v0]))
    X[h_tot.M+h_tot.N:h_tot.M+h_tot.N+2] = _RT2 * np.real(np.array([u0,v0]))

    steps = 200
    times = np.arange(steps) * integrator.dt
    X_rk = np.zeros((steps,X.size))
    for i in range(steps):
        X_rk[i] = X
        X = integrator.rk_step(X)
    
    Tfin = 0.1e6 * 2*np.pi
    Nout = 200
    t_eval = np.linspace(0,Tfin,Nout)
    soln_h = solve_ivp(
        lambda t,y: ham_ps.flow(y),
        (0,Tfin),
        X_rk[0],
        method="Radau",
        t_eval=t_eval,
        jac = lambda t,y: ham_ps.jacobian(y)
    )

    u_soln = (soln_h.y[h_tot.M+h_tot.N] - 1j * soln_h.y[0]) / np.sqrt(2)
    F_e_soln = np.sum([amp*np.exp(1j*(m @ soln_h.y[h_tot.N:h_tot.M+h_tot.N])) for m,amp in tp_h.F_e.items()],axis=0)    
    x_soln = u_soln+F_e_soln

    v_soln = (soln_h.y[1+h_tot.M+h_tot.N] - 1j * soln_h.y[1]) / np.sqrt(2)
    F_inc_soln = np.sum([amp*np.exp(1j*(m @ soln_h.y[h_tot.N:h_tot.M+h_tot.N])) for m,amp in tp_h.F_inc.items()],axis=0)    
    y_soln = v_soln+F_inc_soln

    u_rk_soln = (X_rk.T[h_tot.M+h_tot.N] - 1j * X_rk.T[0]) / np.sqrt(2)
    F_e_rk_soln = np.sum([amp*np.exp(1j*(m @ X_rk.T[h_tot.N:h_tot.M+h_tot.N])) for m,amp in tp_h.F_e.items()],axis=0)    
    x_rk_soln = u_rk_soln+F_e_rk_soln

    v_rk_soln = (X_rk.T[1+h_tot.M+h_tot.N] - 1j * X_rk.T[1]) / np.sqrt(2)

    F_inc_rk_soln = np.sum([amp*np.exp(1j*(m @ X_rk.T[h_tot.N:h_tot.M+h_tot.N])) for m,amp in tp_h.F_inc.items()],axis=0)    

    y_rk_soln = v_rk_soln+F_inc_rk_soln

    return [np.average(abs(u_soln)), np.average(abs(u_rk_soln)), np.average(results["e"][-1]), np.average(abs(v_soln)), np.average(abs(v_rk_soln)), np.average(results['inc'][-1]), tp_h.g0]

with Pool(20) as p:
	g_table = p.map(eccentricity_prediction, file_names)
	df = pd.DataFrame(g_table)
	df.to_csv("eccentricity_comparison.csv")