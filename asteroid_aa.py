###### MODIFIED FROM https://github.com/kavidey/learn-canonical-transform/blob/main/asteroid_aa.py ######
# %%
from pathlib import Path
import copy
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.random as jnr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from scipy.optimize import minimize

import rebound as rb

from celmech.nbody_simulation_utilities import get_simarchive_integration_results
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft

import sympy

jax.config.update("jax_enable_x64", True)
np.set_printoptions(suppress=True, precision=4, linewidth=100)
# %%
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def closest_key_entry(d, target):
    """
    Given a dictionary `d` with float keys and a target float `target`,
    returns a tuple (key, value) where the key is the one in `d`
    closest to `target`.

    Parameters
    ----------
    d : dict
        Dictionary with float keys.
    target : float
        The float to compare keys against.

    Returns
    -------
    tuple
        The (key, value) pair whose key is closest to `target`.
    """
    closest_key = min(d.keys(), key=lambda k: abs(k - target))
    return closest_key, d[closest_key]

def symmetrize_axes(axes):
    y_max = np.max(np.abs(axes.get_ylim()))
    x_max = np.max(np.abs(axes.get_xlim()))

    ax_max = np.max([x_max, y_max])

    axes.set_ylim(ymin=-ax_max, ymax=ax_max)
    axes.set_xlim(xmin=-ax_max, xmax=ax_max)
# %%
dataset_path = Path('integrations') / 'uncertainty_testing'
TO_ARCSEC_PER_YEAR = 60*60*180/np.pi * (2*np.pi)
# %%
sims = list(dataset_path.glob('*.sa'))
desns = list(map(lambda p: p.stem.split('_')[-1], sims))
desns = sorted(desns)

base_desn = desns[0]
# %%
rb_sim = rb.Simulation(str(dataset_path/'planets.bin'))
masses = np.array(list(map(lambda ps: ps.m, rb_sim.particles))[1:] + [1e-11], dtype=np.float64)
N = len(masses)

def load_sim(desn):
    results = get_simarchive_integration_results(str(dataset_path / f"{'_'.join(sims[0].stem.split('_')[:-1])}_{desn}.sa"), coordinates='heliocentric')
    
    # results['X'] = np.sqrt(2*(1-np.sqrt(1-results['e']**2))) * np.exp(1j * results['pomega'])
    # results['Y'] = (1-results['e']**2)**(0.25) * np.sin(0.5 * results['inc'] ) * np.exp(1j * results['Omega']) 

    # prefactor = np.sqrt(masses)[..., None].repeat(results['X'].shape[1], axis=-1) * np.power(rb_sim.G * rb_sim.particles[0].m * results['a'], 1/4)
    # results['G'] = prefactor * results['X']
    # results['F'] = prefactor * results['Y']
    m = masses[..., None].repeat(results['a'].shape[1], axis=-1)
    G = 1
    beta = ((1 * m) / (1 + m))
    mu = G * (1 + m)
    results['Lambda'] = beta * np.sqrt(mu * results['a'])
    
    M = results['l'] - results['pomega']
    results['lambda'] = M + results['pomega']

    results['x'] = np.sqrt(results['Lambda']) * np.sqrt(1 - np.sqrt(1-results['e']**2)) * np.exp(1j * results['pomega'])
    results['y'] = np.sqrt(2 * results['Lambda']) * np.power(1-results['e']**2, 1/4) * np.sin(results['inc']/2) * np.exp(1j * results['Omega'])

    # coordinate pairs are:
    # - Lambda, Lambda
    # - x, -i * x_bar
    # - y, -i * y_bar

    return results
# %%
base_sim = load_sim(base_desn)

planets = ("Jupiter","Saturn","Uranus","Neptune")
planet_ecc_fmft = dict()
planet_inc_fmft = dict()
for i,pl in enumerate(planets):
    planet_ecc_fmft[pl] = fmft(base_sim['time'],base_sim['x'][i],14)
    planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))
    planet_e_freqs_arcsec_per_yr = planet_e_freqs * TO_ARCSEC_PER_YEAR

    planet_inc_fmft[pl] = fmft(base_sim['time'],base_sim['y'][i],8)
    planet_inc_freqs = np.array(list(planet_inc_fmft[pl].keys()))
    planet_inc_freqs_arcsec_per_yr = planet_inc_freqs * TO_ARCSEC_PER_YEAR

    print("")
    print(pl)
    print("g")
    print("-------")
    for g in planet_e_freqs[:6]:
        print("{:+07.3f} \t {:0.6f}".format(g * TO_ARCSEC_PER_YEAR, np.abs(planet_ecc_fmft[pl][g])))
    print("s")
    print("-------")
    for s in planet_inc_freqs[:4]:
        print("{:+07.3f} \t {:0.6f}".format(s * TO_ARCSEC_PER_YEAR, np.abs(planet_inc_fmft[pl][s])))
# %%
g_vec = np.zeros(4)
s_vec = np.zeros(4)

g_vec[:3] = np.array(list(planet_ecc_fmft['Jupiter'].keys()))[:3]
g_vec[3] = list(planet_ecc_fmft['Neptune'].keys())[0]
s_vec[0] = list(planet_inc_fmft['Jupiter'].keys())[0]
s_vec[1] = list(planet_inc_fmft['Jupiter'].keys())[1]
s_vec[2] = list(planet_inc_fmft['Jupiter'].keys())[2]
s_vec[3] = list(planet_inc_fmft['Jupiter'].keys())[3]

omega_vec = np.concatenate((g_vec,s_vec))
g_and_s_arc_sec_per_yr = omega_vec * TO_ARCSEC_PER_YEAR
with np.printoptions(suppress=True, precision=3):
    print(g_and_s_arc_sec_per_yr)
# %%
freq_thresh = 0.05
ecc_rotation_matrix_T_base = np.zeros((5, 5))

# planet ecc in terms of planet modes
mode_angle = [0, 0, 0, 0, 0]
for i, pl in enumerate(planets):
    planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))
    print(pl)
    for j, g in enumerate(g_vec):
        found_g = find_nearest(planet_e_freqs, g)
        with np.printoptions(suppress=True, precision=3):
            print(np.array([found_g*TO_ARCSEC_PER_YEAR, g*TO_ARCSEC_PER_YEAR, np.abs((found_g - g)/g) * 100]), np.abs((found_g - g)/g) < freq_thresh)
        if np.abs((found_g - g)/g) > freq_thresh:
            continue
        matrix_entry = np.abs(planet_ecc_fmft[pl][found_g])
        if mode_angle[i] == 0:
            mode_angle[i] = np.angle(planet_ecc_fmft[pl][found_g])
        if mode_angle[i] != 0 and np.abs(mode_angle[i] - np.angle(planet_ecc_fmft[pl][found_g])) > np.pi/2:
            matrix_entry *= -1
        ecc_rotation_matrix_T_base[i][j] += matrix_entry
    print()

mode_angle = [0, 0, 0, 0, 0]
inc_rotation_matrix_T_base = np.zeros((5, 5))
for i, pl in enumerate(planets):
    planet_i_freqs = np.array(list(planet_inc_fmft[pl].keys()))
    for j, s in enumerate(s_vec):
        found_s = find_nearest(planet_i_freqs, s)
        if np.abs((found_s - s)/s) > freq_thresh:
            continue
        matrix_entry = np.abs(planet_inc_fmft[pl][found_s])
        if mode_angle[i] == 0:
            mode_angle[i] = np.angle(planet_inc_fmft[pl][found_s])
        if mode_angle[i] != 0 and np.abs(mode_angle[i] - np.angle(planet_inc_fmft[pl][found_s])) > np.pi/2:
            matrix_entry *= -1
        inc_rotation_matrix_T_base[i][j] = matrix_entry
# %%
asteroid_elements = []
for desn in tqdm(desns):
    sim = load_sim(desn)
    asteroid_ecc_fmft = fmft(sim['time'],sim['x'][-1],14)
    asteroid_inc_fmft = fmft(sim['time'],sim['y'][-1],8)

    ecc_rotation_matrix_T = ecc_rotation_matrix_T_base.copy()
    inc_rotation_matrix_T = inc_rotation_matrix_T_base.copy()

    asteroid_e_freqs = np.array(list(asteroid_ecc_fmft.keys()))
    asteroid_i_freqs = np.array(list(asteroid_inc_fmft.keys()))
    print(f"Asteroid {desn}"+"\ng\n--------")
    for g in asteroid_e_freqs[:6]:
        print("{:+07.3f} \t {:0.6f}".format(g * TO_ARCSEC_PER_YEAR, np.abs(asteroid_ecc_fmft[g])))
    print('\ns\n--------')
    for s in asteroid_i_freqs[:4]:
        print("{:+07.3f} \t {:0.6f}".format(s * TO_ARCSEC_PER_YEAR, np.abs(asteroid_inc_fmft[s])))
    print()
    ## ECC ###
    # asteroid ecc in terms of planet modes
    for j, g in enumerate(g_vec):
        found_g = find_nearest(asteroid_e_freqs, g)
        if np.abs((found_g - g)/g) > 0.1:
            continue
        ecc_rotation_matrix_T[4][j] = np.abs(asteroid_ecc_fmft[found_g])

    # find the largest mode that isn't in g_vec, thats the asteroid mode
    for g in asteroid_e_freqs:
        found_g = find_nearest(g_vec, g)
        if np.abs((found_g - g)/g) > 0.1:
            asteroid_g = g
            break
    
    # asteroid ecc in terms of its own mode
    ecc_rotation_matrix_T[4][4] = np.abs(asteroid_ecc_fmft[g])
    
    ### INC ###
    # asteroid inc in terms of planet modes
    for j, s in enumerate(s_vec):
        found_s = find_nearest(asteroid_i_freqs, s)
        if np.abs((found_s - s)/s) > 0.1:
            continue
        inc_rotation_matrix_T[4][j] = np.abs(asteroid_inc_fmft[found_s])

    # find the largest mode that isn't in g_vec, thats the asteroid mode
    for s in asteroid_i_freqs:
        found_s = find_nearest(s_vec, s)
        if np.abs((found_s - s)/s) > 0.1:
            asteroid_s = s
            break
    
    # asteroid ecc in terms of its own mode
    inc_rotation_matrix_T[4][4] = np.abs(asteroid_inc_fmft[s])
    # break

    ecc_rotation_matrix_T = ecc_rotation_matrix_T / np.linalg.norm(ecc_rotation_matrix_T, axis=0)
    inc_rotation_matrix_T = inc_rotation_matrix_T / np.linalg.norm(inc_rotation_matrix_T, axis=0)

    # print("ecc")
    # print(ecc_rotation_matrix_T)
    # print("inc")
    # print(inc_rotation_matrix_T)
    # %%
    def objective(R, G, m):
        R = jnp.reshape(R, (N,N))

        rotation_loss = ((jnp.eye(N) - R @ R.T) ** 2).sum()# + (jnp.linalg.det(R) - 1) ** 2

        Phi = R.T @ G

        J_approx = jnp.abs(Phi).mean(axis=1)
        J_loss = ((jnp.abs(Phi) - J_approx[..., None]) ** 2).sum()

        off_diag_weight = 1 / jnp.pow(jnp.outer(J_approx, J_approx), 1/4)
        # off_diag_weight = jnp.fill_diagonal(off_diag_weight, off_diag_weight.max(), inplace=False)
        off_diag_loss = (((jnp.ones((N,N))-jnp.eye(N)) * R.T * off_diag_weight) ** 2).sum()
        # off_diag_loss = (((R.T - jnp.eye(N)) ** 2) * off_diag_weight).sum()

        loss = rotation_loss + J_loss + off_diag_loss * 1e-10
        return loss

    obj_and_grad = jax.jit(jax.value_and_grad(lambda R: objective(R, sim['x'], masses)))

    sol = minimize(obj_and_grad, ecc_rotation_matrix_T.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
    ecc_rotation_matrix_opt_T = sol.x.reshape(5,5)

    # print(np.linalg.det(ecc_rotation_matrix_opt_T))
    # print(ecc_rotation_matrix_opt_T @ ecc_rotation_matrix_opt_T.T)

    # print("original\n", ecc_rotation_matrix_T)
    # print("optimized\n", ecc_rotation_matrix_opt_T)
    print("ecc")
    print(ecc_rotation_matrix_opt_T)

    Phi = (np.linalg.inv(ecc_rotation_matrix_opt_T) @ sim['x'])

    # fig, axs = plt.subplots(2,5,figsize=(15, 5))
    # for i, pl in enumerate(planets + ('Asteroid',)):
    #     axs[0][i].set_title(pl)
    #     pts = sim['x'][i]
    #     axs[0][i].plot(np.real(pts), np.imag(pts))
    #     axs[0][i].set_aspect('equal')
    #     pts = Phi[i]
    #     axs[1][i].plot(np.real(pts), np.imag(pts))
    #     axs[1][i].set_aspect('equal')
    # %%
    def objective(R, G, m):
        R = jnp.reshape(R, (N,N))

        rotation_loss = ((jnp.eye(N) - R @ R.T) ** 2).sum()# + (jnp.linalg.det(R) - 1) ** 2

        Phi = R.T @ G

        J_approx = jnp.abs(Phi).mean(axis=1)
        J_loss = ((jnp.abs(Phi) - J_approx[..., None]) ** 2).sum()

        off_diag_weight = 1 / jnp.pow(jnp.outer(J_approx, J_approx), 1/4)
        # off_diag_weight = jnp.fill_diagonal(off_diag_weight, off_diag_weight.max(), inplace=False)
        off_diag_loss = (((jnp.ones((N,N))-jnp.eye(N)) * R.T * off_diag_weight) ** 2).sum()
        # off_diag_loss = (((R.T - jnp.eye(N)) ** 2) * off_diag_weight).sum()

        loss = rotation_loss + J_loss + off_diag_loss * 1e-10
        return loss

    obj_and_grad = jax.jit(jax.value_and_grad(lambda R: objective(R, sim['y'], masses)))

    sol = minimize(obj_and_grad, ecc_rotation_matrix_T.reshape(-1), options={'gtol': 1e-8, 'disp': True}, jac=True)
    inc_rotation_matrix_opt_T = sol.x.reshape(5,5)

    # print(np.linalg.det(inc_rotation_matrix_opt_T))
    # print(inc_rotation_matrix_opt_T @ inc_rotation_matrix_opt_T.T)

    Theta = (np.linalg.inv(inc_rotation_matrix_opt_T) @ sim['y'])

    # fig, axs = plt.subplots(2,5,figsize=(15, 5))
    # for i, pl in enumerate(planets + ('Asteroid',)):
    #     axs[0][i].set_title(pl)
    #     pts = sim['y'][i]
    #     axs[0][i].plot(np.real(pts), np.imag(pts))
    #     axs[0][i].set_aspect('equal')
    #     pts = Theta[i]
    #     axs[1][i].plot(np.real(pts), np.imag(pts))
    #     axs[1][i].set_aspect('equal')
    # %%
    planet_ecc_fmft = {}
    planet_inc_fmft = {}
    for i,pl in enumerate(planets + ("Asteroid",)):
        planet_ecc_fmft[pl] = fmft(sim['time'],Phi[i],14)
        planet_e_freqs = np.array(list(planet_ecc_fmft[pl].keys()))

        planet_inc_fmft[pl] = fmft(sim['time'],Theta[i],14)
        planet_i_freqs = np.array(list(planet_inc_fmft[pl].keys()))
        
        if pl == "Asteroid":
            print("")
            print(pl)
            print("-------")
            for g in planet_e_freqs[:8]:
                print(f"{g * TO_ARCSEC_PER_YEAR:+07.3f} \t {np.abs(planet_ecc_fmft[pl][g]):0.8f} ∠{np.angle(planet_ecc_fmft[pl][g]):.2f}")
            print("s")
            print("-------")
            for s in planet_i_freqs[:4]:
                print(f"{s * TO_ARCSEC_PER_YEAR:+07.3f} \t {np.abs(planet_inc_fmft[pl][s]):0.6f} ∠{np.angle(planet_inc_fmft[pl][s]):.2f}")
        
        # return planet_ecc_fmft, planet_inc_fmft

    # planet_ecc_fmft, planet_inc_fmft = planet_fmft(base_sim['time'], Phi, Theta, display=True)
    # %%
    g_vec = np.zeros(5)
    s_vec = np.zeros(5)

    g_vec[0] = list(planet_ecc_fmft['Jupiter'].keys())[0]
    g_vec[1] = list(planet_ecc_fmft['Saturn'].keys())[0]
    g_vec[2] = list(planet_ecc_fmft['Uranus'].keys())[0]
    g_vec[3] = list(planet_ecc_fmft['Neptune'].keys())[0]
    # g_vec[4] = list(planet_ecc_fmft['Asteroid'].keys())[0]
    for g in list(planet_ecc_fmft['Asteroid'].keys()):
        if np.min(np.abs(g_vec[:4] - g)*TO_ARCSEC_PER_YEAR) > 0.1:
            g_vec[4] = g
            break

    s_vec[0] = list(planet_inc_fmft['Jupiter'].keys())[0]
    s_vec[1] = list(planet_inc_fmft['Saturn'].keys())[0]
    s_vec[2] = list(planet_inc_fmft['Uranus'].keys())[0]
    s_vec[3] = list(planet_inc_fmft['Neptune'].keys())[0]
    # s_vec[4] = list(planet_inc_fmft['Asteroid'].keys())[0]
    for s in list(planet_inc_fmft['Asteroid'].keys()):
        if np.min(np.abs(s_vec[:4] - s)*TO_ARCSEC_PER_YEAR) > 0.1:
            s_vec[4] = s
            break

    g_amp = np.zeros(5, dtype=np.complex128)
    s_amp = np.zeros(5, dtype=np.complex128)

    g_amp[0] = planet_ecc_fmft['Jupiter'][g_vec[0]]
    g_amp[1] = planet_ecc_fmft['Saturn'][g_vec[1]]
    g_amp[2] = planet_ecc_fmft['Uranus'][g_vec[2]]
    g_amp[3] = planet_ecc_fmft['Neptune'][g_vec[3]]
    g_amp[4] = planet_ecc_fmft['Asteroid'][g_vec[4]]

    s_amp[0] = planet_inc_fmft['Jupiter'][s_vec[0]]
    s_amp[1] = planet_inc_fmft['Saturn'][s_vec[1]]
    s_amp[2] = planet_inc_fmft['Uranus'][s_vec[2]]
    s_amp[3] = planet_inc_fmft['Neptune'][s_vec[3]]
    s_amp[4] = planet_inc_fmft['Asteroid'][s_vec[4]]

    omega_vec = np.concat([g_vec, s_vec])
    omega_amp = np.concat([g_amp, s_amp])

    s_conserved_idx = np.argmin(np.abs(omega_vec[N:])) + N

    print(omega_vec * TO_ARCSEC_PER_YEAR)
    print(omega_amp)
    # %%
    g = omega_vec[4] * TO_ARCSEC_PER_YEAR
    s = omega_vec[9] * TO_ARCSEC_PER_YEAR

    x_amp = np.abs(omega_amp[4]) / ecc_rotation_matrix_opt_T[4,4]
    y_amp = np.abs(omega_amp[9]) / inc_rotation_matrix_opt_T[4,4]

    a = sim['a'][4].mean()
    Lambda = sim['Lambda'][4].mean()
    e = (x_amp * np.sqrt(2*Lambda - x_amp**2)) / Lambda
    sini = np.sin(2 * np.arcsin(y_amp / (np.sqrt(2*Lambda) * np.power(1-e**2, 1/4))))

    a_elements = [desn, float(g), float(s), float(a), float(e), float(sini)]
    print(a_elements)
    asteroid_elements.append(a_elements)
# %%
df = pd.DataFrame(asteroid_elements, columns=["Des'n", "g", "s", "propa", "prope", "propsini"])
df.to_csv("tables_for_analysis/uncertainty_asteroid_elements.csv")
# %%
df = pd.read_csv("tables_for_analysis/uncertainty_asteroid_elements.csv")
merged_elements = pd.read_csv("merged_elements.csv")

target_asteroids = ["K23Q65H", "K23Q67X"]
qtys = ["g","s","propa","prope","propsini"]
# %%
fig, axs = plt.subplots(len(target_asteroids), len(qtys), figsize=(15, 5))

for i,asteroid in enumerate(target_asteroids):
    row = merged_elements[merged_elements["Des'n"] == asteroid]
    axs[i,0].set_ylabel(asteroid)
    for j,qty in enumerate(qtys):
        if i == 0:
            axs[i,j].set_title(qty)
        axs[i,j].hist(df[df["Des'n"].str.contains(asteroid)][qty])
        axs[i,j].axvline(row[qty].item(), linestyle="--", color="black")
plt.tight_layout()
# %%
