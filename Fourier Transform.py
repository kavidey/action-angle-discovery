import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product, combinations_with_replacement
from multiprocessing import Pool
import os
import rebound as rb
import celmech as cm
from celmech.miscellaneous import frequency_modified_fourier_transform as fmft

# Read the table with the defined column specifications
df = pd.read_fwf('MPCORB.DAT', colspecs=[[0,7], [8,14], [15,19], [20,25], [26,35], [36,46], [47, 57], [58,68], [69,81], [82, 91], [92, 103]])
df = df[df['Epoch'] == 'K239D'] # take only ones at common epoch--almost all of them

df.infer_objects()
for c in ['a', 'e', 'Incl.', 'Node', 'Peri.', 'M']:
	df[c] = pd.to_numeric(df[c])

labels = pd.read_fwf('proper_catalog24.dat', colspecs=[[0,10], [10,18], [19,28], [29,37], [38, 46], [47,55], [56,66], [67,78], [79,85], [86, 89], [90, 97]], header=None, index_col=False, names=['propa', 'da', 'prope', 'de', 'propsini', 'dsini', 'g', 's', 'H', 'NumOpps', "Des'n"])

merged_df = pd.merge(df, labels, on="Des'n", how="inner")

integration_path = "/home/kdey/action-angle-discovery/integrations/outer_planets/"
file_names = os.listdir(integration_path)

transform_path = "/home/lshen/fourier_integration_results/"

def k_vec_generation(max_order = 3, size = 7):
	kvecs = []
	
	for order in range(1, max_order + 1):
		for positions in combinations_with_replacement(range(size), order):
			for variables in combinations_with_replacement([-1, 1], order):
				k = np.zeros(size, dtype = int)
				for i, n in zip(positions, variables):
					k[i] += n
				kvecs.append(k)
	kvecs = np.array(kvecs)
	indices = np.where((kvecs == np.zeros(size)).all(axis = 1))
	kvecs = np.delete(kvecs, indices, axis=0)
	return kvecs

k_vecs = k_vec_generation(max_order = 3, size = 7)
k_vecs

def Fourier_transform(filename):
	results = cm.nbody_simulation_utilities.get_simarchive_integration_results(integration_path + filename,coordinates='heliocentric')
	row = merged_df[merged_df["Des'n"] == filename[21:26]]
	def closest_key_entry(d, target):
		closest_key = min(d.keys(), key=lambda k: abs(k - target))
		return closest_key, d[closest_key]
	results['X'] = np.sqrt(2*(1-np.sqrt(1-results['e']**2))) * np.exp(1j * results['pomega'])
	results['Y'] = (1-results['e']**2)**(0.25) * np.sin(0.5 * results['inc'] )* np.exp(1j * results['Omega'])
	inner_planet_offset = 0

	planets = ("Jupiter","Saturn","Uranus","Neptune", "Asteroid")
	ecc_fmft_results = dict()
	inc_fmft_results = dict()
	for i,pl in enumerate(planets):
		ecc_fmft_results[pl] = fmft(results['time'],results['X'][i + inner_planet_offset],14)
		planet_e_freqs = np.array(list(ecc_fmft_results[pl].keys()))
		planet_e_freqs_arcsec_per_yr = planet_e_freqs * 60*60*180/np.pi * (2*np.pi)

		inc_fmft_results[pl] = fmft(results['time'],results['Y'][i + inner_planet_offset],8)
		planet_inc_freqs = np.array(list(inc_fmft_results[pl].keys()))
		planet_inc_freqs_arcsec_per_yr = planet_inc_freqs * 60*60*180/np.pi * (2*np.pi)
	ARCSEC_PER_YR = 1/(180*60*60*2)
	g_vec = np.zeros(4)
	s_vec = np.zeros(3)

	g_vec[:3] = np.array(list(ecc_fmft_results['Jupiter'].keys()))[:3]
	g_vec[3] = list(ecc_fmft_results['Neptune'].keys())[0]
	s_vec[0] = list(inc_fmft_results['Jupiter'].keys())[0]
	s_vec[1] = list(inc_fmft_results['Jupiter'].keys())[2]
	s_vec[2] = list(inc_fmft_results['Jupiter'].keys())[1]
	omega_vec = np.concatenate((g_vec,s_vec))
	g_and_s_arc_sec_per_yr = omega_vec / ARCSEC_PER_YR
	g_and_s_arc_sec_per_yr
	eye_N = np.eye(omega_vec.size,dtype = int)

	x_dicts = []
	for pl in planets[:4]:
		x_dict = {}
		for i,omega_i in enumerate(omega_vec[:4]):
			omega_N,amp = closest_key_entry(ecc_fmft_results[pl],omega_vec[i])
			omega_error = np.abs(omega_N/omega_i-1)
			if omega_error<0.001:
				x_dict[tuple(eye_N[i])] = amp
		#NL terms
		for a in range(7):
			for b in range(a,7):
				for c in range(7):
					if c==a:
						continue
					if c==b:
						continue
					k = np.zeros(7,dtype = int)
					k[a] +=1
					k[b] +=1
					k[c] -=1
					omega=k@omega_vec
					omega_N,amp = closest_key_entry(ecc_fmft_results[pl],omega)
					omega_error = np.abs(omega_N/omega-1)
					if omega_error<0.001:
						x_dict[tuple(k)] = amp
		x_dicts.append(x_dict)
	for pl in planets[4:]:
		x_dict = {}
		for k in k_vecs:
			omega=k@omega_vec
			omega_N,amp = closest_key_entry(ecc_fmft_results[pl],omega)
			omega_error = np.abs(omega_N/omega-1)
			if omega_error<0.001:
				x_dict[tuple(k)] = amp
				if omega in ecc_fmft_results[pl]:
					del ecc_fmft_results[pl][omega]
				else:
					closest_key = min(ecc_fmft_results[pl].keys(), key=lambda key: abs(key - omega))
					del ecc_fmft_results[pl][closest_key]
		x_dicts.append(x_dict)
	
	# The plots: 
	fig,ax = plt.subplots(5 ,2,sharex=True,figsize=(10,10))
	Xsolns = []
	Ysolns = []
	for i,pl in enumerate(planets):
		freq_amp_dict = ecc_fmft_results[pl]
		ax[i,0].plot(results['time'],np.real(results['X'][i + inner_planet_offset]))
		zsoln = np.sum([np.abs(amp) * np.exp(1j*((np.array(k)@omega_vec)*results['time'] + + np.angle(amp))) for k,amp in x_dicts[i].items()],axis=0)
		ax[i,0].plot(results['time'],np.real(zsoln),'k--')
	plt.savefig(transform_path + "plots/" + filename + ".png")

	freq_amp_dict = ecc_fmft_results['Asteroid']
	zsoln = np.sum([np.abs(amp) * np.exp(1j*((np.array(k)@omega_vec)*results['time'] + np.angle(amp))) for k,amp in x_dicts[-1].items()],axis=0)
	soln_wo_planets = results['X'][i + inner_planet_offset] - zsoln
	ecc_fmft_result = fmft(results['time'], soln_wo_planets, 14)
	asteroid_e_freqs = np.array(list(ecc_fmft_result.keys()))
	asteroid_e_freqs_arcsec_per_yr = asteroid_e_freqs * 60*60*180/np.pi * (2*np.pi)

	g_list = [filename, row['g'].values[0]]
	for g in asteroid_e_freqs[:4]:
		g_list.append(g)

	return g_list

with Pool(20) as p:
	g_table = p.map(Fourier_transform, file_names)
	df = pd.DataFrame(g_table)
	df.to_csv("Fourier Transform results.csv")
	