# %%
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import rebound as rb
import celmech as cm

import sys
sys.path.insert(0, 'SBDynT/src')
import sbdynt as sbd

outer_only = True
print(f"Outer Planets Only: {outer_only}")
# %%
# integration_path = Path("integrations") / ("outer_planets" if outer_only else "full_system")
integration_path = Path("integrations") / "uncertainty_testing"
integration_path.mkdir(parents=True, exist_ok=True)
# %%
# Read the table with the defined column specifications
# df = pd.read_fwf('MPCORB.DAT', colspecs=[[0,7], [8,14], [15,19], [20,25], [26,35], [36,46], [47, 57], [58,68], [69,81], [82, 91], [92, 103]])
# df = df[df['Epoch'] == 'K239D'] # take only ones at common epoch--almost all of them

# df.infer_objects()
# for c in ['a', 'e', 'Incl.', 'Node', 'Peri.', 'M']:
# 	df[c] = pd.to_numeric(df[c])

# labels = pd.read_fwf('proper_catalog24.dat', colspecs=[[0,10], [10,18], [19,28], [29,37], [38, 46], [47,55], [56,66], [67,78], [79,85], [86, 89], [90, 97]], header=None, index_col=False, names=['propa', 'da', 'prope', 'de', 'propsini', 'dsini', 'g', 's', 'H', 'NumOpps', "Des'n"])

# merged_df = pd.merge(df, labels, on="Des'n", how="inner")
merged_df = pd.read_csv("tables_for_analysis/uncertainty_asteroid_testing.csv", index_col=0)
asteroid_time = {
    "K23Q65H": 2460193.500000,
    "K23Q67X": 2460180.500000
}
start_time = 2460200.5
# %%
sim = rb.Simulation()
# date = "JD"+str(start_time)
# sim.add("Sun", date=date)
# if not outer_only:
#     sim.add("Mercury", date=date)
#     sim.add("Venus", date=date)
#     sim.add("Earth", date=date)
#     sim.add("Mars", date=date)
# sim.add("Jupiter", date=date)
# sim.add("Saturn", date=date)
# sim.add("Uranus", date=date)
# sim.add("Neptune", date=date)
(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='sun',epoch=start_time)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))
if not outer_only:
    (flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='mercury',epoch=start_time)
    if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

    (flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='venus',epoch=start_time)
    if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

    (flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='earth',epoch=start_time)
    if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

    (flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='mars',epoch=start_time)
    if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='jupiter',epoch=start_time)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='saturn',epoch=start_time)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='uranus',epoch=start_time)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

(flag, mass, radius, [plx,ply,plz],[plvx,plvy,plvz]) = sbd.query_horizons_planets(obj='neptune',epoch=start_time)
if flag: sim.add(m=mass, x=plx, y=ply, z=plz, vx=plvx/(np.pi*2), vy=plvy/(np.pi*2), vz=plvz/(np.pi*2))

if outer_only:
    assert len(sim.particles) == 5, "Error adding planets"
else:
    assert len(sim.particles) == 9, "Error adding planets"

sim.save_to_file(str(integration_path/"planets.bin"))
# %%
def run_sim(r):
    idx, row = r
    # sim = rb.Simulation('planets.bin')
    # if outer_only:
    #     for i in range(4):
    #         sim.remove(1)
    sim = rb.Simulation(str(integration_path/'planets.bin'))
    rel_int_time = (asteroid_time[row["Des'n"]]-start_time) / ((1/(2.*np.pi))*365.25)
    sim.integrate(rel_int_time)
    sim.move_to_hel()
    # sim.add(a=row['a'], e=row['e'], inc=row['Incl.']*np.pi/180, Omega=row['Node']*np.pi/180, omega=row['Peri.']*np.pi/180, M=row['M'], primary=sim.particles[0])
    sim.add(x=row['x'], y=row['y'], z=row['z'], vx=row['vx']/(np.pi*2), vy=row['vy']/(np.pi*2), vz=row['vz']/(np.pi*2))
    sim.move_to_com()

    ps = sim.particles
    ps[-1].a

    sim.integrator='whfast'
    sim.dt = ps[1].P/100.
    sim.ri_whfast.safe_mode = 0

    Tfin_approx = 5e7*ps[-1].P
    total_steps = np.ceil(Tfin_approx / sim.dt)
    Tfin = total_steps * sim.dt + sim.dt
    Nout = 50_000

    sim_file = integration_path / f"asteroid_integration_{row["Des'n"]}-{idx}.sa"
    sim.save_to_file(str(sim_file), step=int(np.floor(total_steps/Nout)), delete_file=True)
    sim.integrate(Tfin, exact_finish_time=0)
# %%
with Pool(40) as p:
      p.map(run_sim, merged_df.iterrows())
# %%
