import matplotlib.pyplot as plt  # matplotlib for some plotting
import numpy as np  # numeric python functions
import pandas as pd  # need this to load our data from the csv files

from pyconturb import gen_turb, gen_spat_grid, TimeConstraint  # generate turbulence, useful helper
from pyconturb.sig_models import iec_sig  # IEC 61400-1 turbulence std dev
from pyconturb.spectral_models import kaimal_spectrum  # Kaimal spectrum
from pyconturb.wind_profiles import constant_profile, power_profile  # wind-speed profile functions

from _nb_utils import plot_slice
import h5py
import time
import sys

# copied from /Users/bstanisl/repos/pyconturb/pyconturb/duramat_constrained_turb.ipynb
# to run on hpc terminal in hopes that it's faster

start_time = time.time()

# inputs
save_turb_files_flag = True
tf = 30.0 # 30.0 #s
dt = 0.02 #s
parent_dir = '/projects/pvopt/brooke/duramat-validation-turbinflow/pyconturb/pyconturb-pvade/'
gen_csv_fname = 'generated_DuraMAT_tilt40deg_turbulent_inflow_{}s_{}Hz.csv'.format(int(tf),int(1/dt))

# should match dimensions of pvade sim
y_min = -13.1
y_max = 13.1
z_max = 20.0
l_char = 0.17
ny = int((y_max-y_min)/l_char)

# for testing
# y_min = -10.0
# y_max = 10.0
# z_max = 20.0
# ny = 80

# Step 1: generate dataframe of measurement data ----------------------------
# ---------------------------------------------------------------------------

# sonic position
x_sonic1 = 0.0
y_sonic1 = 4.5
z_sonic1 = 2.23 # m above ground

# define spatial info of measurement data
sonic_spat_df = gen_spat_grid(y_sonic1, z_sonic1)  # if `comps` not passed in, assumes all 3 components are wanted
# print('sonic_spat_df = ', sonic_spat_df.head())  # look at the first few rows

# Step 2: read csv of measurement data --------------------------------------
# ---------------------------------------------------------------------------

print(f'reading measured data from {parent_dir+gen_csv_fname} to generate con_tc', flush=True)
con_tc = TimeConstraint(pd.read_csv(parent_dir+gen_csv_fname, index_col=0))  # load data from csv directly into tc
con_tc.index = con_tc.index.map(lambda x: float(x) if (x not in 'kxyz') else x)  # index cleaning
# print('con_tc = ', con_tc.iloc[:7, :])  # look at the first 7 rows

# calc u_mean 
time_df = con_tc.get_time()
u_mean_sonic1 = time_df.filter(regex='u_', axis=1).mean()
print(f'u_mean_sonic1 = {u_mean_sonic1.values[0]:.3f} m/s', flush=True)

# Step 3: define spatial info of generated turbulence -----------------------
# ---------------------------------------------------------------------------
y = np.linspace(y_min, y_max, ny)

# heights are linearly increasing as you move above the surface
# steps = 0.15 + 0.04 * np.arange(1000) # 0.148 ensures a point at z_sonic = 2.23 m
steps = 0.15 + 0.02 * np.arange(1000) # 0.148 ensures a point at z_sonic = 2.23 m
print('last step = ', steps[-1], flush=True)
try:
    if steps[-1] < z_max:
        # print('WARNING: steps does not reach z_max')
        sys.exit('Terminating script because steps[-1] < z_max')
except SystemExit as e:
    print(e, flush=True)  # Optional: Print explanation before stopping
    raise  # Ensures the script terminates
z = 0.0001 + np.cumsum(steps)
z = z[z <= z_max]
nz = len(z)

dy = y[1] - y[0] # (y[-1]-y[0])/ny
dz = z[1] - z[0] # (z[-1]-z[0])/nz

# this resolution should be approximately equal to or smaller than l_char
print(f'ny = {ny}, nz = {nz}', flush=True)
print(f'dy = {dy:.3f} m, smallest dz = {dz:.5f} m', flush=True)

spat_df = gen_spat_grid(y, z)  # if `comps` not passed in, assumes all 3 components are wanted
# print('spat_df = ',spat_df.head())  # look at the first few rows

# Step 4: generate constrained turbulence -----------------------------------
# ---------------------------------------------------------------------------
fudge_factor = 2.1
u_ref = float(u_mean_sonic1.values[0])
z_ref = z_sonic1

u_ref_sim = u_ref * fudge_factor
print(f'generating turb, u_ref = {u_ref_sim:.3f} m/s, z_ref = {z_ref} m', flush=True)

kwargs = {'u_ref': u_ref_sim, 'turb_class': 'B', 'z_hub': z_ref,  # necessary keyword arguments for IEC turbulence
      'T': con_tc.get_T(), 'nt': con_tc.get_time().index.size}  # simulation length (s) and time step (s)
interp_data = 'none' # 'all'  # use the default IEC 61400-1 profile instead of interpolating from constraints

# generate turbulence
sim_turb_df = gen_turb(spat_df, con_tc=con_tc, interp_data=interp_data, verbose=True, **kwargs)

# Step 5: save turbulence to csv file ---------------------------------------
# ---------------------------------------------------------------------------
sim_turb_fname = f'constrained_turb_ny{ny}_nz{nz}_sonic1_{int(tf)}s_u{round(u_mean_sonic1.values[0],3)}_{int(1/dt)}Hz.csv'
if save_turb_files_flag:
    sim_turb_df.to_csv(sim_turb_fname)
    print(f'saved turbulence to csv file: {sim_turb_fname}', flush=True)

# Step 6: check accurace of generated turbulence ----------------------------
# ---------------------------------------------------------------------------

# reshape to 3D array
data = {}
data['u'] = sim_turb_df.filter(regex='u').values.reshape(len(sim_turb_df),y.size,z.size).transpose((0, 2, 1))
data['v'] = sim_turb_df.filter(regex='v').values.reshape(len(sim_turb_df),y.size,z.size).transpose((0, 2, 1))
data['w'] = sim_turb_df.filter(regex='w').values.reshape(len(sim_turb_df),y.size,z.size).transpose((0, 2, 1))

# print out error as a check
j = np.argmin(abs(y - y_sonic1))
k = np.argmin(abs(z - z_sonic1))

print(f'comparing timeseries at (y,z) = ({y[j]:.2f},{z[k]:.2f}) to sonic loc of ({y_sonic1:.2f},{z_sonic1:.2f})', flush=True)

usim = data['u'][:,k,j]
ucon = con_tc.get_time()['u_p0']

measured_u_mean = np.average(ucon)
simulated_u_mean = np.average(usim)
print(f'measured mean u = {measured_u_mean:.3f} m/s', flush=True)
print(f'simulated mean u = {simulated_u_mean:.3f} m/s', flush=True)
print('percent error of mean u = {:.2f}%'.format(100*(simulated_u_mean - measured_u_mean) / measured_u_mean), flush=True)
print('abs error = {:.2f} m/s'.format(np.average(abs(usim - ucon))), flush=True)

# Step 7: save turbulence to .h5 file for input to PVade --------------------
# ---------------------------------------------------------------------------

if save_turb_files_flag:
    h5_filename = f'pct_{sim_turb_fname[:-4]}.h5'

    t_steps = con_tc.get_time().index.size
    time_values = con_tc.get_time().index.values.astype(float)

    with h5py.File(h5_filename, "w") as fp:
        fp.create_dataset("time_index", shape=(t_steps,))
        fp["time_index"][:] = time_values
        
        fp.create_dataset("y_coordinates", shape=(ny,))
        fp["y_coordinates"][:] = y
            
        fp.create_dataset("z_coordinates", shape=(nz,))
        fp["z_coordinates"][:] = z
            
        fp.create_dataset("u", shape=(t_steps, nz, ny))
        fp["u"][:] = data['u'][:]
        
        fp.create_dataset("v", shape=(t_steps, nz, ny))
        fp["v"][:] = data['v'][:]
        
        fp.create_dataset("w", shape=(t_steps, nz, ny))
        fp["w"][:] = data['w'][:]
    
    print(f'saved turb to h5 file: {h5_filename}', flush=True)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds", flush=True)
