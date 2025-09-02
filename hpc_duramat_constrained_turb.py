import matplotlib.pyplot as plt  # matplotlib for some plotting
import numpy as np  # numeric python functions
import pandas as pd  # need this to load our data from the csv files

from pyconturb import gen_turb, gen_spat_grid, TimeConstraint  # generate turbulence, useful helper
from pyconturb.sig_models import iec_sig  # IEC 61400-1 turbulence std dev
from pyconturb.spectral_models import kaimal_spectrum  # Kaimal spectrum
from pyconturb.wind_profiles import constant_profile, power_profile  # wind-speed profile functions

from _nb_utils import plot_slice
import h5py

# copied from /Users/bstanisl/repos/pyconturb/pyconturb/duramat_constrained_turb.ipynb
# to run on hpc terminal in hopes that it's faster

# Step 1: generate dataframe of measurement data ----------------------------
# ---------------------------------------------------------------------------

# sonic position
x_sonic1 = 0.0
y_sonic1 = 0.0
z_sonic1 = 2.23 # m above ground

sonic_spat_df = gen_spat_grid(y_sonic1, z_sonic1)  # if `comps` not passed in, assumes all 3 components are wanted
print('sonic_spat_df = ', sonic_spat_df.head())  # look at the first few rows

# m2 tower data
m2 = {}
m2['u'] = [9.3218, 10.212, 11.211]
m2['z'] = [2, 5, 10]
m2['w_dir'] = [282.07, 275.67, 275.35]
m2['Iu'] = [0.1542, 0.149, 0.1455]

# def read_csv_data(sonic_data_fn, dt_sonic1):
def read_csv_data(raw_data):
    # read csv
    # raw_data = pd.read_csv(sonic_data_fn, header=None)

    # select only sonic data columns
    sonic_data = raw_data[['u (m/s)', 'v (m/s)', 'w (m/s)']]

    # downsample from 50 Hz to 4 Hz (sonic resolution)
    # sonic_data = sonic_data.resample('250ms').median() # 4 Hz = 250 milliseconds

    # construct time index
    tmp = (sonic_data.index[1]-sonic_data.index[0])
    dt_sonic1 = round(tmp.total_seconds(), 3)
    # print(dt_sonic1)
    tf_sonic1 = len(sonic_data) * dt_sonic1 # final time [s]
    # print(tf_sonic1)
    t_sonic1 = np.arange(0.0, tf_sonic1, dt_sonic1)

    sonic_data = sonic_data.rename(columns={'u (m/s)':'u_p0', 'v (m/s)':'v_p0', 'w (m/s)':'w_p0'})
    for col in sonic_data.filter(regex='u_', axis=1).columns:
        sonic_data[col] = -1.0*sonic_data[col] # to make it positive from the west
    sonic_data['index'] = t_sonic1
    sonic_data = sonic_data.set_index('index')

    return tf_sonic1, sonic_data

# sonic_data_fn = 'sonic1_halfsecond_20Hz.csv'
# sonic_data_fn = 'sonic1_5s_20Hz.csv'
# sonic_data_fn = 'sonic1_20s_20Hz.csv'
# sonic_data_fn = 'sonic1_10min_20Hz.csv'
# sonic_data_fn = 'sonic1_10min_50Hz.csv'
sonic_data_fn = 'DuraMAT_tilt40deg_turbulent_inflow_10min_timeseries.csv'

raw_data = pd.read_csv(sonic_data_fn, index_col='Time')
raw_data.index = pd.to_datetime(raw_data.index)
# raw_data

# tmp = (raw_data.index[1]-raw_data.index[0])
# dt_sonic1 = round(tmp.total_seconds(), 3)

tf_sonic1, sonic_data = read_csv_data(raw_data)
dt = sonic_data.index[1]-sonic_data.index[0]
print('dt = ', dt)
print('sonic_data = ', sonic_data.head())

# optional: cut down duration of signal for testing
tf_sonic1 = 30.0 # [s] - this is also how long the synthetic turbulence signal will be
sonic_data = sonic_data.loc[sonic_data.index <= tf_sonic1]

fig, axs = plt.subplots()
for col in sonic_data.columns:
    plt.plot(sonic_data[col], marker='.', label=col)  # subselect long. wind component
    # plt.plot(sonic_data[col].resample('50L').first(), marker='.', label=col)  # subselect long. wind component
axs.set_ylabel('velocity [m/s]');
plt.show()

sonic_df = pd.concat([sonic_spat_df, sonic_data], axis=0)

gen_csv_fname = 'generated_{}_{}s_{}Hz.csv'.format(sonic_data_fn[:34],int(tf_sonic1),int(1/dt))
sonic_df.to_csv(gen_csv_fname)
print(f'saved sonic data to csv {gen_csv_fname}')

# Step 2: read in csv of measurement data ----------------------------
# --------------------------------------------------------------------

con_tc = TimeConstraint(pd.read_csv(gen_csv_fname, index_col=0))  # load data from csv directly into tc
# con_tc = TimeConstraint(pd.read_csv('generated_sonic1_1s.csv', index_col=0))  # load data from csv directly into tc
con_tc.index = con_tc.index.map(lambda x: float(x) if (x not in 'kxyz') else x)  # index cleaning
con_tc.iloc[:7, :]  # look at the first 7 rows

time_df = con_tc.get_time()

# for var in ['u_','v_','w_']:
#     ax = time_df.filter(regex=var, axis=1).plot(lw=0.75)  # subselect long. wind component
#     ax.set_ylabel(var + ' [m/s]');

# [print(x) for x in time_df.filter(regex='u_', axis=1).mean()];  # print mean values
u_mean_sonic1 = time_df.filter(regex='u_', axis=1).mean()
print('u_mean_sonic1 = ', u_mean_sonic1)

# Step 3: generate constrained turbulence box ----------------------------
# ------------------------------------------------------------------------

# to match pvade sim
y_min = -30.0
y_max = 30.0
z_max = 20.0

ny = 300 # 130 #41
# nz = 60 #130 #20
# y = np.linspace(-10.0, 10.0, ny) #11) #22)  # 11 lateral points from -50 to 50 (center @ 0)
y = np.linspace(y_min, y_max, ny) #11) #22)  # 11 lateral points from -50 to 50 (center @ 0)
# linearly increasing grid size in z
steps = 0.15 + 0.02 * np.arange(60)
z = 0.0001 + np.cumsum(steps)
z = z[z <= z_max]
nz = len(z)

dy = y[1] - y[0] # (y[-1]-y[0])/ny
dz = z[1] - z[0] # (z[-1]-z[0])/nz

# this resolution should be approximately equal to or smaller than l_char
print(f'dy = {dy:.3f} m, dz = {dz:.5f} m')

spat_df = gen_spat_grid(y, z)  # if `comps` not passed in, assumes all 3 components are wanted
print(spat_df.head())  # look at the first few rows