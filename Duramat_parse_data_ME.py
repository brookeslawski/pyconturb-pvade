from scipy.io import loadmat

#### NOTES

# standardized files are here: \\nrel.gov\shared\Wind-data\Restricted\Projects\Duramat\Loads Analysis\data_FT_formatted
# standardize files binned by tracker angle here: \\nrel.gov\shared\Wind-data\Restricted\Projects\Duramat\Loads Analysis\data_filtered
# files that contain aggregated stats of all files: \\nrel.gov\shared\Wind-data\Restricted\Projects\Duramat\Loads Analysis\results

# dictionary contains a bunch of metadata, actual time series data can be found data[<variable name>]['processed']
# definition of variables can be found from the calibration and installation report here: \\nrel.gov\shared\Wind-data\Restricted\Projects\Duramat\Cal&InstallReport

# filepath
# fpath = 'Y:\Wind-data/Restricted/Projects/Duramat/Loads Analysis/data_FT_formatted/DuraMAT_Fast_2019_10_04_23_35_55_50Hz.mat'
# fpath = 'smb://nrel.gov/shared/Wind-data/Restricted/Projects/Duramat/Loads Analysis/data_FT_formatted/DuraMAT_Fast_2019_10_04_23_35_55_50Hz.mat'
fpath = '/Volumes/Duramat/Loads Analysis/data_FT_formatted/DuraMAT_Fast_2019_10_04_23_35_55_50Hz.mat'
# load in data as dictionary
out = loadmat(fpath, simplify_cells=True)
data = out['data']

print('done')

import numpy as np

def wind_components(speed, elevation, direction):
    """
    Converts wind speed, elevation, and direction to u, v, w components.

    Args:
        speed (float or array-like): Wind speed in m/s.
        elevation (float or array-like): Elevation angle in degrees.
        direction (float or array-like): Wind direction in degrees (meteorological convention, 
                                         where 0/360 is North, 90 is East, 180 is South, and 270 is West).

    Returns:
        tuple: A tuple containing u, v, and w components as numpy arrays.
               u: Eastward wind component (m/s)
               v: Northward wind component (m/s)
               w: Vertical wind component (m/s)
    """

    rad_elevation = np.radians(elevation)
    rad_direction = np.radians(direction)

    # Calculate u, v, w components
    u = speed * np.cos(rad_elevation) * np.sin(rad_direction)
    v = speed * np.cos(rad_elevation) * np.cos(rad_direction)
    w = speed * np.sin(rad_elevation)

    return u, v, w

#timestamp = data.get("LabVIEW_Timestamp")
#timestamp = list(timestamp.values())
#timestamp = timestamp[13]
#timestamp_int = np.around(timestamp).astype(int)


# Convert time zone to local (Dataframe)
#Denver_tz = pytz.timezone('America/Denver')
# Assuming 'index' of DataFrames is already in UTC
#data.index = data.index.tz_localize('UTC').tz_convert(Denver_tz)

Sonic_1_WS = data.get("Sonic_1_3DWindSpeed")
Sonic_1_WS = list(Sonic_1_WS.values())
Sonic_1_WS = Sonic_1_WS[0]
Sonic_2_WS = data.get("Sonic_2_3DWindSpeed")
Sonic_2_WS = list(Sonic_2_WS.values())
Sonic_2_WS = Sonic_2_WS[0]
Sonic_3_WS = data.get("Sonic_3_3DWindSpeed")
Sonic_3_WS = list(Sonic_3_WS.values())
Sonic_3_WS = Sonic_3_WS[0]

Sonic_1_WD = data.get("Sonic_1_WindDirection")
Sonic_1_WD = list(Sonic_1_WD.values())
Sonic_1_WD = Sonic_1_WD[0]
Sonic_2_WD = data.get("Sonic_2_WindDirection")
Sonic_2_WD = list(Sonic_2_WD.values())
Sonic_2_WD = Sonic_2_WD[0]
Sonic_3_WD = data.get("Sonic_3_WindDirection")
Sonic_3_WD = list(Sonic_3_WD.values())
Sonic_3_WD = Sonic_3_WD[0]

Sonic_1_Elevation = data.get("Sonic_1_Elevation")
Sonic_1_Elevation = list(Sonic_1_Elevation.values())
Sonic_1_Elevation = Sonic_1_Elevation[0]
Sonic_2_Elevation = data.get("Sonic_2_Elevation")
Sonic_2_Elevation = list(Sonic_2_Elevation.values())
Sonic_2_Elevation = Sonic_2_Elevation[0]
Sonic_3_Elevation = data.get("Sonic_3_Elevation")
Sonic_3_Elevation = list(Sonic_3_Elevation.values())
Sonic_3_Elevation = Sonic_3_Elevation[0]

Sonic_1_Temperature = data.get("Sonic_1_Temperature")
Sonic_1_Temperature = list(Sonic_1_Temperature.values())
Sonic_1_Temperature = Sonic_1_Temperature[0]

Sonic_1_u, Sonic_1_v, Sonic_1_w = wind_components(Sonic_1_WS, Sonic_1_Elevation, Sonic_1_WD)
Sonic_2_u, Sonic_2_v, Sonic_2_w = wind_components(Sonic_2_WS, Sonic_2_Elevation, Sonic_2_WD)
Sonic_3_u, Sonic_3_v, Sonic_3_w = wind_components(Sonic_3_WS, Sonic_3_Elevation, Sonic_3_WD)

Sonic_1_Umag = (Sonic_1_u**2+Sonic_1_v**2)**0.5
Sonic_2_Umag = (Sonic_2_u**2+Sonic_2_v**2)**0.5
Sonic_3_Umag = (Sonic_3_u**2+Sonic_3_v**2)**0.5

Sonic_1_Umagmean = Sonic_1_Umag.mean()
Sonic_2_Umagmean = Sonic_2_Umag.mean()
Sonic_3_Umagmean = Sonic_3_Umag.mean()

Sonic_1_WDmean = Sonic_1_WD.mean()
Sonic_2_WDmean = Sonic_2_WD.mean()
Sonic_3_WDmean = Sonic_3_WD.mean()

Sonic_1_u_std = np.std(Sonic_1_u)
Sonic_2_u_std = np.std(Sonic_2_u)
Sonic_3_u_std = np.std(Sonic_3_u)

Sonic_1_v_std = np.std(Sonic_1_v)
Sonic_2_v_std = np.std(Sonic_2_v)
Sonic_3_v_std = np.std(Sonic_3_v)

Sonic_1_w_std = np.std(Sonic_1_w)
Sonic_2_w_std = np.std(Sonic_2_w)
Sonic_3_w_std = np.std(Sonic_3_w)

Sonic_1_Iu = Sonic_1_u_std/Sonic_1_Umagmean
Sonic_2_Iu = Sonic_2_u_std/Sonic_2_Umagmean
Sonic_3_Iu = Sonic_3_u_std/Sonic_3_Umagmean

Sonic_1_Iv = Sonic_1_v_std/Sonic_1_Umagmean
Sonic_2_Iv = Sonic_2_v_std/Sonic_2_Umagmean
Sonic_3_Iv = Sonic_3_v_std/Sonic_3_Umagmean

Sonic_1_Iw = Sonic_1_w_std/Sonic_1_Umagmean
Sonic_2_Iw = Sonic_2_w_std/Sonic_2_Umagmean
Sonic_3_Iw = Sonic_3_w_std/Sonic_3_Umagmean


# Array downsampling

import pandas as pd
import numpy as np
from scipy.signal import resample

# Original sampling rate
original_fs = 50  # Hz

# Desired sampling rate
new_fs = 20  # Hz

# Time vector
duration = 600  # seconds
num_samples = original_fs * duration
time = np.linspace(0, duration, num_samples, endpoint=False)

# Calculate the new number of samples
new_num_samples = int(new_fs * duration)

# New time vector for the resampled signals
new_time = np.linspace(0, duration, new_num_samples, endpoint=False)

# Sonic heights and resampled frequency
heights = [2.23,2.23,2.26] 
fs = 20

# Resample signals
Sonic_1_u_20Hz = resample(Sonic_1_u, new_num_samples)
Sonic_2_u_20Hz = resample(Sonic_2_u, new_num_samples)
Sonic_3_u_20Hz = resample(Sonic_3_u, new_num_samples)

Sonic_1_v_20Hz = resample(Sonic_1_v, new_num_samples)
Sonic_2_v_20Hz = resample(Sonic_2_v, new_num_samples)
Sonic_3_v_20Hz = resample(Sonic_3_v, new_num_samples)

Sonic_1_w_20Hz = resample(Sonic_1_w, new_num_samples)
Sonic_2_w_20Hz = resample(Sonic_2_w, new_num_samples)
Sonic_3_w_20Hz = resample(Sonic_3_w, new_num_samples)

Sonic_1_Temperature_20Hz = resample(Sonic_1_Temperature, new_num_samples)

Sonic_1_WD_20Hz = resample(Sonic_1_WD, new_num_samples)
Sonic_2_WD_20Hz = resample(Sonic_2_WD, new_num_samples)
Sonic_3_WD_20Hz = resample(Sonic_3_WD, new_num_samples)

Sonic_1_Umag_20Hz = (Sonic_1_u_20Hz**2+Sonic_1_v_20Hz**2)**0.5
Sonic_2_Umag_20Hz = (Sonic_2_u_20Hz**2+Sonic_2_v_20Hz**2)**0.5
Sonic_3_Umag_20Hz = (Sonic_3_u_20Hz**2+Sonic_3_v_20Hz**2)**0.5

Sonic_1_Tmean_20Hz = Sonic_1_Temperature_20Hz.mean()

Sonic_1_Umean_20Hz = Sonic_1_u_20Hz.mean()
Sonic_2_Umean_20Hz = Sonic_2_u_20Hz.mean()
Sonic_3_Umean_20Hz = Sonic_3_u_20Hz.mean()

Sonic_1_Umagmean_20Hz = Sonic_1_Umag_20Hz.mean()
Sonic_2_Umagmean_20Hz = Sonic_2_Umag_20Hz.mean()
Sonic_3_Umagmean_20Hz = Sonic_3_Umag_20Hz.mean()

Sonic_1_WDmean_20Hz = Sonic_1_WD_20Hz.mean()
Sonic_2_WDmean_20Hz = Sonic_2_WD_20Hz.mean()
Sonic_3_WDmean_20Hz = Sonic_3_WD_20Hz.mean()

Sonic_1_u_std_20Hz = np.std(Sonic_1_u_20Hz)
Sonic_2_u_std_20Hz = np.std(Sonic_2_u_20Hz)
Sonic_3_u_std_20Hz = np.std(Sonic_3_u_20Hz)

Sonic_1_v_std_20Hz = np.std(Sonic_1_v_20Hz)
Sonic_2_v_std_20Hz = np.std(Sonic_2_v_20Hz)
Sonic_3_v_std_20Hz = np.std(Sonic_3_v_20Hz)

Sonic_1_w_std_20Hz = np.std(Sonic_1_w_20Hz)
Sonic_2_w_std_20Hz = np.std(Sonic_2_w_20Hz)
Sonic_3_w_std_20Hz = np.std(Sonic_3_w_20Hz)

Sonic_1_Iu_20Hz = Sonic_1_u_std_20Hz/Sonic_1_Umagmean_20Hz
Sonic_2_Iu_20Hz = Sonic_2_u_std_20Hz/Sonic_2_Umagmean_20Hz
Sonic_3_Iu_20Hz = Sonic_3_u_std_20Hz/Sonic_3_Umagmean_20Hz

Sonic_1_Iv_20Hz = Sonic_1_v_std_20Hz/Sonic_1_Umagmean_20Hz
Sonic_2_Iv_20Hz = Sonic_2_v_std_20Hz/Sonic_2_Umagmean_20Hz
Sonic_3_Iv_20Hz = Sonic_3_v_std_20Hz/Sonic_3_Umagmean_20Hz

Sonic_1_Iw_20Hz = Sonic_1_w_std_20Hz/Sonic_1_Umagmean_20Hz
Sonic_2_Iw_20Hz = Sonic_2_w_std_20Hz/Sonic_2_Umagmean_20Hz
Sonic_3_Iw_20Hz = Sonic_3_w_std_20Hz/Sonic_3_Umagmean_20Hz


# Plot signals
    
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.figure()
plt.title("u time series (50Hz)")  
plt.plot(Sonic_1_u,label='Sonic_1')
plt.plot(Sonic_2_u,label='Sonic_2')
plt.plot(Sonic_3_u,label='Sonic_3')
plt.legend()

plt.figure()
plt.title("wind direction time series (50Hz)")  
plt.plot(Sonic_1_WD,label='Sonic_1')
plt.plot(Sonic_2_WD,label='Sonic_2')
plt.plot(Sonic_3_WD,label='Sonic_3')
plt.legend(fontsize=8)

plt.figure()
plt.title("u time series (20Hz)")  
plt.plot(Sonic_1_u_20Hz,label='Sonic_1')
# plt.plot(Sonic_2_u_20Hz,label='Sonic_2')
# plt.plot(Sonic_3_u_20Hz,label='Sonic_3')
plt.legend(fontsize=8)

plt.figure()
plt.title("v time series (20Hz)")  
plt.plot(Sonic_1_v_20Hz,label='Sonic_1')
# plt.plot(Sonic_2_v_20Hz,label='Sonic_2')
# plt.plot(Sonic_3_v_20Hz,label='Sonic_3')
plt.legend(fontsize=8)

plt.figure()
plt.title("w time series (20Hz)")  
plt.plot(Sonic_1_w_20Hz,label='Sonic_1')
# plt.plot(Sonic_2_w_20Hz,label='Sonic_2')
# plt.plot(Sonic_3_w_20Hz,label='Sonic_3')
plt.legend(fontsize=8)

plt.figure()
plt.title("wind direction time series (20Hz)")  
plt.plot(Sonic_1_WD_20Hz,label='Sonic_1')
plt.plot(Sonic_2_WD_20Hz,label='Sonic_2')
plt.plot(Sonic_3_WD_20Hz,label='Sonic_3')
plt.legend(fontsize=8)


plt.figure()
plt.title("Sonic 1 vel time series (20Hz)")  
plt.plot(Sonic_1_u_20Hz,label='u')
plt.plot(Sonic_1_v_20Hz,label='v')
plt.plot(Sonic_1_w_20Hz,label='w')
plt.legend(fontsize=8)



import scipy

# Extract data by sonic in a series
U_corr_inflow_west_10042019_2325_2335_Sonic_1 = pd.Series(Sonic_1_u_20Hz)
U_corr_inflow_west_10042019_2325_2335_Sonic_2 = pd.Series(Sonic_2_u_20Hz)
U_corr_inflow_west_10042019_2325_2335_Sonic_3 = pd.Series(Sonic_3_u_20Hz)

V_corr_inflow_west_10042019_2325_2335_Sonic_1 = pd.Series(Sonic_1_v_20Hz)
V_corr_inflow_west_10042019_2325_2335_Sonic_2 = pd.Series(Sonic_2_v_20Hz)
V_corr_inflow_west_10042019_2325_2335_Sonic_3 = pd.Series(Sonic_3_v_20Hz)

W_corr_inflow_west_10042019_2325_2335_Sonic_1 = pd.Series(Sonic_1_w_20Hz)
W_corr_inflow_west_10042019_2325_2335_Sonic_2 = pd.Series(Sonic_2_w_20Hz)
W_corr_inflow_west_10042019_2325_2335_Sonic_3 = pd.Series(Sonic_3_w_20Hz)

T_corr_inflow_west_10042019_2325_2335_Sonic_1 = pd.Series(Sonic_1_Temperature_20Hz)

# Detrend
U_corr_inflow_west_10042019_2325_2335_Sonic_1[U_corr_inflow_west_10042019_2325_2335_Sonic_1.isna()==False] = scipy.signal.detrend(U_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()) 
U_corr_inflow_west_10042019_2325_2335_Sonic_2[U_corr_inflow_west_10042019_2325_2335_Sonic_2.isna()==False] = scipy.signal.detrend(U_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()) 
U_corr_inflow_west_10042019_2325_2335_Sonic_3[U_corr_inflow_west_10042019_2325_2335_Sonic_3.isna()==False] = scipy.signal.detrend(U_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()) 

V_corr_inflow_west_10042019_2325_2335_Sonic_1[V_corr_inflow_west_10042019_2325_2335_Sonic_1.isna()==False] = scipy.signal.detrend(V_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()) 
V_corr_inflow_west_10042019_2325_2335_Sonic_2[V_corr_inflow_west_10042019_2325_2335_Sonic_2.isna()==False] = scipy.signal.detrend(V_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()) 
V_corr_inflow_west_10042019_2325_2335_Sonic_3[V_corr_inflow_west_10042019_2325_2335_Sonic_3.isna()==False] = scipy.signal.detrend(V_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()) 

W_corr_inflow_west_10042019_2325_2335_Sonic_1[W_corr_inflow_west_10042019_2325_2335_Sonic_1.isna()==False] = scipy.signal.detrend(W_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()) 
W_corr_inflow_west_10042019_2325_2335_Sonic_2[W_corr_inflow_west_10042019_2325_2335_Sonic_2.isna()==False] = scipy.signal.detrend(W_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()) 
W_corr_inflow_west_10042019_2325_2335_Sonic_3[W_corr_inflow_west_10042019_2325_2335_Sonic_3.isna()==False] = scipy.signal.detrend(W_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()) 

T_corr_inflow_west_10042019_2325_2335_Sonic_1[T_corr_inflow_west_10042019_2325_2335_Sonic_1.isna()==False] = scipy.signal.detrend(T_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()) 

# Reynolds stresses and length scales (north)

inflow_uv_Sonic_1_west_10042019_2325_2335 = (U_corr_inflow_west_10042019_2325_2335_Sonic_1*V_corr_inflow_west_10042019_2325_2335_Sonic_1).mean()-(U_corr_inflow_west_10042019_2325_2335_Sonic_1.mean()*V_corr_inflow_west_10042019_2325_2335_Sonic_1.mean());
inflow_vw_Sonic_1_west_10042019_2325_2335 = (V_corr_inflow_west_10042019_2325_2335_Sonic_1*W_corr_inflow_west_10042019_2325_2335_Sonic_1).mean()-(V_corr_inflow_west_10042019_2325_2335_Sonic_1.mean()*W_corr_inflow_west_10042019_2325_2335_Sonic_1.mean());
inflow_uw_Sonic_1_west_10042019_2325_2335 = (U_corr_inflow_west_10042019_2325_2335_Sonic_1*W_corr_inflow_west_10042019_2325_2335_Sonic_1).mean()-(U_corr_inflow_west_10042019_2325_2335_Sonic_1.mean()*W_corr_inflow_west_10042019_2325_2335_Sonic_1.mean());
inflow_wT_Sonic_1_west_10042019_2325_2335 = (W_corr_inflow_west_10042019_2325_2335_Sonic_1*T_corr_inflow_west_10042019_2325_2335_Sonic_1).mean()-(W_corr_inflow_west_10042019_2325_2335_Sonic_1.mean()*T_corr_inflow_west_10042019_2325_2335_Sonic_1.mean());

inflow_uv_Sonic_2_west_10042019_2325_2335 = (U_corr_inflow_west_10042019_2325_2335_Sonic_2*V_corr_inflow_west_10042019_2325_2335_Sonic_2).mean()-(U_corr_inflow_west_10042019_2325_2335_Sonic_2.mean()*V_corr_inflow_west_10042019_2325_2335_Sonic_2.mean());
inflow_vw_Sonic_2_west_10042019_2325_2335 = (V_corr_inflow_west_10042019_2325_2335_Sonic_2*W_corr_inflow_west_10042019_2325_2335_Sonic_2).mean()-(V_corr_inflow_west_10042019_2325_2335_Sonic_2.mean()*W_corr_inflow_west_10042019_2325_2335_Sonic_2.mean());
inflow_uw_Sonic_2_west_10042019_2325_2335 = (U_corr_inflow_west_10042019_2325_2335_Sonic_2*W_corr_inflow_west_10042019_2325_2335_Sonic_2).mean()-(U_corr_inflow_west_10042019_2325_2335_Sonic_2.mean()*W_corr_inflow_west_10042019_2325_2335_Sonic_2.mean());

inflow_uv_Sonic_3_west_10042019_2325_2335 = (U_corr_inflow_west_10042019_2325_2335_Sonic_3*V_corr_inflow_west_10042019_2325_2335_Sonic_3).mean()-(U_corr_inflow_west_10042019_2325_2335_Sonic_3.mean()*V_corr_inflow_west_10042019_2325_2335_Sonic_3.mean());
inflow_vw_Sonic_3_west_10042019_2325_2335 = (V_corr_inflow_west_10042019_2325_2335_Sonic_3*W_corr_inflow_west_10042019_2325_2335_Sonic_3).mean()-(V_corr_inflow_west_10042019_2325_2335_Sonic_3.mean()*W_corr_inflow_west_10042019_2325_2335_Sonic_3.mean());
inflow_uw_Sonic_3_west_10042019_2325_2335 = (U_corr_inflow_west_10042019_2325_2335_Sonic_3*W_corr_inflow_west_10042019_2325_2335_Sonic_3).mean()-(U_corr_inflow_west_10042019_2325_2335_Sonic_3.mean()*W_corr_inflow_west_10042019_2325_2335_Sonic_3.mean());

utau_Sonic_1_west_10042019_2325_2335 = (inflow_uw_Sonic_1_west_10042019_2325_2335**2+inflow_vw_Sonic_1_west_10042019_2325_2335**2)**(1/4) 
utau_Sonic_2_west_10042019_2325_2335 = (inflow_uw_Sonic_2_west_10042019_2325_2335**2+inflow_vw_Sonic_2_west_10042019_2325_2335**2)**(1/4) 
utau_Sonic_3_west_10042019_2325_2335 = (inflow_uw_Sonic_3_west_10042019_2325_2335**2+inflow_vw_Sonic_3_west_10042019_2325_2335**2)**(1/4) 

L_Sonic_1_west_10042019_2325_2335 = -1*(utau_Sonic_1_west_10042019_2325_2335**3)/(0.4*(9.81/Sonic_1_Tmean_20Hz)*inflow_wT_Sonic_1_west_10042019_2325_2335)
zL_Sonic_1_west_10042019_2325_2335 = heights[0]/L_Sonic_1_west_10042019_2325_2335
inflow_uprimewprime_Sonic_1_west_10042019_2325_2335 = (U_corr_inflow_west_10042019_2325_2335_Sonic_1*W_corr_inflow_west_10042019_2325_2335_Sonic_1);



#%% PSD analysis


heights = [2.23,2.23,2.26] 
fs = 20

# Spectra
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from numpy import hanning
import math

overlap = 0
nblock = len(U_corr_inflow_west_10042019_2325_2335_Sonic_1)
win = np.hamming(math.floor(nblock/10))

f_U_corr_inflow_west_10042019_2325_2335_Sonic_1, Pxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_1 = welch(U_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), fs, window=win, noverlap=overlap, nfft=nblock, detrend='constant', return_onesided=True)
nf_U_corr_inflow_west_10042019_2325_2335_Sonic_1 = f_U_corr_inflow_west_10042019_2325_2335_Sonic_1*heights[0]/abs(Sonic_1_Umean_20Hz)
nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_1 = (f_U_corr_inflow_west_10042019_2325_2335_Sonic_1*Pxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_1)/Sonic_1_u_std_20Hz**2

f_U_corr_inflow_west_10042019_2325_2335_Sonic_2, Pxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_2 = welch(U_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), fs, window=win, noverlap=overlap, nfft=nblock, detrend='constant', return_onesided=True)
nf_U_corr_inflow_west_10042019_2325_2335_Sonic_2 = f_U_corr_inflow_west_10042019_2325_2335_Sonic_2*heights[1]/abs(Sonic_2_Umean_20Hz)
nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_2 = (f_U_corr_inflow_west_10042019_2325_2335_Sonic_2*Pxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_2)/Sonic_2_u_std_20Hz**2
 
f_U_corr_inflow_west_10042019_2325_2335_Sonic_3, Pxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_3 = welch(U_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), fs, window=win, noverlap=overlap, nfft=nblock, detrend='constant', return_onesided=True)
nf_U_corr_inflow_west_10042019_2325_2335_Sonic_3 = f_U_corr_inflow_west_10042019_2325_2335_Sonic_3*heights[2]/abs(Sonic_3_Umean_20Hz)
nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_3 = (f_U_corr_inflow_west_10042019_2325_2335_Sonic_3*Pxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_3)/Sonic_3_u_std_20Hz**2              
      
f_V_corr_inflow_west_10042019_2325_2335_Sonic_1, Pxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_1 = welch(V_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), fs, window=win, noverlap=overlap, nfft=nblock, detrend='constant', return_onesided=True)
nf_V_corr_inflow_west_10042019_2325_2335_Sonic_1 = f_V_corr_inflow_west_10042019_2325_2335_Sonic_1*heights[0]/abs(Sonic_1_Umean_20Hz)
nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_1 = (f_V_corr_inflow_west_10042019_2325_2335_Sonic_1*Pxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_1)/Sonic_1_v_std_20Hz**2

f_V_corr_inflow_west_10042019_2325_2335_Sonic_2, Pxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_2 = welch(V_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), fs, window=win, noverlap=overlap, nfft=nblock, detrend='constant', return_onesided=True)
nf_V_corr_inflow_west_10042019_2325_2335_Sonic_2 = f_V_corr_inflow_west_10042019_2325_2335_Sonic_2*heights[1]/abs(Sonic_2_Umean_20Hz)
nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_2 = (f_V_corr_inflow_west_10042019_2325_2335_Sonic_2*Pxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_2)/Sonic_2_v_std_20Hz**2
 
f_V_corr_inflow_west_10042019_2325_2335_Sonic_3, Pxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_3 = welch(V_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), fs, window=win, noverlap=overlap, nfft=nblock, detrend='constant', return_onesided=True)
nf_V_corr_inflow_west_10042019_2325_2335_Sonic_3 = f_V_corr_inflow_west_10042019_2325_2335_Sonic_3*heights[2]/abs(Sonic_3_Umean_20Hz)
nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_3 = (f_V_corr_inflow_west_10042019_2325_2335_Sonic_3*Pxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_3)/Sonic_3_v_std_20Hz**2              
      
f_W_corr_inflow_west_10042019_2325_2335_Sonic_1, Pxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_1 = welch(W_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), fs, window=win, noverlap=overlap, nfft=nblock, detrend='constant', return_onesided=True)
nf_W_corr_inflow_west_10042019_2325_2335_Sonic_1 = f_W_corr_inflow_west_10042019_2325_2335_Sonic_1*heights[0]/abs(Sonic_1_Umean_20Hz)
nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_1 = (f_W_corr_inflow_west_10042019_2325_2335_Sonic_1*Pxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_1)/Sonic_1_w_std_20Hz**2

f_W_corr_inflow_west_10042019_2325_2335_Sonic_2, Pxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_2 = welch(W_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), fs, window=win, noverlap=overlap, nfft=nblock, detrend='constant', return_onesided=True)
nf_W_corr_inflow_west_10042019_2325_2335_Sonic_2 = f_W_corr_inflow_west_10042019_2325_2335_Sonic_2*heights[1]/abs(Sonic_2_Umean_20Hz)
nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_2 = (f_W_corr_inflow_west_10042019_2325_2335_Sonic_2*Pxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_2)/Sonic_2_w_std_20Hz**2
 
f_W_corr_inflow_west_10042019_2325_2335_Sonic_3, Pxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_3 = welch(W_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), fs, window=win, noverlap=overlap, nfft=nblock, detrend='constant', return_onesided=True)
nf_W_corr_inflow_west_10042019_2325_2335_Sonic_3 = f_W_corr_inflow_west_10042019_2325_2335_Sonic_3*heights[2]/abs(Sonic_3_Umean_20Hz)
nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_3 = (f_W_corr_inflow_west_10042019_2325_2335_Sonic_3*Pxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_3)/Sonic_3_w_std_20Hz**2              
      

# Smooth high frequency region
def runningMeanFast(x, N):
    """
    Calculates the running mean of an array x over a window size N.
    """
    return np.convolve(x, np.ones(N)/N, mode='same') 

index_highfreq_U_corr_inflow_west_10042019_2325_2335_Sonic_1 = list(np.where([abs(nf_U_corr_inflow_west_10042019_2325_2335_Sonic_1)>0.3]))
nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_1 = nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_1[index_highfreq_U_corr_inflow_west_10042019_2325_2335_Sonic_1[0][0]:len(nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_1)]
nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_1 = runningMeanFast(nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_1,200)
nPxxf_mod_U_corr_inflow_west_10042019_2325_2335_Sonic_1 = [nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_1[0:index_highfreq_U_corr_inflow_west_10042019_2325_2335_Sonic_1[0][0]-1],nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_1]

index_highfreq_U_corr_inflow_west_10042019_2325_2335_Sonic_2 = list(np.where([abs(nf_U_corr_inflow_west_10042019_2325_2335_Sonic_2)>0.3]))
nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_2 = nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_2[index_highfreq_U_corr_inflow_west_10042019_2325_2335_Sonic_2[0][0]:len(nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_2)]
nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_2 = runningMeanFast(nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_2,200)
nPxxf_mod_U_corr_inflow_west_10042019_2325_2335_Sonic_2 = [nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_2[0:index_highfreq_U_corr_inflow_west_10042019_2325_2335_Sonic_2[0][0]-1],nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_2]

index_highfreq_U_corr_inflow_west_10042019_2325_2335_Sonic_3 = list(np.where([abs(nf_U_corr_inflow_west_10042019_2325_2335_Sonic_3)>0.3]))
nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_3 = nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_3[index_highfreq_U_corr_inflow_west_10042019_2325_2335_Sonic_3[0][0]:len(nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_3)]
nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_3 = runningMeanFast(nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_3,200)
nPxxf_mod_U_corr_inflow_west_10042019_2325_2335_Sonic_3 = [nPxxf_U_corr_inflow_west_10042019_2325_2335_Sonic_3[0:index_highfreq_U_corr_inflow_west_10042019_2325_2335_Sonic_3[0][0]-1],nPxxf_smooth_U_corr_inflow_west_10042019_2325_2335_Sonic_3]

index_highfreq_V_corr_inflow_west_10042019_2325_2335_Sonic_1 = list(np.where([abs(nf_V_corr_inflow_west_10042019_2325_2335_Sonic_1)>0.3]))
nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_1 = nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_1[index_highfreq_V_corr_inflow_west_10042019_2325_2335_Sonic_1[0][0]:len(nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_1)]
nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_1 = runningMeanFast(nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_1,200)
nPxxf_mod_V_corr_inflow_west_10042019_2325_2335_Sonic_1 = [nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_1[0:index_highfreq_V_corr_inflow_west_10042019_2325_2335_Sonic_1[0][0]-1],nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_1]

index_highfreq_V_corr_inflow_west_10042019_2325_2335_Sonic_2 = list(np.where([abs(nf_V_corr_inflow_west_10042019_2325_2335_Sonic_2)>0.3]))
nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_2 = nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_2[index_highfreq_V_corr_inflow_west_10042019_2325_2335_Sonic_2[0][0]:len(nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_2)]
nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_2 = runningMeanFast(nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_2,200)
nPxxf_mod_V_corr_inflow_west_10042019_2325_2335_Sonic_2 = [nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_2[0:index_highfreq_V_corr_inflow_west_10042019_2325_2335_Sonic_2[0][0]-1],nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_2]

index_highfreq_V_corr_inflow_west_10042019_2325_2335_Sonic_3 = list(np.where([abs(nf_V_corr_inflow_west_10042019_2325_2335_Sonic_3)>0.3]))
nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_3 = nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_3[index_highfreq_V_corr_inflow_west_10042019_2325_2335_Sonic_3[0][0]:len(nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_3)]
nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_3 = runningMeanFast(nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_3,200)
nPxxf_mod_V_corr_inflow_west_10042019_2325_2335_Sonic_3 = [nPxxf_V_corr_inflow_west_10042019_2325_2335_Sonic_3[0:index_highfreq_V_corr_inflow_west_10042019_2325_2335_Sonic_3[0][0]-1],nPxxf_smooth_V_corr_inflow_west_10042019_2325_2335_Sonic_3]

index_highfreq_W_corr_inflow_west_10042019_2325_2335_Sonic_1 = list(np.where([abs(nf_W_corr_inflow_west_10042019_2325_2335_Sonic_1)>0.3]))
nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_1 = nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_1[index_highfreq_W_corr_inflow_west_10042019_2325_2335_Sonic_1[0][0]:len(nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_1)]
nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_1 = runningMeanFast(nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_1,200)
nPxxf_mod_W_corr_inflow_west_10042019_2325_2335_Sonic_1 = [nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_1[0:index_highfreq_W_corr_inflow_west_10042019_2325_2335_Sonic_1[0][0]-1],nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_1]

index_highfreq_W_corr_inflow_west_10042019_2325_2335_Sonic_2 = list(np.where([abs(nf_W_corr_inflow_west_10042019_2325_2335_Sonic_2)>0.3]))
nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_2 = nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_2[index_highfreq_W_corr_inflow_west_10042019_2325_2335_Sonic_2[0][0]:len(nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_2)]
nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_2 = runningMeanFast(nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_2,200)
nPxxf_mod_W_corr_inflow_west_10042019_2325_2335_Sonic_2 = [nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_2[0:index_highfreq_W_corr_inflow_west_10042019_2325_2335_Sonic_2[0][0]-1],nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_2]

index_highfreq_W_corr_inflow_west_10042019_2325_2335_Sonic_3 = list(np.where([abs(nf_W_corr_inflow_west_10042019_2325_2335_Sonic_3)>0.3]))
nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_3 = nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_3[index_highfreq_W_corr_inflow_west_10042019_2325_2335_Sonic_3[0][0]:len(nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_3)]
nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_3 = runningMeanFast(nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_3,200)
nPxxf_mod_W_corr_inflow_west_10042019_2325_2335_Sonic_3 = [nPxxf_W_corr_inflow_west_10042019_2325_2335_Sonic_3[0:index_highfreq_W_corr_inflow_west_10042019_2325_2335_Sonic_3[0][0]-1],nPxxf_smooth_W_corr_inflow_west_10042019_2325_2335_Sonic_3]


plt.figure()
plt.subplot(1,2,1)
plt.loglog(abs(nf_U_corr_inflow_west_10042019_2325_2335_Sonic_1[0:len(nPxxf_mod_U_corr_inflow_west_10042019_2325_2335_Sonic_1[1])]), nPxxf_mod_U_corr_inflow_west_10042019_2325_2335_Sonic_1[1], label='Sonic_1')            
plt.loglog(abs(nf_U_corr_inflow_west_10042019_2325_2335_Sonic_2[0:len(nPxxf_mod_U_corr_inflow_west_10042019_2325_2335_Sonic_2[1])]), nPxxf_mod_U_corr_inflow_west_10042019_2325_2335_Sonic_2[1], label='Sonic_2')            
plt.loglog(abs(nf_U_corr_inflow_west_10042019_2325_2335_Sonic_3[0:len(nPxxf_mod_U_corr_inflow_west_10042019_2325_2335_Sonic_3[1])]), nPxxf_mod_U_corr_inflow_west_10042019_2325_2335_Sonic_3[1], label='Sonic_3')            
plt.legend(loc='lower left',fontsize=8)
plt.xlabel("$fz/U$")
plt.ylabel("$fS_u/\sigma_u$")
plt.title('inflow (west) streamwise u spectra')
plt.xlim(10e-3, 10e-1)
plt.ylim(10e-3, 10e-1)
plt.show()
    
plt.subplot(1,2,1)
plt.loglog(abs(nf_V_corr_inflow_west_10042019_2325_2335_Sonic_1[0:len(nPxxf_mod_V_corr_inflow_west_10042019_2325_2335_Sonic_1[1])]), nPxxf_mod_V_corr_inflow_west_10042019_2325_2335_Sonic_1[1], label='Sonic_1')            
plt.loglog(abs(nf_V_corr_inflow_west_10042019_2325_2335_Sonic_2[0:len(nPxxf_mod_V_corr_inflow_west_10042019_2325_2335_Sonic_2[1])]), nPxxf_mod_V_corr_inflow_west_10042019_2325_2335_Sonic_2[1], label='Sonic_2')            
plt.loglog(abs(nf_V_corr_inflow_west_10042019_2325_2335_Sonic_3[0:len(nPxxf_mod_V_corr_inflow_west_10042019_2325_2335_Sonic_3[1])]), nPxxf_mod_V_corr_inflow_west_10042019_2325_2335_Sonic_3[1], label='Sonic_3')            
plt.legend(loc='lower left',fontsize=8)
plt.xlabel("$fz/U$")
plt.ylabel("$fS_v/\sigma_v$")
plt.title('inflow (west) spanwise v spectra')
plt.xlim(10e-3, 10e-1)
plt.ylim(10e-3, 10e-1)
plt.show()

plt.subplot(1,2,1)
plt.loglog(abs(nf_W_corr_inflow_west_10042019_2325_2335_Sonic_1[0:len(nPxxf_mod_W_corr_inflow_west_10042019_2325_2335_Sonic_1[1])]), nPxxf_mod_W_corr_inflow_west_10042019_2325_2335_Sonic_1[1], label='Sonic_1')            
plt.loglog(abs(nf_W_corr_inflow_west_10042019_2325_2335_Sonic_2[0:len(nPxxf_mod_W_corr_inflow_west_10042019_2325_2335_Sonic_2[1])]), nPxxf_mod_W_corr_inflow_west_10042019_2325_2335_Sonic_2[1], label='Sonic_2')            
plt.loglog(abs(nf_W_corr_inflow_west_10042019_2325_2335_Sonic_3[0:len(nPxxf_mod_W_corr_inflow_west_10042019_2325_2335_Sonic_3[1])]), nPxxf_mod_W_corr_inflow_west_10042019_2325_2335_Sonic_3[1], label='Sonic_3')            
plt.legend(loc='lower left',fontsize=8)
plt.xlabel("$fz/U$")
plt.ylabel("$fS_w/\sigma_w$")
plt.title('inflow (west) vertical w spectra')
plt.xlim(10e-3, 10e-1)
plt.ylim(10e-3, 10e-1)
plt.show()



#%% LS exponential fit method

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

autocorr_inflow_west_10042019_2325_2335 = np.correlate(U_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), U_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), mode='full') 
autocorr_inflow_west_10042019_2325_2335 /= np.sqrt(np.dot(U_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), U_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()) * np.dot(U_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), U_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()))  # Normalize the result
lags = np.arange(-len(U_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()) + 1, len(U_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()))
Y = (lags, autocorr_inflow_west_10042019_2325_2335)
Lux_west_10042019_2325_2335_Sonic_1 = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))]*(1/fs)*abs(Sonic_1_Umean_20Hz)
Lux_west_10042019_2325_2335_Sonic_1 = Lux_west_10042019_2325_2335_Sonic_1[Lux_west_10042019_2325_2335_Sonic_1>0]

autocorr_inflow_west_10042019_2325_2335 = np.correlate(U_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), U_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), mode='full') 
autocorr_inflow_west_10042019_2325_2335 /= np.sqrt(np.dot(U_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), U_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()) * np.dot(U_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), U_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()))  # Normalize the result
lags = np.arange(-len(U_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()) + 1, len(U_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()))
Y = (lags, autocorr_inflow_west_10042019_2325_2335)
Lux_west_10042019_2325_2335_Sonic_2 = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))]*(1/fs)*abs(Sonic_2_Umean_20Hz)
Lux_west_10042019_2325_2335_Sonic_2 = Lux_west_10042019_2325_2335_Sonic_2[Lux_west_10042019_2325_2335_Sonic_2>0]

autocorr_inflow_west_10042019_2325_2335 = np.correlate(U_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), U_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), mode='full') 
autocorr_inflow_west_10042019_2325_2335 /= np.sqrt(np.dot(U_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), U_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()) * np.dot(U_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), U_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()))  # Normalize the result
lags = np.arange(-len(U_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()) + 1, len(U_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()))
Y = (lags, autocorr_inflow_west_10042019_2325_2335)
Lux_west_10042019_2325_2335_Sonic_3 = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))]*(1/fs)*abs(Sonic_3_Umean_20Hz)
Lux_west_10042019_2325_2335_Sonic_3 = Lux_west_10042019_2325_2335_Sonic_3[Lux_west_10042019_2325_2335_Sonic_3>0]

autocorr_inflow_west_10042019_2325_2335 = np.correlate(V_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), V_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), mode='full') 
autocorr_inflow_west_10042019_2325_2335 /= np.sqrt(np.dot(V_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), V_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()) * np.dot(V_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), V_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()))  # Normalize the result
lags = np.arange(-len(V_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()) + 1, len(V_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()))
Y = (lags, autocorr_inflow_west_10042019_2325_2335)
Lvx_west_10042019_2325_2335_Sonic_1 = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))]*(1/fs)*abs(Sonic_1_Umean_20Hz)
Lvx_west_10042019_2325_2335_Sonic_1 = Lvx_west_10042019_2325_2335_Sonic_1[Lvx_west_10042019_2325_2335_Sonic_1>0]

autocorr_inflow_west_10042019_2325_2335 = np.correlate(V_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), V_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), mode='full') 
autocorr_inflow_west_10042019_2325_2335 /= np.sqrt(np.dot(V_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), V_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()) * np.dot(V_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), V_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()))  # Normalize the result
lags = np.arange(-len(V_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()) + 1, len(V_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()))
Y = (lags, autocorr_inflow_west_10042019_2325_2335)
Lvx_west_10042019_2325_2335_Sonic_2 = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))]*(1/fs)*abs(Sonic_2_Umean_20Hz)
Lvx_west_10042019_2325_2335_Sonic_2 = Lvx_west_10042019_2325_2335_Sonic_2[Lvx_west_10042019_2325_2335_Sonic_2>0]

autocorr_inflow_west_10042019_2325_2335 = np.correlate(V_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), V_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), mode='full') 
autocorr_inflow_west_10042019_2325_2335 /= np.sqrt(np.dot(V_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), V_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()) * np.dot(V_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), V_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()))  # Normalize the result
lags = np.arange(-len(V_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()) + 1, len(V_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()))
Y = (lags, autocorr_inflow_west_10042019_2325_2335)
Lvx_west_10042019_2325_2335_Sonic_3 = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))]*(1/fs)*abs(Sonic_3_Umean_20Hz)
Lvx_west_10042019_2325_2335_Sonic_3 = Lvx_west_10042019_2325_2335_Sonic_3[Lvx_west_10042019_2325_2335_Sonic_3>0]

autocorr_inflow_west_10042019_2325_2335 = np.correlate(W_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), W_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), mode='full') 
autocorr_inflow_west_10042019_2325_2335 /= np.sqrt(np.dot(W_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), W_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()) * np.dot(W_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna(), W_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()))  # Normalize the result
lags = np.arange(-len(W_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()) + 1, len(W_corr_inflow_west_10042019_2325_2335_Sonic_1.dropna()))
Y = (lags, autocorr_inflow_west_10042019_2325_2335)
Lwx_west_10042019_2325_2335_Sonic_1 = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))]*(1/fs)*abs(Sonic_1_Umean_20Hz)
Lwx_west_10042019_2325_2335_Sonic_1 = Lwx_west_10042019_2325_2335_Sonic_1[Lwx_west_10042019_2325_2335_Sonic_1>0]

autocorr_inflow_west_10042019_2325_2335 = np.correlate(W_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), W_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), mode='full') 
autocorr_inflow_west_10042019_2325_2335 /= np.sqrt(np.dot(W_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), W_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()) * np.dot(W_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna(), W_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()))  # Normalize the result
lags = np.arange(-len(W_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()) + 1, len(W_corr_inflow_west_10042019_2325_2335_Sonic_2.dropna()))
Y = (lags, autocorr_inflow_west_10042019_2325_2335)
Lwx_west_10042019_2325_2335_Sonic_2 = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))]*(1/fs)*abs(Sonic_2_Umean_20Hz)
Lwx_west_10042019_2325_2335_Sonic_2 = Lwx_west_10042019_2325_2335_Sonic_2[Lwx_west_10042019_2325_2335_Sonic_2>0]

autocorr_inflow_west_10042019_2325_2335 = np.correlate(W_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), W_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), mode='full') 
autocorr_inflow_west_10042019_2325_2335 /= np.sqrt(np.dot(W_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), W_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()) * np.dot(W_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna(), W_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()))  # Normalize the result
lags = np.arange(-len(W_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()) + 1, len(W_corr_inflow_west_10042019_2325_2335_Sonic_3.dropna()))
Y = (lags, autocorr_inflow_west_10042019_2325_2335)
Lwx_west_10042019_2325_2335_Sonic_3 = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))]*(1/fs)*abs(Sonic_3_Umean_20Hz)
Lwx_west_10042019_2325_2335_Sonic_3 = Lwx_west_10042019_2325_2335_Sonic_3[Lwx_west_10042019_2325_2335_Sonic_3>0]







# Dataframe downsampling

#import pandas as pd

#dates = pd.to_datetime(['2025-03-25 08:00', '2025-03-25 08:10', '2025-03-25 08:20', '2025-03-25 08:30', '2025-03-25 08:40'])
#values = [10, 15, 20, 25, 30]
#df = pd.DataFrame({'value': values}, index=dates)

#downsampled_df = df.resample('20Min').mean()


