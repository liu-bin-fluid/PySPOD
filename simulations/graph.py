import os
import sys
import time
import xarray as xr
import numpy  as np
from scipy.io import loadmat
from pathlib import Path

# Import library specific modules
sys.path.append("/hpctmp/e0546050/PySPOD/")
import pyspod
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming
import pyspod.postprocessing as ps
import pyspod.weights as weights

# Current path
CWD = os.getcwd()

# Data path
DP = '/hpctmp2/e0546050/data/ERA5/'

# Inspect and load data
#file = os.path.join(CWD,'ERA5_RA_2019_MSLP.nc')
file = os.path.join(DP,'ERA5_MSL_Summer_2010_2019_hourly.nc')
ds = xr.open_dataset(file)
print(ds)


# we extract time, longitude and latitude
t = np.array(ds['time'])
x1 = np.array(ds['longitude'])
x2 = np.array(ds['latitude'])
print('shape of t (time): ', t.shape)
print('shape of x1 (longitude): ', x1.shape)
print('shape of x2 (latitude) : ', x2.shape)

def read_data(data, t_0, t_end, variables):
	if t_0 == t_end: ti = [t_0]
	else           : ti = np.arange(t_0,t_end)
	X = np.empty([len(ti), x2.shape[0], x1.shape[0], len(variables)])
	for _,var in enumerate(variables):
		X = np.array(ds[var].isel(time=ti))
	return X


# we set the variables we want to use for the analysis
# (we select all the variables present) and load the in RAM
s = time.time()
variables = ['msl']
X = read_data(data=ds, t_0=0, t_end=0, variables=variables)
# for i,var in enumerate(variables):
#     X[...,i] = np.array(ds[var])
# #   X[...,i] = np.einsum('ijk->ikj', np.array(ds[var]))
# #   X[...,i] = np.nan_to_num(X[...,i])
# print('shape of data matrix X: ', X.shape)
# print('elapsed time: ', time.time() - s, 's.')

# define required and optional parameters
params = dict()

CFD = '/hpctmp/e0546050/PySPOD/pyspod/'

coast_1 = loadmat(os.path.join(CFD,'plotting_support','coast.mat'))

print(coast_1)


# plot data
'''
if imaginary:
	real_ax = fig.add_subplot(1, 2, 1)
	real = real_ax.contourf(
		x1, x2, np.real(mode).T,
		vmin=-np.abs(mode).max()*1.,
		vmax= np.abs(mode).max()*1.,
		origin=origin)
	imag_ax = fig.add_subplot(1, 2, 2)
	imag = imag_ax.contourf(
		x1, x2, np.imag(mode).T,
		vmin=-np.abs(mode).max()*1.,
		vmax= np.abs(mode).max()*1.,
		origin=origin)
	if plot_max:
		idx_x1,idx_x2 = np.where(np.abs(mode) == np.amax(np.abs(mode)))
		real_ax.axhline(x1[idx_x1], xmin=0, xmax=1,color='k',linestyle='--')
		real_ax.axvline(x2[idx_x2], ymin=0, ymax=1,color='k',linestyle='--')
		imag_ax.axhline(x1[idx_x1], xmin=0, xmax=1,color='k',linestyle='--')
		imag_ax.axvline(x2[idx_x2], ymin=0, ymax=1,color='k',linestyle='--')
	real_divider = make_axes_locatable(real_ax)
	imag_divider = make_axes_locatable(imag_ax)
	real_cax = real_divider.append_axes("right", size="5%", pad=0.05)
	imag_cax = imag_divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(real, cax=real_cax)
	plt.colorbar(imag, cax=imag_cax)
'''




'''


coast_1 = loadmat(os.path.join(CFD,'plotting_support','coast.mat'))
real_ax.scatter(coast_1['coastlon'], coast['coastlat'], marker='.', c='k', s=1)
imag_ax.scatter(coast_1['coastlon'], coast['coastlat'], marker='.', c='k', s=1)
filename_1 = 'Map_1'
plt.savefig(os.path.join(CWD,filename_1),dpi=400)
plt.close(fig)



coast_2 = loadmat(os.path.join(CFD,'plotting_support','coast_centred.mat'))
real_ax.scatter(coast_2['coastlon'], coast['coastlat'], marker='.', c='k', s=1)
imag_ax.scatter(coast_2['coastlon'], coast['coastlat'], marker='.', c='k', s=1)

filename_2 = 'Map_2'
plt.savefig(os.path.join(CWD,filename_2),dpi=400)
plt.close(fig)


'''


