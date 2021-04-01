import os
import sys
import xarray as xr
import numpy  as np

# Current, parent and file paths
CWD = os.getcwd()

# Import library specific modules
sys.path.append( "../../../")
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming




# Let's create some 2D syntetic data
# and store them into a variable called p
variables = ['p']
x1 = np.linspace(0,10,1000)
x2 = np.linspace(0, 5, 500)
xx1, xx2 = np.meshgrid(x1, x2)
t = np.linspace(0, 200, 1000)
s_component = np.sin(xx1 * xx2) + np.cos(xx1)**2 + np.sin(0.1*xx2)
# s_component = s_component.T
t_component = np.sin(0.1 * t)**2 + np.cos(t) * np.sin(0.5*t)
p = np.empty((t_component.shape[0],)+s_component.shape)
for i, t_c in enumerate(t_component):
	p[i] = s_component * t_c

# We now save the data into netCDF format
ds = xr.Dataset(
        {"p": (("time", "x1", "x2"), p)},
        coords={
            "x1": x2,
            "x2": x1,
            "time": t,
        },
    )
ds.to_netcdf("data.nc")

# We now show how to construct a data reader that can be passed
# to the constructor of pyspod to read data sequentially (thereby
# reducing RAM requirements)

# Reader for netCDF
def read_data_netCDF(data, t_0, t_end, variables):
    if t_0 == t_end: ti = [t_0]
    else           : ti = np.arange(t_0,t_end)
    X = np.empty([len(ti), x2.shape[0], x1.shape[0], len(variables)])
    for _,var in enumerate(variables):
        X = np.array(ds[var].isel(time=ti))
    return X
x_nc = read_data_netCDF('data.nc', t_0=0, t_end=t.shape[0], variables=variables)
x_nc_ssn = read_data_netCDF('data.nc', t_0=0, t_end=0, variables=variables)
print('x_nc.shape = ', x_nc.shape)
print('x_nc_ssn.shape = ', x_nc_ssn.shape)



# Let's define the required parameters into a dictionary
params = dict()

# -- required parameters
params['dt'          ] = 1                	# data time-sampling
params['nt'          ] = t.shape[0]       	# number of time snapshots (we consider all data)
params['xdim'        ] = 2                	# number of spatial dimensions (longitude and latitude)
params['nv'          ] = len(variables)     # number of variables
params['n_FFT'       ] = 100          		# length of FFT blocks (100 time-snapshots)
params['n_freq'      ] = params['n_FFT'] / 2 + 1   			# number of frequencies
params['n_overlap'   ] = np.ceil(params['n_FFT'] * 0 / 100) # dimension block overlap region
params['mean'        ] = 'blockwise' 						# type of mean to subtract to the data
params['normalize'   ] = False        						# normalization of weights by data variance
params['savedir'     ] = os.path.join(CWD, 'results', 'simple_test') # folder where to save results

# -- optional parameters
params['weights']      = None # if set to None, no weighting (if not specified, Default is None)
params['savefreqs'   ] = np.arange(0,params['n_freq']) # frequencies to be saved
params['n_modes_save'] = 3      # modes to be saved
params['normvar'     ] = False  # normalize data by data variance
params['conf_level'  ] = 0.95   # calculate confidence level
params['savefft'     ] = True   # save FFT blocks to reuse them in the future (saves time)


# Initialize libraries by using data_handler for the low storage algorithm
spod_ram = SPOD_low_ram(
    data=os.path.join(CWD,'data.nc'),
    params=params,
    data_handler=read_data_netCDF,
    variables=variables)
spod_ram.fit()
