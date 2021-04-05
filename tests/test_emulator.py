import os
import sys
import shutil
import subprocess
import numpy as np
import tensorflow as tf
from numpy import linalg
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM
np.random.seed(5)
tf.random.set_seed(5)

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD,"../"))
sys.path.append(os.path.join(CFD,"../pyspod"))
from pyspod.spod_low_storage import SPOD_low_storage
from pyspod.spod_low_ram     import SPOD_low_ram
from pyspod.spod_streaming   import SPOD_streaming



# parameters for burgulence test (see R. Maulik, 2020)
nx = 2048
alpha = 2.0e-3
x = np.linspace(0.0, 2.0 * np.pi, num=2048)
dx = 2.0 * np.pi / np.shape(x)[0]
Nt = 10000
N  = 5.0
dt = N / Nt
tsteps = np.linspace(0.0, N, num=int(N / dt))

# load data
X = np.load(os.path.join('data', 'burgers_data_long_large.npy'))

t = np.arange(np.shape(tsteps)[0])
X_mean = np.mean(X, axis=0)
print(X_mean)
print(X)
print(t)
print(X.shape)
print(t.shape)

num_modes = 3
deployment_mode = 'test' # or 'train'



# Let's define the required parameters into a dictionary
params = dict()

# -- required parameters
params['dt'          ] = dt                	# data time-sampling
params['nt'          ] = t.shape[0]       	# number of time snapshots (we consider all data)
params['xdim'        ] = 1                	# number of spatial dimensions (longitude and latitude)
params['nv'          ] = 1     				# number of variables
params['n_FFT'       ] = 200          		# length of FFT blocks (100 time-snapshots)
params['n_freq'      ] = params['n_FFT'] / 2 + 1   			     # number of frequencies
params['n_overlap'   ] = np.ceil(params['n_FFT'] * 0 / 100)      # dimension block overlap region
params['mean'        ] = 'longtime' 						     # type of mean to subtract to the data
params['normalize'   ] = False        						     # normalization of weights by data variance
params['savedir'     ] = os.path.join(CWD, 'results', 'burgers') # folder where to save results

# -- optional parameters
params['weights']      = None # if set to None, no weighting (if not specified, Default is None)
params['savefreqs'   ] = np.arange(0, params['n_freq']) # frequencies to be saved
params['n_modes_save'] = num_modes # modes to be saved
params['normvar'     ] = False     # normalize data by data variance
params['conf_level'  ] = 0.95      # calculate confidence level
params['savefft'     ] = False     # save FFT blocks to reuse them in the future (saves time)



def test_burgers_spod():

	# initialize libraries for the low_storage algorithm
	spod = SPOD_low_storage(X, params=params, data_handler=False, variables=['p'])
	spod.fit()


	# let's plot the data
	spod.plot_1D_data(time_idx=[0,99,199,299,399])
	spod.plot_data_tracers(coords_list=[(10,)], time_limits=[0,t.shape[0]])

	# show results
	T_approx = 10
	freq = spod.freq
	spod.plot_eigs()
	spod.plot_1D_modes_at_frequency(freq_required=1/T_approx, freq=freq, modes_idx=[0,1,2])
	spod.plot_eigs_vs_period()
	modes = spod.modes
	eigs = spod.eigs
	spod.compute_coeffs()





def test_burgers_pod():

	# Truth - each column of phi spans the global domain
	phi_r, coeffs_r = generate_pod_bases(X, params['n_modes_save'], tsteps=tsteps)
	perfect_output = coeffs_r[:,-1]
	print('perfect_output = ', perfect_output)

	# POD Galerkin - for comparison
	output_state_gp, state_tracker_gp = galerkin_projection(phi_r, coeffs_r, X_mean, tsteps, alpha, dt, dx, num_modes)
	print('output_state_gp  = ', output_state_gp )
	print('state_tracker_gp = ', state_tracker_gp)
	print('state_tracker_gp.shape = ', state_tracker_gp.shape)

	# LSTM network - note this will only give good predictions till the last three timesteps
	model = lstm_for_dynamics(coeffs_r, deployment_mode='train')
	output_state_lstm, state_tracker_lstm = evaluate_rom_deployment_lstm(model, coeffs_r, tsteps)
	# np.save('burgulence_LSTM_coeffs.npy', state_tracker_lstm)

	# visualization - spectra
	u_true = X_mean + (np.matmul(phi_r, perfect_output        ))[:]
	u_gp   = X_mean + (np.matmul(phi_r, output_state_gp       ))[:]
	u_lstm = X_mean + (np.matmul(phi_r, output_state_lstm[:,0]))[:]


	# plots
	# -------------------------------------------------------

	## coefficients
	plt.plot(state_tracker_gp[0,:-1], label='gp_tracker')
	plt.plot(coeffs_r[0,:], label='coeff')
	plt.legend(); plt.show()
	plt.plot(state_tracker_gp[1,:-1], label='gp_tracker')
	plt.plot(coeffs_r[1,:], label='coeff')
	plt.legend(); plt.show()
	plt.plot(state_tracker_gp[2,:-1], label='gp_tracker')
	plt.plot(coeffs_r[2,:], label='coeff')
	plt.legend(); plt.show()

	## spectra
	plt.figure()
	kx_plot = np.array([float(i) for i in list(range(0, nx // 2))])
	espec1 = spectra_calculation(u_true)
	espec2 = spectra_calculation(u_gp)
	espec3 = spectra_calculation(u_lstm)
	plt.loglog(kx_plot, espec1,label='Truth')
	plt.loglog(kx_plot, espec2,label='GP')
	plt.loglog(kx_plot, espec3,label='LSTM')
	plt.legend()
	plt.show()

	# spectra residuals
	plt.figure()
	kx_plot = np.array([float(i) for i in list(range(0, nx // 2))])
	plt.loglog(kx_plot, np.abs(espec2 - espec1), label='GP-Residual')
	plt.loglog(kx_plot, np.abs(espec3 - espec1), label='LSTM-Residual')
	plt.legend()
	plt.show()

	# dynamics
	plt.figure()
	plt.plot(x[:], u_true[:], label='Truth')
	plt.plot(x[:], u_gp  [:], label='POD-GP')
	plt.plot(x[:], u_lstm[:], label='POD-LSTM')
	plt.legend(fontsize=18)
	plt.show()

	# state stabilization - modal coefficient 3
	mode_num = 2
	plt.figure()
	plt.plot(coeffs_r          [mode_num,:-1], label='Truth')
	plt.plot(state_tracker_gp  [mode_num,:-1], label='POD-GP')
	plt.plot(state_tracker_lstm[mode_num,:-1], label='POD-LSTM')
	plt.legend()
	plt.show()





def generate_pod_bases(snapshot_matrix, num_modes, tsteps):
	'''
	Takes input of a snapshot matrix with mean removed and computes POD bases
	Outputs truncated POD bases and coefficients
	'''

	snapshot_matrix_mean = np.mean(snapshot_matrix, axis=0)
	snapshot_matrix = snapshot_matrix - snapshot_matrix_mean

	# print(snapshot_matrix.shape)
	# snapshot_matrix = snapshot_matrix.T
	# print(snapshot_matrix.shape)

	# eigendecomposition of covariance matrix
	new_mat = np.matmul(snapshot_matrix, snapshot_matrix.T)
	print('new_mat.shape = ', new_mat.shape)

	w, v = linalg.eig(new_mat)
	print('w.shape = ', w.shape)
	print('v.shape = ', v.shape)

	# basis
	phi = np.real(np.matmul(snapshot_matrix.T, v))
	t = np.arange(np.shape(tsteps)[0])
	phi[:,t] = phi[:,t] / np.sqrt(w[:])
	print('phi.shape = ', phi.shape)

	# coefficients
	# equation (10): https://arxiv.org/pdf/1906.07815.pdf
	coefficient_matrix = np.matmul(phi.T, snapshot_matrix.T)
	print('coefficient_matrix.shape = ', coefficient_matrix.shape)

	# truncate coefficient and basis matrices
	phi_r = phi[:,0:num_modes]
	coeffs_r = coefficient_matrix[0:num_modes,:]
	print('phi_r.shape = ', phi_r.shape)
	print('coeffs_r.shape = ', coeffs_r.shape)

	# for i in range(0,4):
	# 	plot_pod_modes(phi, mode_num=i)
	#
	# for i in range(0,40,10):
	# 	plot_pod_coeffs(coefficient_matrix, mode_num=i)

	return phi_r, coeffs_r



def plot_pod_modes(phi, mode_num):
	plt.figure()
	plt.plot(phi[:,mode_num])
	plt.title('MODES - Mode ID: ' + str(mode_num))
	plt.show()



def plot_pod_coeffs(coeffs, mode_num):
	plt.figure()
	plt.plot(coeffs[mode_num,:])
	plt.title('COEFFS - Mode ID: ' + str(mode_num))
	plt.show()



def galerkin_projection(phi_r, coeffs_r, X_mean, tsteps, alpha, dt, dx, num_modes):

	# setup offline operators
	# function for linear operator
	dataset_temp = np.copy(coeffs_r)

	def linear_operator(u): # Requires ghost-points - this is laplacian
		u_per = np.zeros(shape=(np.shape(u)[0]+2), dtype='double')
		u_per[1:-1] = u[ :]
		u_per[ 0]   = u[-1]
		u_per[-1]   = u[ 0]
		ulinear = (u_per[0:-2] + u_per[2:] - 2.0 * u_per[1:-1]) / (dx * dx)
		return ulinear

	def nonlinear_operator(u, g):
		g_per = np.zeros(shape=(np.shape(g)[0]+2), dtype='double')
		g_per[1:-1] = g[ :]
		g_per[ 0]   = g[-1]
		g_per[-1]   = g[ 0]
		dgdx = (g_per[0:-2] - g_per[2:]) / (2.0 * dx)
		return u * dgdx

	# calculate mode-wise offline operators
	lin_ubar  = alpha * linear_operator(X_mean)
	nlin_ubar = nonlinear_operator(X_mean, X_mean)
	print('lin_ubar = ', lin_ubar.shape)
	print('nlin_ubar = ', nlin_ubar.shape)

	# calculate linear and non-linear terms
	b1k   = np.zeros(shape=(np.shape(coeffs_r)[0]), dtype='double')
	b2k   = np.zeros(shape=(np.shape(coeffs_r)[0]), dtype='double')
	lik_1 = np.zeros(shape=(np.shape(coeffs_r)[0], np.shape(coeffs_r)[0]), dtype='double')
	lik_2 = np.zeros(shape=(np.shape(coeffs_r)[0], np.shape(coeffs_r)[0]), dtype='double')
	nijk  = np.zeros(shape=(np.shape(coeffs_r)[0], np.shape(coeffs_r)[0], np.shape(coeffs_r)[0]), dtype='double')

	for k in range(num_modes):
		b1k[k] = np.sum(lin_ubar [:] * phi_r[:,k])
		b2k[k] = np.sum(nlin_ubar[:] * phi_r[:,k])
		print('b1k[k] = ', b1k[k])
		print('b2k[k] = ', b2k[k])

		for i in range(num_modes):
			lin_phi = alpha * linear_operator(phi_r[:,i])
			lik_1[i,k] = np.sum(lin_phi[:] * phi_r[:,k])

			nlin_phi = nonlinear_operator(X_mean, phi_r[:,i]) + nonlinear_operator(phi_r[:,i], X_mean)
			lik_2[i,k] = np.sum(nlin_phi[:] * phi_r[:,k])

			for j in range(num_modes):
				nlin_phi = nonlinear_operator(phi_r[:,i], phi_r[:,j])
				nijk[i,j,k] = np.sum(nlin_phi[:] * phi_r[:,k])

	# operators fixed - one time cost
	# evaluation using GP
	def gp_rhs(b1k, b2k, lik_1, lik_2, nijk, state):
		rhs = np.zeros(np.shape(state)[0])
		rhs = b1k[:] + b2k[:]
		rhs = rhs + np.matmul(state, lik_1) + np.matmul(state, lik_2)

		# Nonlinear global operator
		for k in range(num_modes):
			rhs[k] = rhs[k] + np.matmul(np.matmul(nijk[:,:,k], state), state)
		return rhs

	state = dataset_temp[:,0]
	state_tracker = np.zeros(shape=(np.shape(tsteps)[0], np.shape(state)[0]), dtype='double')

	trange = np.arange(int(np.shape(tsteps)[0])-1)
	for t in trange:
		state_tracker[t,:] = state[:]

		# TVDRK3 - POD GP
		rhs = gp_rhs(b1k, b2k, lik_1, lik_2, nijk, state)
		l1  = state + dt * rhs
		rhs = gp_rhs(b1k, b2k, lik_1, lik_2, nijk, l1)
		l2  = 0.75 * state + 0.25 * l1 + 0.25 * dt * rhs
		rhs = gp_rhs(b1k, b2k, lik_1, lik_2, nijk, l2)
		state[:] = 1.0 / 3.0 * state[:] + 2.0 / 3.0 * l2[:] + 2.0 / 3.0 * dt * rhs[:]

	return state, np.transpose(state_tracker)




def lstm_for_dynamics(coeffs_r, deployment_mode='test'):

	# LSTM hyperparameters
	seq_num    = 30
	num_units  = 73
	lrate      = 0.0005440360402
	rho        = 0.998848446841937
	decay      = 6.1587540045897e-06
	num_epochs = 317
	batch_size = 23

	features = np.transpose(coeffs_r)
	# rows are time, columns are state values
	states = np.copy(features[:,:])

	# need to make batches of 10 input sequences and 1 output
	total_size = np.shape(features)[0]-seq_num
	input_seq = np.zeros(shape=(total_size,seq_num,np.shape(states)[1]))
	output_seq = np.zeros(shape=(total_size,np.shape(states)[1]))

	for t in range(total_size):
		input_seq [t,:,:] = states[None     ,t:t+seq_num,:]
		output_seq[t,:]   = states[t+seq_num,:]

	idx = np.arange(total_size)
	np.random.shuffle(idx)

	input_seq  = input_seq [idx,:,:]
	output_seq = output_seq[idx,:]

	# Model architecture
	model = Sequential()
	# returns a sequence of vectors of dimension 32
	model.add(LSTM(num_units, input_shape=(seq_num, np.shape(states)[1])))
	model.add(Dense(np.shape(states)[1], activation='linear'))

	# design network
	my_adam = optimizers.RMSprop(lr=lrate, rho=rho, epsilon=None, decay=decay)

	# load model if present
	filepath = "best_weights_lstm.h5"
	checkpoint = ModelCheckpoint(
		filepath,
		verbose=1,
		mode='min',
		monitor='val_loss',
		save_best_only=True,
		save_weights_only=True)
	callbacks_list = [checkpoint]

	# fit network
	model.compile(optimizer=my_adam, loss='mean_squared_error', metrics=[coeff_determination])

	# train if required
	if deployment_mode == 'train':
		train_history = model.fit(
			input_seq,
			output_seq,
			epochs=num_epochs,
			batch_size=batch_size,
			validation_split=0.33,
			callbacks=callbacks_list)
		np.save('train_loss.npy',train_history.history['loss'])
		np.save('valid_loss.npy',train_history.history['val_loss'])
	model.load_weights(filepath)
	return model



def evaluate_rom_deployment_lstm(model, dataset, tsteps):

	seq_num = 30

	# make the initial condition from the first seq_num columns of the dataset
	features = np.transpose(dataset)
	input_state = np.copy(features[0:seq_num,:])
	state_tracker = np.zeros(shape=(1,int(np.shape(tsteps)[0]),np.shape(features)[1]),dtype='double')
	state_tracker[0,0:seq_num,:] = input_state[0:seq_num,:]

	for t in range(seq_num,int(np.shape(tsteps)[0])):
		lstm_input = state_tracker[:,t-seq_num:t,:]
		output_state = model.predict(lstm_input)
		state_tracker[0,t,:] = output_state[:]

	return np.transpose(output_state), np.transpose(state_tracker[0,:,:])



def coeff_determination(y_pred, y_true): # order of function inputs is important here
	SS_res =  K.sum(K.square( y_true - y_pred ))
	SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
	return ( 1 - SS_res / (SS_tot + K.epsilon()) )



def spectra_calculation(u):
	# transform to Fourier space
	array_hat = np.real(np.fft.fft(u))

	# normalizing data
	array_new = np.copy(array_hat / float(nx))
	# energy spectrum
	espec = 0.5 * np.absolute(array_new)**2
	# angle averaging
	eplot = np.zeros(nx // 2, dtype='double')
	for i in range(1, nx // 2):
		eplot[i] = 0.5 * (espec[i] + espec[nx - i])

	return eplot



if __name__ == "__main__":
	test_burgers_spod()
	# test_burgers_pod()
