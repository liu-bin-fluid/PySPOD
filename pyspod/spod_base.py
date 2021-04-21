'''
Base module for the SPOD:
	- `fit` and `predict` methods must be implemented in inherited classes
'''
from __future__ import division

# Import standard Python packages
import os
import sys
import psutil
import warnings
import numpy as np
import scipy.special as sc
from scipy.fft import fft
from numpy import linalg as la

# Import custom Python packages
import pyspod.utils_weights as utils_weights
import pyspod.postprocessing as post

# Current file path
CWD = os.getcwd()
BYTE_TO_GB = 9.3132257461548e-10



class SPOD_base(object):
	'''
	Spectral Proper Orthogonal Decomposition base class.
	'''
	def __init__(self, data, params, data_handler, variables, weights=None):

		# store mandatory parameters in class
		self._dt           		= params['time_step'   ]	# time-step of the data
		self._nt           		= params['n_snapshots' ]	# number of time-frames
		self._xdim         		= params['n_space_dims'] 	# number of spatial dimensions
		self._nv           		= params['n_variables' ]	# number of variables
		self._n_DFT        		= int(params['n_DFT'   ])	# number of DFT (per block)

		# store optional parameters in class
		self._overlap      		= params.get('overlap', 0)			  	 # percentage overlap
		self._mean_type    		= params.get('mean_type', 'longtime')	 # type of mean
		self._normalize_weights = params.get('normalize_weights', False) # normalize weights if required
		self._normalize_data 	= params.get('normalize_data', False)    # normalize data by variance if required
		self._n_modes_save      = params.get('n_modes_save', 1e10)       # default is all (large number)
		self._conf_level		= params.get('conf_level', 0.95) 	     # what confidence level to use fo eigs
		self._reuse_blocks 		= params.get('reuse_blocks', False)      # reuse blocks if present
		self._savefft           = params.get('savefft', False) 		     # save fft block if required
		self._save_dir          = params.get('savedir', os.path.join(CWD, 'results')) # where to save data

		# type of data management
		# - data_handler: read type online
		# - not data_handler: data is entirely pre-loaded
		self._data_handler = data_handler
		self._variables = variables
		if data_handler:
			self._data = data
			X = data_handler(self._data, t_0=0, t_end=1, variables=variables)
			if self._nv == 1 and (X.ndim != self._xdim + 2):
				X = X[...,np.newaxis]
		else:
			def data_handler(data, t_0, t_end, variables):
				if t_0 > t_end:
					raise ValueError('`t_0` cannot be greater than `t_end`.')
				elif t_0 >= self._nt:
					raise ValueError('`t_0` cannot be greater or equal to time dimension.')
				elif t_0 == t_end:
					ti = np.arange(t_0, t_0+1)
					d = data[[t_0],...,:]
				else:
					ti = np.arange(t_0, t_end)
					d = data[ti,...,:]
				return d
			self._data_handler = data_handler
			self._data = np.array(data)
			X = self._data_handler(self._data, t_0=0, t_end=0, variables=self._variables)
			if self._nv == 1 and (self._data.ndim != self._xdim + 2):
				X = X[...,np.newaxis]
				self._data = self._data[...,np.newaxis]

		# get data dimensions and store in class
		self._nx     = X[0,...,0].size
		self._dim    = X.ndim
		self._shape  = X.shape
		self._xdim   = X[0,...,0].ndim
		self._xshape = X[0,...,0].shape

		# check weights
		if isinstance(weights, dict):
			self._weights = weights['weights']
			self._weights_name = weights['weights_name']
			if np.size(self._weights) != int(self.nx * self.nv):
				raise ValueError(
					'parameter ``weights`` must have the '
					'same size as flattened data spatial '
					'dimensions, that is: ', int(self.nx * self.nv))
		else:
			self._weights = np.ones(self._xshape+(self._nv,))
			self._weights_name = 'uniform'
			warnings.warn(
				'Parameter `weights` not equal to an `numpy.ndarray`.'
				'Using default uniform weighting')

		# normalize weigths if required
		if self._normalize_weights:
			self._weights = utils_weights.apply_normalization(
				data=self._data,
				weights=self._weights,
				n_variables=self._nv,
				method='variance')

		# flatten weights to number of spatial point
		try:
			self._weights = np.reshape(
				self._weights, [int(self._nx*self._nv), 1])
		except:
			raise ValurError(
				'parameter ``weights`` must be cast into '
				'1d array with dimension equal to flattened '
				'spatial dimension of data.')

		# Determine whether data is real-valued or complex-valued-valued
		# to decide on one- or two-sided spectrum from data
		self._isrealx = np.isreal(X[0]).all()

		# get default spectral estimation parameters and options
		# define default spectral estimation parameters
		if isinstance(self._n_DFT, int):
			self._window = SPOD_base._hamming_window(self._n_DFT)
			self._window_name = 'hamming'
		else:
			self._n_DFT = int(2**(np.floor(np.log2(self.nt / 10))))
			self._window = SPOD_base._hamming_window(self._n_DFT)
			self._window_name = 'hamming'
			warnings.warn(
				'Parameter `n_DFT` not equal to an integer.'
				'Using default `n_DFT` = ', self._n_DFT)

		# define block overlap
		self._n_overlap = int(np.ceil(self._n_DFT * self._overlap / 100))
		if self._n_overlap > self._n_DFT - 1:
			raise ValueError('Overlap is too large.')

		# define number of blocks
		self._n_blocks = \
			int(np.floor((self.nt - self._n_overlap) \
			/ (self._n_DFT - self._n_overlap)))

		# set number of modes to save
		if self._n_modes_save > self._n_blocks:
			self._n_modes_save = self._n_blocks

		# test feasibility
		if (self._n_DFT < 4) or (self._n_blocks < 2):
			raise ValueError(
				'Spectral estimation parameters not meaningful.')

		# apply mean
		self.select_mean()

		# get frequency axis
		self.get_freq_axis()

		# determine correction for FFT window gain
		self._winWeight = 1 / np.mean(self._window)
		self._window = self._window.reshape(self._window.shape[0], 1)

		# get default for confidence interval
		self._xi2_upper = 2 * sc.gammaincinv(self._n_blocks, 1 - self._conf_level)
		self._xi2_lower = 2 * sc.gammaincinv(self._n_blocks,     self._conf_level)
		self._eigs_c = np.zeros([self._n_freq,self._n_blocks,2], dtype='complex_')

		# create folder to save results
		self._save_dir_blocks = os.path.join(self._save_dir, \
			'nfft'+str(self._n_DFT)+'_novlp'+str(self._n_overlap) \
			+'_nblks'+str(self._n_blocks))
		if not os.path.exists(self._save_dir_blocks):
			os.makedirs(self._save_dir_blocks)
		print('Results folder exist/created')
      
		# create folder to save graph results 
		self._save_Pdir_blocks = os.path.join(self._save_dir,'Graph_of_nfft'+str(self._n_DFT)+\
			'_novlp'+str(self._n_overlap)+'_nblks'+str(self._n_blocks))
		if not os.path.exists(self._save_Pdir_blocks):
			os.makedirs(self._save_Pdir_blocks)
		print('Graph folder exist/created')

		# compute approx problem size (assuming double)
		self._pb_size = self._nt * self._nx * self._nv * 8 * BYTE_TO_GB

		# print parameters to the screen
		self.print_parameters()



	# basic getters
	# ---------------------------------------------------------------------------

	@property
	def save_dir(self):
		'''
		Get the directory where results are saved.

		:return: path to directory where results are saved.
		:rtype: str
		'''
		return self._save_dir
	
	@property
	def save_Pdir(self):
		'''
		Get the directory where the graph are saved.

		:return: path to directory where graph are saved.
		:rtype: str
		'''
		return self._save_Pdir_blocks

	@property
	def dim(self):
		'''
		Get the number of dimensions of the data matrix.

		:return: number of dimensions of the data matrix.
		:rtype: int
		'''
		return self._dim

	@property
	def shape(self):
		'''
		Get the shape of the data matrix.

		:return: shape of the data matrix.
		:rtype: int
		'''
		return self._shape

	@property
	def nt(self):
		'''
		Get the number of time-steps of the data matrix.

		:return: the number of time-steps of the data matrix.
		:rtype: int
		'''
		return self._nt

	@property
	def nx(self):
		'''
		Get the number of spatial points of the data matrix.

		:return: the number of spatial points [dim1:] of the data matrix.
		:rtype: int
		'''
		return self._nx

	@property
	def nv(self):
		'''
		Get the number of variables of the data matrix.

		:return: the number of variables of the data matrix.
		:rtype: int
		'''
		return self._nv

	@property
	def xdim(self):
		'''
		Get the number of spatial dimensions of the data matrix.

		:return: number of spatial dimensions of the data matrix.
		:rtype: tuple(int,)
		'''
		return self._xdim

	@property
	def xshape(self):
		'''
		Get the spatial shape of the data matrix.

		:return: spatial shape of the data matrix.
		:rtype: tuple(int,)
		'''
		return self._xshape

	@property
	def n_freq(self):
		'''
		Get the number of frequencies.

		:return: the number of frequencies computed by the SPOD algorithm.
		:rtype: int
		'''
		return self._n_freq

	@property
	def freq(self):
		'''
		Get the number of modes.

		:return: the number of modes computed by the SPOD algorithm.
		:rtype: int
		'''
		return self._freq

	@property
	def dt(self):
		'''
		Get the time-step.

		:return: the time-step used by the SPOD algorithm.
		:rtype: double
		'''
		return self._dt

	@property
	def n_DFT(self):
		'''
		Get the number of DFT per block.

		:return: the number of DFT per block.
		:rtype: int
		'''
		return self._n_DFT

	@property
	def variables(self):
		'''
		Get the variable list.

		:return: the variable list used.
		:rtype: list or strings
		'''
		return self._variables

	@property
	def eigs(self):
		'''
		Get the eigenvalues of the SPOD matrix.

		:return: the eigenvalues from the eigendecomposition the SPOD matrix.
		:rtype: numpy.ndarray
		'''
		return self._eigs

	@property
	def n_blocks(self):
		'''
		Get the number of blocks used.

		:return: the number of blocks used by the SPOD algorithms.
		:rtype: int
		'''
		return self._n_blocks

	@property
	def n_modes(self):
		'''
		Get the number of modes.

		:return: the number of modes computed by the SPOD algorithm.
		:rtype: int
		'''
		return self._n_modes

	@property
	def n_modes_save(self):
		'''
		Get the number of modes.

		:return: the number of modes computed by the SPOD algorithm.
		:rtype: int
		'''
		return self._n_modes_save

	@property
	def modes(self):
		'''
		Get the dictionary containing the path to the SPOD modes saved.

		:return: the dictionary containing the path to the SPOD modes saved.
		:rtype: dict
		'''
		return self._modes

	@property
	def weights(self):
		'''
		Get the weights used to compute the inner product.

		:return: weight matrix used to compute the inner product.
		:rtype: np.ndarray
		'''
		return self._weights

	@property
	def coeffs(self):
		'''
		Get the dictionary containing the path to the SPOD coefficients saved.

		:return: the dictionary containing the path to the SPOD coefficients saved.
		:rtype: dict
		'''
		return self._coeffs

	@property
	def dynamics(self):
		'''
		Get the dictionary containing the path to the SPOD dynamics saved.

		:return: the dictionary containing the path to the SPOD dynamics saved.
		:rtype: dict
		'''
		pass
	# ---------------------------------------------------------------------------



	# Common methods
	# ---------------------------------------------------------------------------

	def select_mean(self):
		"""Select mean."""
		if self._mean_type.lower() == 'longtime':
			self._x_mean = self.longtime_mean()
			self._mean_name = 'longtime'
		elif self._mean_type.lower() == 'blockwise':
			self._x_mean = 0
			self._mean_name = 'blockwise'
		elif self._mean_type.lower() == 'zero':
			self._x_mean = 0
			self._mean_name = 'zero'
			warnings.warn(
				'No mean subtracted. '
				'Consider providing longtime mean.')
		else:
			raise ValueError(self._mean_type, 'not recognized.')



	def longtime_mean(self):
		"""Get longtime mean."""
		split_block = self.nt // self._n_blocks
		split_res = self.nt % self._n_blocks
		x_sum = np.zeros(self.xshape+(self.nv,))
		for iBlk in range(0, self._n_blocks):
			lb = iBlk * split_block
			ub = lb + split_block
			x_data = self._data_handler(
				data=self._data,
				t_0=lb,
				t_end=ub,
				variables=self.variables)
			x_sum += np.sum(x_data, axis=0)
		if split_res > 0:
			x_data = self._data_handler(
				data=self._data,
				t_0=self.nt-split_res,
				t_end=self.nt,
				variables=self.variables)
			x_sum += np.sum(x_data, axis=0)
		x_mean = x_sum / self.nt
		x_mean = np.reshape(x_mean, (int(self.nx*self.nv)))
		return x_mean



	def get_freq_axis(self):
		"""Obtain frequency axis."""
		self._freq = np.arange(0, self._n_DFT, 1) \
			/ self._dt / self._n_DFT
		if self._isrealx:
			self._freq = np.arange(
				0, np.ceil(self._n_DFT/2)+1, 1) \
				/ self._n_DFT / self._dt
		else:
			if (n_DFT % 2 == 0):
				self._freq[int(n_DFT/2)+1:] = \
					freq[int(self._n_DFT/2)+1:] \
					- 1 / self._dt
			else:
				self._freq[(n_DFT+1)/2+1:] = \
					freq[(self._n_DFT+1)/2+1:] \
					- 1 / self._dt
		self._n_freq = len(self._freq)



	def compute_blocks(self, iBlk):
		"""Compute FFT blocks."""

		# get time index for present block
		offset = min(iBlk * (self._n_DFT - self._n_overlap) \
			+ self._n_DFT, self._nt) - self._n_DFT

		# Get data
		Q_blk = self._data_handler(
			self._data,
			t_0=offset,
			t_end=self._n_DFT+offset,
			variables=self._variables)
		Q_blk = Q_blk.reshape(self._n_DFT, self._nx * self._nv)

		# Subtract longtime or provided mean
		Q_blk = Q_blk[:] - self._x_mean

		# if block mean is to be subtracted,
		# do it now that all data is collected
		if self._mean_type.lower() == 'blockwise':
			Q_blk = Q_blk - np.mean(Q_blk, axis=0)

		# normalize by pointwise variance
		if self._normalize_data:
			Q_var = np.sum((Q_blk - np.mean(Q_blk, axis=0))**2, axis=0) / (self._n_DFT-1)
			# address division-by-0 problem with NaNs
			Q_var[Q_var < 4 * np.finfo(float).eps] = 1;
			Q_blk = Q_blk / Q_var

		# window and Fourier transform block
		self._window = self._window.reshape(self._window.shape[0],1)
		Q_blk = Q_blk * self._window
		Q_blk_hat = (self._winWeight / self._n_DFT) * fft(Q_blk, axis=0);
		Q_blk_hat = Q_blk_hat[0:self._n_freq,:];

		# correct Fourier coefficients for one-sided spectrum
		if self._isrealx:
			Q_blk_hat[1:-1,:] = 2 * Q_blk_hat[1:-1,:]

		return Q_blk_hat, offset



	def compute_standard_spod(self, Q_hat_f, iFreq):
		"""Compute standard SPOD."""

		# compute inner product in frequency space, for given frequency
		M = np.matmul(Q_hat_f.conj().T, (Q_hat_f * self._weights))  / self._n_blocks

		# extract eigenvalues and eigenvectors
		L,V = la.eig(M)
		L = np.real_if_close(L, tol=1000000)

		# reorder eigenvalues and eigenvectors
		idx = np.argsort(L)[::-1]
		L = L[idx]
		V = V[:,idx]

		# compute spatial modes for given frequency
		Psi = np.matmul(Q_hat_f, np.matmul(\
			V, np.diag(1. / np.sqrt(L) / np.sqrt(self._n_blocks))))

		# save modes in storage too in case post-processing crashes
		Psi = Psi[:,0:self._n_modes_save]
		Psi = Psi.reshape(self._xshape+(self._nv,)+(self._n_modes_save,))
		file_psi = os.path.join(self._save_dir_blocks,
			'modes1to{:04d}_freq{:04d}.npy'.format(self._n_modes_save, iFreq))
		np.save(file_psi, Psi)
		self._modes[iFreq] = file_psi
		self._eigs[iFreq,:] = abs(L)

		# get and save confidence interval
		self._eigs_c[iFreq,:,0] = \
			self._eigs[iFreq,:] * 2 * self._n_blocks / self._xi2_lower
		self._eigs_c[iFreq,:,1] = \
			self._eigs[iFreq,:] * 2 * self._n_blocks / self._xi2_upper



	def store_and_save(self):
		"""Store and save results."""

		self._eigs_c_u = self._eigs_c[:,:,0]
		self._eigs_c_l = self._eigs_c[:,:,1]
		file = os.path.join(self._save_dir_blocks, 'spod_energy')
		np.savez(file,
			eigs=self._eigs,
			eigs_c_u=self._eigs_c_u,
			eigs_c_l=self._eigs_c_l,
			f=self._freq)
		self._n_modes = self._eigs.shape[-1]



	def print_parameters(self):

		# display parameter summary
		print('')
		print('SPOD parameters')
		print('------------------------------------')
		print('Problem size               : ', self._pb_size, 'GB. (double)')
		print('No. of snapshots per block : ', self._n_DFT)
		print('Block overlap              : ', self._n_overlap)
		print('No. of blocks              : ', self._n_blocks)
		print('Windowing fct. (time)      : ', self._window_name)
		print('Weighting fct. (space)     : ', self._weights_name)
		print('Mean                       : ', self._mean_name)
		print('Number of frequencies      : ', self._n_freq)
		print('Time-step                  : ', self._dt)
		print('Time snapshots             : ', self._nt)
		print('Space dimensions           : ', self._xdim)
		print('Number of variables        : ', self._nv)
		print('Normalization weights      : ', self._normalize_weights)
		print('Normalization data         : ', self._normalize_data)
		print('Number of modes to be saved: ', self._n_modes_save)
		print('Confidence level for eigs  : ', self._conf_level)
		print('Results to be saved in     : ', self._save_dir)
		print('Save FFT blocks            : ', self._savefft)
		print('Reuse FFT blocks           : ', self._reuse_blocks)
		if self._isrealx: print('Spectrum type             : ',
			'one-sided (real-valued signal)')
		else            : print('Spectrum type             : ',
			'two-sided (complex-valued signal)')
		print('------------------------------------')
		print('')

	# ---------------------------------------------------------------------------



	# getters with arguments
	# ---------------------------------------------------------------------------

	def find_nearest_freq(self, freq_required, freq=None):
		'''
		See method implementation in the postprocessing module.
		'''
		if not isinstance(freq, (list,np.ndarray,tuple)):
			if not freq:
				freq = self.freq
		nearest_freq, idx = post.find_nearest_freq(freq_required=freq_required, freq=freq)
		return nearest_freq, idx

	def find_nearest_coords(self, coords, x):
		'''
		See method implementation in the postprocessing module.
		'''
		xi, idx = post.find_nearest_coords(coords=coords, x=x, data_space_dim=self.xshape)
		return xi, idx

	def get_modes_at_freq(self, freq_idx):
		'''
		See method implementation in the postprocessing module.
		'''
		if self._modes is None:
			raise ValueError('Modes not found. Consider running fit()')
		elif isinstance(self._modes, dict):
			gb_memory_modes = freq_idx * self.nx * self._n_modes_save * \
				sys.getsizeof(complex()) * BYTE_TO_GB
			gb_vram_avail = psutil.virtual_memory()[1] * BYTE_TO_GB
			gb_sram_avail = psutil.swap_memory()[2] * BYTE_TO_GB
			print('- RAM required for loading all modes ~', gb_memory_modes, 'GB')
			print('- Available RAM memory               ~', gb_vram_avail  , 'GB')
			if gb_memory_modes >= gb_vram_avail:
				raise ValueError('Not enough RAM memory to load modes stored, '
								 'for all frequencies.')
			else:
				m = post.get_mode_from_file(self._modes[freq_idx])
		else:
			raise TypeError('Modes must be a dictionary')
		return m

	def get_data(self, t_0, t_end):
		'''
		Get the original input data.

		:return: the matrix that contains the original snapshots.
		:rtype: numpy.ndarray
		'''
		if self._data_handler:
			X = self._data_handler(
				data=self._data, t_0=t_0, t_end=t_end, variables=self._variables)
			if self._nv == 1 and (X.ndim != self._xdim + 2):
				X = X[...,np.newaxis]
		else:
			X = self._data[t_0, t_end]
		return X

	# ---------------------------------------------------------------------------



	# static methods
	# ---------------------------------------------------------------------------

	@staticmethod
	def _are_blocks_present(n_blocks, n_freq, saveDir):
		print('Checking if blocks are already present ...')
		all_blocks_exist = 0
		for iBlk in range(0,n_blocks):
			all_freq_exist = 0
			for iFreq in range(0,n_freq):
				file = os.path.join(saveDir,
					'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
				if os.path.exists(file):
					all_freq_exist = all_freq_exist + 1
			if (all_freq_exist == n_freq):
				print('block '+str(iBlk+1)+'/'+str(n_blocks)+\
					' is present in: ', saveDir)
				all_blocks_exist = all_blocks_exist + 1
		if all_blocks_exist == n_blocks:
			print('... all blocks are present - loading from storage.')
			return True
		else:
			print('... blocks are not present - proceeding to compute them.\n')
			return False

	@staticmethod
	def _are_freq_present(n_blocks, n_freq, saveDir, mode_save):
		print('Checking if frequencies are already present ...')
		all_Mode_Freq_exist = 0
		for iFreq in range(0,n_freq):
			file = os.path.join(saveDir,
				'modes1to{:04d}_freq{:04d}.npy'.format(mode_save,iFreq))
			if os.path.exists(file):
				all_Mode_Freq_exist = all_Mode_Freq_exist + 1
				#print('Mode 1 to '+str(mode_save+1)+'Freq'+str(iFreq)+'/'+str(n_freq)+\
					#' is present in: ', saveDir)
		if (all_Mode_Freq_exist == n_freq):
			print('All frequncies file present for'+'Mode 1 to '+str(mode_save+1))
			return True
		else:
			print('... Frequencies are not present/incomplete - proceeding to compute them.\n')
			return False

	# @staticmethod
	# def _nextpow2(a):
	# 	'''
	# 		Returns the exponents for the smallest powers
	# 		of 2 that satisfy 2^p >= abs(a)
	# 	'''
	# 	p = 0
	# 	v = 0
	# 	while v < np.abs(a):
	# 		v = 2 ** p
	# 		p += 1
	# 	return p

	@staticmethod
	def _hamming_window(N):
		'''
			Standard Hamming window of length N
		'''
		x = np.arange(0,N,1)
		window = (0.54 - 0.46 * np.cos(2 * np.pi * x / (N-1))).T
		return window

	# ---------------------------------------------------------------------------



	# abstract methods
	# ---------------------------------------------------------------------------

	def fit(self):
		'''
		Abstract method to fit the data matrices.
		Not implemented, it has to be implemented in subclasses.
		'''
		raise NotImplementedError(
			'Subclass must implement abstract method {}.fit'.format(
				self.__class__.__name__))

	def predict(self):
		'''
		Abstract method to predict the next time frames.
		Not implemented, it has to be implemented in subclasses.
		'''
		raise NotImplementedError(
			'Subclass must implement abstract method {}.predict'.format(
				self.__class__.__name__))

	# ---------------------------------------------------------------------------



	# plotting methods
	# ---------------------------------------------------------------------------

	def plot_eigs(self,
				  title='',
				  figsize=(12,8),
				  show_axes=True,
				  equal_axes=False,
				  filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_eigs(
			self.eigs, title=title, figsize=figsize, show_axes=show_axes,
			equal_axes=equal_axes, path=self.save_Pdir, filename=filename)

	def plot_eigs_vs_frequency(self,
							   freq=None,
							   title='',
							   xticks=None,
							   yticks=None,
							   show_axes=True,
							   equal_axes=False,
							   figsize=(12,8),
							   filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		if freq is None: freq = self.freq
		post.plot_eigs_vs_frequency(
			self.eigs, freq=freq, title=title, xticks=xticks, yticks=yticks,
			show_axes=show_axes, equal_axes=equal_axes, figsize=figsize,
			path=self.save_Pdir, filename=filename)

	def plot_eigs_vs_period(self,
							freq=None,
							title='',
							xticks=None,
							yticks=None,
							show_axes=True,
							equal_axes=False,
							figsize=(12,8),
							filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		if freq is None: freq = self.freq
		post.plot_eigs_vs_period(
			self.eigs, freq=freq, title=title, xticks=xticks, yticks=yticks,
			figsize=figsize, show_axes=show_axes, equal_axes=equal_axes,
			path=self.save_Pdir, filename=filename)

	def plot_2D_modes_at_frequency(self,
								   freq_required,
								   freq,
								   vars_idx=[0],
								   modes_idx=[0],
								   x1=None,
								   x2=None,
								   fftshift=False,
								   imaginary=False,
								   plot_max=False,
								   coastlines='',
								   title='',
								   xticks=None,
								   yticks=None,
								   figsize=(12,8),
								   equal_axes=False,
								   filename=None,
                                   origin=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_2D_modes_at_frequency(
			self.modes, freq_required=freq_required, freq=freq, vars_idx=vars_idx,
			modes_idx=modes_idx, x1=x1, x2=x2, fftshift=fftshift, imaginary=imaginary,
			plot_max=plot_max, coastlines=coastlines, title=title, xticks=xticks, yticks=yticks,
			figsize=figsize, equal_axes=equal_axes, path=self.save_Pdir, filename=filename)

	def plot_2D_mode_slice_vs_time(self,
								   freq_required,
								   freq,
								   vars_idx=[0],
								   modes_idx=[0],
								   x1=None,
								   x2=None,
								   max_each_mode=False,
								   fftshift=False,
								   title='',
								   figsize=(12,8),
								   equal_axes=False,
								   filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_2D_mode_slice_vs_time(
			self.modes, freq_required=freq_required, freq=freq, vars_idx=vars_idx,
			modes_idx=modes_idx, x1=x1, x2=x2, max_each_mode=max_each_mode,
			fftshift=fftshift, title=title, figsize=figsize, equal_axes=equal_axes,
			path=self.save_Pdir, filename=filename)

	def plot_3D_modes_slice_at_frequency(self,
										 freq_required,
										 freq,
										 vars_idx=[0],
										 modes_idx=[0],
										 x1=None,
										 x2=None,
										 x3=None,
										 slice_dim=0,
										 slice_id=None,
										 fftshift=False,
										 imaginary=False,
										 plot_max=False,
										 coastlines='',
										 title='',
										 xticks=None,
										 yticks=None,
										 figsize=(12,8),
										 equal_axes=False,
										 filename=None,
                                         origin=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_3D_modes_slice_at_frequency(
			self.modes, freq_required=freq_required, freq=freq,
			vars_idx=vars_idx, modes_idx=modes_idx, x1=x1, x2=x2,
			x3=x3, slice_dim=slice_dim, slice_id=slice_id, fftshift=fftshift,
			imaginary=imaginary, plot_max=plot_max, coastlines=coastlines,
			title=title, xticks=xticks, yticks=yticks, figsize=figsize,
			equal_axes=equal_axes, path=self.save_Pdir, filename=filename)

	def plot_mode_tracers(self,
						  freq_required,
						  freq,
						  coords_list,
						  x=None,
						  vars_idx=[0],
						  modes_idx=[0],
						  fftshift=False,
						  title='',
						  figsize=(12,8),
						  filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_mode_tracers(
			self.modes, freq_required=freq_required, freq=freq, coords_list=coords_list,
			x=x, vars_idx=vars_idx, modes_idx=modes_idx, fftshift=fftshift,
			title=title, figsize=figsize, path=self.save_Pdir, filename=filename)

	def plot_2D_data(self,
					 time_idx=[0],
					 vars_idx=[0],
					 x1=None,
					 x2=None,
					 title='',
					 coastlines='',
					 figsize=(12,8),
					 filename=None,
                     origin=None):
		'''
		See method implementation in the postprocessing module.
		'''
		max_time_idx = np.max(time_idx)
		post.plot_2D_data(
			X=self.get_data(t_0=0, t_end=max_time_idx+1),
			time_idx=time_idx, vars_idx=vars_idx, x1=x1, x2=x2,
			title=title, coastlines=coastlines, figsize=figsize,
			path=self.save_Pdir, filename=filename)

	def plot_data_tracers(self,
						  coords_list,
						  x=None,
						  time_limits=[0,10],
						  vars_idx=[0],
						  title='',
						  figsize=(12,8),
						  filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_data_tracers(
			X=self.get_data(t_0=time_limits[0], t_end=time_limits[-1]),
			coords_list=coords_list, x=x, time_limits=time_limits,
			vars_idx=vars_idx, title=title, figsize=figsize, path=self.save_Pdir,
			filename=filename)

	# ---------------------------------------------------------------------------



	# Generate animations
	# ---------------------------------------------------------------------------
	def generate_2D_data_video(self,
							   time_limits=[0,10],
							   vars_idx=[0],
							   sampling=1,
							   x1=None,
							   x2=None,
							   coastlines='',
							   figsize=(12,8),
							   filename='data_video.mp4'):
		'''
		See method implementation in the postprocessing module.
		'''
		post.generate_2D_data_video(
			X=self.get_data(t_0=time_limits[0], t_end=time_limits[-1]),
			time_limits=[0,time_limits[-1]], vars_idx=vars_idx, sampling=sampling,
			x1=x1, x2=x2, coastlines=coastlines, figsize=figsize, path=self.save_Pdir,
			filename=filename)


	# ---------------------------------------------------------------------------



	# Data-driven emulation (after modes and DFT blocks are saved to disk)
	# ---------------------------------------------------------------------------
	
	#def get_coefficients(self):
		'''
		Get Fourier transformed data and modes
		'''

		'''
		Inner product for projection - requires loading Qfft blocks
		'''
		# Load modes
		'''
		n_modes_save = self._n_blocks
		if 'n_modes_save' in self._params: n_modes_save = self._params['n_modes_save']

		# For each frequency
		for iFreq in tqdm(range(0,self._n_freq),desc='computing coefficients'):
			# load FFT data from previously saved file
			Q_hat_f = np.zeros([self._nx,self._n_blocks], dtype='complex_')
			for iBlk in range(0,self._n_blocks):
				file = os.path.join(self._save_dir_blocks,'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
				Q_hat_f[:,iBlk] = np.load(file)

			file_psi = os.path.join(self._save_dir_blocks,'modes1to{:04d}_freq{:04d}.npy'.format(n_modes_save,iFreq))
			Psi = np.load(file_psi)
			Psi = Psi.reshape(-1,n_modes_save)

			# compute inner product between Qfft and modes
			a_k = np.matmul(Psi.T, Q_hat_f * self._weights).T
			# Save these coefficients for posterity
			file_a_k = os.path.join(self._save_dir_blocks,'coeffs1to{:04d}_freq{:04d}.npy'.format(n_modes_save,iFreq))
			np.save(file_a_k, a_k)
			'''

	def build_emulator(self):

		# Load coefficients
		coeff_list = [] # Need to stack coefficients from multiple frequencies

		# Load coefficients and append
		n_modes_save = self._n_blocks
		if 'n_modes_save' in self._params: n_modes_save = self._params['n_modes_save']

		for iFreq in tqdm(range(0,self._n_freq),desc='loading coefficients'):
			file_a_k = os.path.join(self._save_dir_blocks,'coeffs1to{:04d}_freq{:04d}.npy'.format(n_modes_save,iFreq))
			a_k = np.load(file_a_k)
			coeff_list.append(a_k)

		# Training data
		emulator_prep_data = np.moveaxis(np.asarray(coeff_list),0,1)

		# Set up the data in the right shape for input<->output relationship
		input_block = []
		output_block = []
		for sample in range(self._n_blocks-1):
			input_block.append(emulator_prep_data[sample])
			output_block.append(emulator_prep_data[sample+1])

		# Do the training and testing split
		num_train_blocks = int(0.7*self._n_blocks)

		# Split
		self.training_data_ip = np.asarray(input_block[:num_train_blocks])
		self.training_data_op = np.asarray(output_block[:num_train_blocks])

		self.testing_data_ip = np.asarray(input_block[num_train_blocks:])
		self.testing_data_op = np.asarray(output_block[num_train_blocks:])

		# if normalize:
		# 	self.lstm_normalize = True

		# 	train_ip_shape = self.training_data_ip.shape
		# 	train_op_shape = self.training_data_op.shape
		# 	test_ip_shape = self.testing_data_ip.shape
		# 	test_op_shape = self.testing_data_op.shape
			
		# 	from sklearn.preprocessing import MinMaxScaler
			
		# 	self.ip_scaler = MinMaxScaler()

		# 	self.training_data_ip = self.ip_scaler.fit_transform(
		# 					self.training_data_ip.reshape(-1,train_ip_shape[-1])).reshape(train_ip_shape)
			
		# 	self.testing_data_ip = self.ip_scaler.fit(
		# 					self.testing_data_ip.reshape(-1,test_ip_shape[-1])).reshape(test_ip_shape)

		# 	self.op_scaler = MinMaxScaler()

		# 	self.training_data_op = self.op_scaler.fit_transform(
		# 					self.training_data_op.reshape(-1,train_op_shape[-1])).reshape(train_op_shape)
			
		# 	self.testing_data_op = self.op_scaler.fit(
		# 					self.testing_data_op.reshape(-1,test_op_shape[-1])).reshape(test_op_shape)

		# Construct the LSTM model
		self.generate_lstm_model()


	def generate_lstm_model(self):
		'''
		We will need to update the requirements for this guy here
		Tested with TensorFlow 2.3
		'''
		import tensorflow as tf
		from tensorflow.keras.layers import LSTM, Dense, Reshape, Input
		from tensorflow.keras import optimizers, models, regularizers
		from tensorflow.keras import backend as K
		from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
		from tensorflow.keras.models import load_model, Model
		from tensorflow.keras.utils import plot_model

		def coeff_determination(y_pred, y_true): 
			SS_res =  K.sum(K.square( y_true-y_pred ),axis=0) 
			SS_tot = K.sum(K.square( y_true - K.mean(y_true,axis=0) ),axis=0 )
			return K.mean(1 - SS_res/(SS_tot + K.epsilon()) )

		n_modes_save = self._n_blocks
		if 'n_modes_save' in self._params: n_modes_save = self._params['n_modes_save']

		lstm_inputs = Input(shape=(self._n_freq,n_modes_save,),name='lstm_inputs')
		x = LSTM(50,return_sequences=True)(lstm_inputs)
		x = LSTM(50,return_sequences=True)(x)
		lstm_outputs = Dense(n_modes_save,activation=None)(x)

		self.lstm_model = Model(inputs=lstm_inputs,outputs=lstm_outputs,name='Emulation_Model')

		# design network
		self.lstm_filepath = os.path.join(self._save_dir_blocks,'emulator_weights.h5')
		lstm_adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		checkpoint = ModelCheckpoint(self.lstm_filepath, monitor='loss', verbose=0, save_best_only=True, mode='min',save_weights_only=True)
		earlystopping = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
		self.lstm_callbacks_list = [checkpoint, earlystopping]

		# fit network
		self.lstm_model.compile(optimizer=lstm_adam,loss='mean_squared_error',metrics=[coeff_determination])
		self.lstm_model.summary()

		return None

	def fit_emulator(self,batch_size,num_epochs):
		self.lstm_train_history = self.lstm_model.fit(self.training_data_ip, self.training_data_op, 
			epochs=num_epochs, batch_size=batch_size, callbacks=self.lstm_callbacks_list)

		# Load the best weights after training
		self.lstm_model.load_weights(self.lstm_filepath)

	def test_emulator(self):
		self.testing_data_pred = self.lstm_model.predict(self.testing_data_ip)
		test_op_shape = self.testing_data_pred.shape
		
		# if self.lstm_normalize:
		# 	self.testing_data_pred = self.op_scaler.inverse_transform(
		# 					self.testing_data_pred.reshape(-1,test_op_shape[-1])).reshape(test_op_shape)

		'''
		Reconstruct Qfft blocks
		'''
		# Testing reconstruction
		num_train_blocks = int(0.7*self._n_blocks)
		num_test_blocks = self._n_blocks - int(0.7*self._n_blocks)


		# Load modes
		n_modes_save = self._n_blocks
		if 'n_modes_save' in self._params: n_modes_save = self._params['n_modes_save']

		# For each frequency
		for iFreq in tqdm(range(0,self._n_freq),desc='computing predicted coefficients'):
			# Load modes
			file_psi = os.path.join(self._save_dir_blocks,'modes1to{:04d}_freq{:04d}.npy'.format(n_modes_save,iFreq))
			Psi = np.load(file_psi)
			Psi = Psi.reshape(-1,n_modes_save)

			# Compute FFT data from modes and LSTM predicted coefficients
			Q_hat_f_pred = np.zeros([self._nx,num_test_blocks], dtype='complex_')

			for iBlk in range(num_test_blocks):
				#Q_hat_f[:,iBlk] = 

				# compute inner product between Qfft and modes
				a_k = np.matmul(Psi.T, Q_hat_f * self._weights).T
			

			file_a_k = os.path.join(self._save_dir_blocks,'coeffs1to{:04d}_freq{:04d}.npy'.format(n_modes_save,iFreq))
			a_k = np.load(file_a_k)

			print(a_k.shape)

			# self.testing_data_pred
		


			



