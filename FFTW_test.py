import pyfftw
from pyfftw import FFTW
import multiprocessing
from scipy.fftpack import fft
import numpy as np
import timeit

X = np.random.uniform(0.5,5.5,size=(int(1),int(1e6)));
X = X.astype('complex64');

iteration = 20;
startTime = timeit.default_timer();
for i in range(iteration):
    outputPy = fft(X,axis=0)
print('built-in scipy.fft (20 iteration+scipy.fft): '+str(timeit.default_timer()-startTime)+' sec');

startTime = timeit.default_timer();
flags = ['FFTW_MEASURE']; n_threads = multiprocessing.cpu_count();
output = pyfftw.empty_aligned(X.shape,dtype='complex64');
for i in range(iteration):
    fftw_obj = pyfftw.FFTW(X,output,axes=(0,),flags=flags,threads=n_threads)
    fftw_obj();
print('pyfftw backend (20 iteration+pyfftw.FFTW): '+str(timeit.default_timer()-startTime)+' sec')


startTime = timeit.default_timer();
flags = ['FFTW_MEASURE']; n_threads = multiprocessing.cpu_count();
output = pyfftw.empty_aligned(X.shape,dtype='complex64');
for i in range(iteration):
    fftw_obj = FFTW(X,output,axes=(0,),flags=flags,threads=n_threads)
    fftw_obj();
print('pyfftw backend (20 iteration+FFTW (direct call)): '+str(timeit.default_timer()-startTime)+' sec')
