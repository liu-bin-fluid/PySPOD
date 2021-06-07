"""Derived module from spod_base.py for low storage SPOD."""

# import standard python packages
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import psutil

#parallel implementation
from mpi4py import MPI
import shutil

# import PySPOD base class for SPOD_low_storage
from pyspod.spod_base_parallel import SPOD_base


BYTE_TO_GB = 9.3132257461548e-10

comm = MPI.COMM_WORLD
rank = comm.Get_rank();
size = comm.Get_size();

class SPOD_low_storage(SPOD_base):
    """
    Class that implements the Spectral Proper Orthogonal Decomposition
    to the input data using RAM to reduce the amount of I/O
    and disk storage (for small datasets / large RAM machines).

    The computation is performed on the data *X* passed to the
    constructor of the `SPOD_low_storage` class, derived from
    the `SPOD_base` class.
    """

    def fit(self):
        """
        Class-specific method to fit the data matrix X using
        the SPOD low storage algorithm.
        """
        #start = time.time()

        if rank == 0:
            print(' ',flush=True)
            print('Calculating temporal DFT (low_storage)',flush=True)
            print('--------------------------------------',flush=True)
        else:
            None;
        comm.Barrier();
        # check RAM requirements
        
        gb_vram_required = self._n_DFT * self._nx * self._nv * sys.getsizeof(complex()) * BYTE_TO_GB
        gb_vram_avail = (psutil.virtual_memory()[1])/size * BYTE_TO_GB
        
        print('On Processor '+str(rank)+'; RAM available = ', gb_vram_avail,flush=True)
        comm.Barrier();         
        print('On Processor '+str(rank)+'; RAM required  = ', gb_vram_required,flush=True)
        comm.Barrier();       
        if gb_vram_required > 1.5 * gb_vram_avail:
            raise ValueError(
                'On Processor '+str(rank)+'RAM required larger than RAM available... '
                'consider running spod_low_ram to avoid system freezing.',flush=True)
        else:
            None;
        comm.barrier();

        # check if blocks are already saved in memory
        blocks_present = False
        if self._reuse_blocks:
            blocks_present = self._are_blocks_present(
                self._n_blocks,self._n_freq,self._save_dir_blocks)
        else:
            None;

        if not blocks_present:
            # loop over number of blocks and generate Fourier realizations
            # if blocks are not saved in storage
            if size >= self._n_blocks:
                if rank < self._n_blocks:
                    iBlk = rank;
                    # compute block
                    Q_blk_hat, offset = self.compute_blocks(iBlk)
                    # print info file
                    print('block '+str(iBlk+1)+'/'+str(self._n_blocks)+\
                          ' ('+str(offset)+':'+str(self._n_DFT+offset)+')',flush=True)
                    # save FFT blocks in storage memory if required
                    if self._savefft:
                        for iFreq in range(0,self._n_freq):
                            file = os.path.join(self._save_dir_blocks,
                                'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
                            Q_blk_hat_fi = Q_blk_hat[iFreq,:]
                            np.save(file, Q_blk_hat_fi);
                    else:
                        None;
                else:
                    None;
                comm.Barrier();
            else:    
                if rank == 0:
                    #print("Split file list ...",flush=True);
                    def split(a, n):
                        k, m = divmod(len(a), n)
                        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
                    chunks = split(list(range(self._n_blocks)),size);
                else:
                    chunks = None;        
                local_n_blocks = comm.scatter(chunks, root=0);
                comm.Barrier(); # wait for processors to finish
                for iBlk in local_n_blocks:
                    # compute block
                    Q_blk_hat, offset = self.compute_blocks(iBlk)
                    # print info file
                    print('block '+str(iBlk+1)+'/'+str(self._n_blocks)+\
                          ' ('+str(offset)+':'+str(self._n_DFT+offset)+')',flush=True)
                    # save FFT blocks in storage memory if required
                    if self._savefft:
                        for iFreq in range(0,self._n_freq):
                            file = os.path.join(self._save_dir_blocks,
                                'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
                            Q_blk_hat_fi = Q_blk_hat[iFreq,:]
                            np.save(file, Q_blk_hat_fi);
                    else:
                        None;   
                comm.Barrier();
        else:
            None;
            
        if rank == 0:
            print('--------------------------------------',flush=True)
        else:
            None;

        #------------ parallel eigendecomposition ---------------
        # loop over all frequencies and calculate SPOD
        if rank == 0:
            print(' ',flush=True)
            print('Calculating SPOD (low_storage)',flush=True)
            print('--------------------------------------',flush=True)
        else:
            None;
        comm.Barrier();

        self._eigs = np.zeros([self._n_freq,self._n_blocks], dtype='complex_')
        self._modes = dict()


        # load FFT blocks from hard drive and save modes on hard drive (for large data)
        for iFreq in tqdm(range(0,self._n_freq),desc='computing frequencies'):
            # load FFT data from previously saved file
            Q_hat_f = np.zeros([self._nx,self._n_blocks], dtype='complex_');
            if rank == 0:
                for iBlk in range(0,self._n_blocks):
                    file = os.path.join(self._save_dir_blocks,
                        'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
                    Q_hat_f[:,iBlk] = np.load(file)
            else:
                None;
            Q_hat_f = comm.bcast(Q_hat_f,root=0);
            comm.Barrier();
            # compute standard spod
            self.compute_standard_spod(Q_hat_f, iFreq)
            
        # store and save results
        self.store_and_save()

        # delete FFT blocks from memory if saving not required
        if self._savefft == False and rank== 0:
            for iBlk in range(0,self._n_blocks):
                for iFreq in range(0,self._n_freq):
                    file = os.path.join(self._save_dir_blocks,
                        'fft_block{:04d}_freq{:04d}.npy'.format(iBlk,iFreq))
                    os.remove(file)
        else:
            None;
                    
        if rank == 0:
            print('------------------------------------',flush=True)
            print(' ',flush=True)
            print('Results saved in folder ', self._save_dir_blocks,flush=True)          
        else:
            None;
            
        return self
