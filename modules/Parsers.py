
import os
import h5py
import numpy as np
import yaml
from yaml import CLoader as Loader

def print_error(txt):
    print('\nERROR: {} parameter seems wrong\n'.format(txt),flush=True)
    exit()

class parse_input:
    def __init__(self,input_file):
        ### defaults
        self.compress = False
        self.debug = False
        self.plot_previous = False
        self.plot_slice = False
        self.with_eigs = False
        self.num_basis = 0
        self.basis_list = []
        self.q_slice_index = 0
        self.num_bins = 1
        self.out_prefix = 'lammps-run'
        self.pos_vels_file = 'pos_vels.dat'
        self.lattice_file = 'lattice.dat'
        self.database_file = 'dat.hdf5'
        self.file_format = 'xyz'
        self.md_units  = 'metal'
        self.plot_lorentz = False
        self.input_file = input_file
        self.plot_cut = '0'
        self.steps_per_split = 0
        self.num_qpoints = None
        self.num_cores = 1
        self.do_dsf = False
        self.lorentz_fit_mode = 'manual'
        # input_txt = open(input_file,'r').readlines()
        
        with open(input_file, 'r', encoding='utf-8') as fr:
            input_txt = fr.readlines()
            
        for line in input_txt:
            txt = line.strip()
            # skip blank and comment lines     
            if '#' in txt:
                txt = txt.split('#')[0]
            txt = txt.strip().split()
            if len(txt) == 0: 
                continue
            
            # crystal info
            if txt[0] == 'LAT_PARAMS':
                try:
                    self.lat_params = np.array(txt[(txt.index('=')+1):
                                            (txt.index('=')+4)]).astype(float)
                except:
                    print_error('LAT_PARAMS')
            elif txt[0] == 'PRIM_VECS':
                try:
                    self.prim_vecs = np.array(txt[(txt.index('=')+1):
                        (txt.index('=')+10)]).astype(float)
                    self.prim_vecs = self.prim_vecs.reshape(3,3)
                except:
                    print_error('PRIM_VECS')
            elif txt[0] == 'SUPERCELL_DIM':
                try:
                    self.supercell_dim = np.array(txt[(txt.index('=')+1):
                        (txt.index('=')+4)]).astype(int)
                    if self.supercell_dim.shape != (3,) or np.any(self.supercell_dim <= 0):
                        raise ValueError('\nERROR: SUPERCELL_DIM must contain three positive integers\n')
                except:
                    print_error('SUPERCELL_DIM')
            
            # Q-points
            elif txt[0] == 'NUM_QPATHS':
                try:
                    self.num_qpaths = int(txt[txt.index('=')+1])
                except:
                    print_error('NUM_QPATHS')
            elif txt[0] == 'NUM_QPOINTS':
                try:
                    self.num_qpoints = np.array(txt[(txt.index('=')+1):
                        (txt.index('=')+1+self.num_qpaths)]).astype(int)
                except:
                    print_error('NUM_QPOINTS')
            elif txt[0] == 'QSYM_POINTS':
                # (h, k, l): cofficients in reciprocal vector (b1,b2,b3)
                try:
                    self.qsym_points = np.array(txt[(txt.index('=')+1):
                        (txt.index('=')+1+int((self.num_qpaths+1)*3))]).astype(float)
                    self.qsym_points = self.qsym_points.reshape(self.num_qpaths+1,3)
                except:
                    print_error('QSYM_POINTS')

            # simulation control parameters
            elif txt[0] == 'NUM_ATOMS':
                try:
                    self.num_atoms = int(txt[txt.index('=')+1])
                except:
                    print_error('NUM_ATOMS')
            elif txt[0] == 'NUM_STEPS':
                try:
                    self.num_steps = int(eval(txt[txt.index('=')+1]))
                except:
                    print_error('NUM_STEPS')
            elif txt[0] == 'TIME_STEP':
                try:
                    self.time_step = float(txt[txt.index('=')+1])
                except:
                    print_error('TIME_STEP')
                self.time_step = self.time_step*1e-15  # fs to s
            elif txt[0] == 'MD_UNITS':
                try:
                    self.md_units = str(txt[txt.index('=')+1].strip('"').strip("'"))
                except:
                    print_error('MD_UNITS')
                else:
                    if self.md_units not in ['metal', 'real']:
                        print_error('MD_UNITS')
            elif txt[0] == 'IF_DSF':
                try:
                    self.do_dsf = bool(int(txt[txt.index('=')+1]))
                except:
                    print_error('IF_DSF')
            elif txt[0] == 'STRIDE':
                try:
                    self.stride = int(eval(txt[txt.index('=')+1]))
                except:
                    print_error('STRIDE')

            # data averaging control parameters
            elif txt[0] == 'NUM_SPLITS':
                try:
                    self.num_splits = int(txt[txt.index('=')+1])
                except:
                    print_error('NUM_SPLITS')
            elif txt[0] == 'STEPS_PER_SPLITS':
                try:
                    self.steps_per_split = int(eval(txt[txt.index('=')+1]))
                except:
                    print_error('STEPS_PER_SPLITS')
            elif txt[0] == 'NUM_CORES':
                try:
                    self.num_cores = int(eval(txt[txt.index('=')+1]))
                except:
                    print_error('NUM_CORES')
                else:
                    if self.num_cores < 1:
                        print_error('NUM_CORES(should be a positive integer)')
                                
            # whether or not to build hdf5 database
            elif txt[0] == 'COMPRESS':
                try:
                    self.compress = int(txt[txt.index('=')+1])
                except:
                    print_error('COMPRESS')

            # options
            elif txt[0] == 'DEBUG':
                try:
                    self.debug = bool(int(txt[txt.index('=')+1]))
                except:
                    print_error('DEBUG')
            elif txt[0] == 'WITH_EIGS':
                try:
                    self.with_eigs = bool(int(txt[txt.index('=')+1]))
                except:
                    print_error('WITH_EIGS')
            elif txt[0] == 'NUM_BASIS':
                try:
                    self.num_basis = (int(txt[txt.index('=')+1]))
                except:
                    print_error('NUM_BASIS')
            elif txt[0] == 'BASIS_LIST':
                try:
                    self.basis_list = list(map(int,txt[txt.index('=')+1:
                        txt.index('=')+1+self.num_basis]))
                except:
                    print_error('BASIS_LIST')

            # plotting
            elif txt[0] == 'PLOT_BANDS':
                try:
                    self.plot_bands = bool(int(txt[txt.index('=')+1]))
                except:
                    print_error('PLOT_BANDS')
            elif txt[0] == 'BAND_TO_PLOT':
                try:
                    self.band_to_plot = (int(txt[txt.index('=')+1]))
                except:
                    print_error('BAND_TO_PLOT')
            elif txt[0] == 'PLOT_SLICE':
                try:
                    self.plot_slice = bool(int(txt[txt.index('=')+1]))
                except:
                    print_error('PLOT_SLICE')
            elif txt[0] == 'Q_SLICE_INDEX':
                try:
                    self.q_slice_index = list(map(int, txt[txt.index('=')+1:]))
                except ValueError:
                    self.q_slice_index = eval(txt[txt.index('=')+1])
                    if type(self.q_slice_index) != range:
                        print_error('Q_SLICE_INDEX')
            elif txt[0] == 'NUM_BINS':
                try:
                    self.num_bins = int(txt[txt.index('=')+1]) 
                except:
                    print_error('NUM_BINS')
            elif txt[0] == 'LORENTZ':
                try:
                    self.lorentz = bool(int(txt[txt.index('=')+1]))
                except:
                    print_error('LORENTZ')
            elif txt[0] == 'FIT_MODE':
                try:
                    self.lorentz_fit_mode = txt[txt.index('=')+1].strip("'").strip('"')
                    if not self.lorentz_fit_mode in ['auto', 'manual']:
                        print_error('FIT_MODE(TypeError, not supported fitting type "{}")'.format(self.lorentz_fit_mode))
                except:
                    print_error('FIT_MODE')
            elif txt[0] == 'NUM_GUESSES':
                try:
                    self.num_guesses = int(txt[txt.index('=')+1])
                except:
                    print_error('NUM_GUESSES')
            elif txt[0] == 'PEAK_GUESSES':
                try:
                    self.peak_guesses = list(map(int,txt[txt.index('=')+1:
                        txt.index('=')+1+self.num_guesses]))
                except:
                    print_error('PEAK_GUESSES')

            # file names
            elif txt[0] == 'FILE_FORMAT':
                try:
                    self.file_format = str(txt[txt.index('=')+1].strip("'").strip('"'))
                except:
                    print_error('FILE_FORMAT')
            elif txt[0] == 'POS_VELS_FILE':
                try:
                    self.pos_vels_file = str(txt[txt.index('=')+1].strip("'").strip('"'))
                except:
                    print_error('POS_VELS_FILE')
            elif txt[0] == 'LATTICE_FILE':
                try:
                    self.lattice_file = str(txt[txt.index('=')+1].strip("'").strip('"'))
                except:
                    print_error('LATTICE_FILE')
#                if not os.path.exists(self.lattice_file):
#                    print('\nERROR: file {} not found\n'.format(self.lattice_file))
#                    exit()
            elif txt[0] == 'OUT_PREFIX':
                try:
                    self.out_prefix = str(txt[txt.index('=')+1].strip("'").strip('"'))
                except:
                    print_error('OUT_PREFIX')
#            elif txt[0] == 'EIGVECS_FILE':
#                try:
#                    self.eigvecs_file = str(txt[txt.index('=')+1].strip('\''))
#                except:
#                    print_error('EIGVECS_FILE')
#                if not os.path.exists(self.eigvecs_file):
#                    print('\nERROR: file {} not found\n'.format(self.eigvecs_file))
#                    self.with_eigs = False
#                    exit()
            # unknown options
            elif txt[0] == 'FREQ_CUTOFF':
                try:
                    self.plot_cut = str(txt[txt.index('=')+1].strip("'").strip('"'))
                except:
                    print_error('FREQ_CUTOFF')
            else: 
                print_error('{}(unsupported input_arg)'.format(txt[0]))
        
        # checking if set correctly
        if self.num_qpoints is None:
            # default num_qpoints is all zeros
             self.num_qpoints = np.zeros(self.num_qpaths)
        
        if len(self.num_qpoints) != self.num_qpaths:
            print_error('NUM_QPOINTS(inconsistent counting with NUM_QPATHS) input')


def parse_lattice_file(params):
    if not os.path.exists(params.lattice_file):
        print('\nERROR: file {} not found\n'.format(params.lattice_file),flush=True)
        exit()

    # read the lattice info from a file
    atom_ids, unit_cells, basis_pos, masses = np.loadtxt(
            params.lattice_file,
            unpack=True
    )

    # set proper data types
    atom_ids = atom_ids.astype(int)
    unit_cells = unit_cells.astype(int)
    basis_pos = basis_pos.astype(int)
    masses = masses.astype(float)

    return atom_ids, unit_cells, basis_pos, masses


class parse_eigen_vecs:
    def __init__(self,params):
        if params.with_eigs: # if WITH_EIGS = 1 in input file
            phonopy_data = yaml.load(open(params.eigvecs_file),Loader=Loader)
            
            # data from the phonopy output file
            self.natom = phonopy_data['natom']
            self.num_qpoints = phonopy_data['nqpoint']
            self.qpoints = np.zeros((self.num_qpoints,3))

            # incase the order of basis atoms in phonopy don't match
            if len(params.basis_list) == 0:
                basis_slice = np.arange(self.natom)
            else:
                basis_slice = sorted(params.basis_list)
                for i in range(len(basis_slice)):
                    basis_slice[i] = basis_slice[i]-1
        
            # dispersions from phonopy
            self.freq = np.zeros((self.num_qpoints,self.natom*3))
            self.eig_vecs = np.zeros((self.num_qpoints,self.natom*3,
                self.natom,3)).astype(complex)

            # look up the eigenvectors
            for i in range(self.num_qpoints):
                # loop over q-points
                self.qpoints[i,:] = phonopy_data['phonon'][i]['q-position']

                # loop over bands
                for j in range(self.natom*3):
                    self.freq[i][j] = phonopy_data['phonon'][i]['band'][j]['frequency']

                    # loop over basis atoms
                    for ind in basis_slice:
                            self.eig_vecs[i,j,ind,0] = (
                        phonopy_data['phonon'][i]['band'][j]['eigenvector'][ind][0][0]+
                        1j*phonopy_data['phonon'][i]['band'][j]['eigenvector'][ind][0][1]
                            )
                            self.eig_vecs[i,j,ind,1] = (
                        phonopy_data['phonon'][i]['band'][j]['eigenvector'][ind][1][0]+
                        1j*phonopy_data['phonon'][i]['band'][j]['eigenvector'][ind][1][1]
                            )
                            self.eig_vecs[i,j,ind,2] = (
                        phonopy_data['phonon'][i]['band'][j]['eigenvector'][ind][2][0]+
                        1j*phonopy_data['phonon'][i]['band'][j]['eigenvector'][ind][2][1]
                            )

        else: # intialize empty array 
            self.eig_vecs = []


