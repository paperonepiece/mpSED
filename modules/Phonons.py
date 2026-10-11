import numpy as np
from scipy.fft import fft
import Lattice 

class spectral_energy_density:
    def __init__(self,params):
        print('Phonon Spectral Energy Density (SED) Calculation...', flush=True)
        self.g = 1e-3
        self.mol = 6.02214129e23
        self.ev  = 1.602176565e-19
        self.ps  = 1e-12 # s
        self.vc  = 100   # A/ps to m/s
        if params.md_units == 'real':
            self.vc = 1000*100  # A/fs to m/s
            
        self.reduced_frames = params.num_steps // params.stride
        self.frames_per_split = self.reduced_frames // params.num_splits
        
        # My peronsalized setting
        # It helps to add more frames in each split, e.g., loop 1e5 with 4 times in 2e5
        self.move_frames = self.frames_per_split
        
        if self.frames_per_split != params.steps_per_split // params.stride:
            if params.steps_per_split // params.stride > self.frames_per_split:
                self.frames_per_split = params.steps_per_split // params.stride
                self.move_frames = (self.reduced_frames - self.frames_per_split) // params.num_splits
            else:
                print('\nCAUTION: User set {} steps_per_split is not enough for {} splits, auto-change to {}'.format(params.steps_per_split, 
                                                                                                          params.num_splits,
                                                                                                          self.frames_per_split * params.stride), flush=True)             
                self.frames_per_split = self.reduced_frames // params.num_splits
      
        self.thz = (np.arange(self.frames_per_split)/
                   (self.frames_per_split*params.time_step*params.stride)*self.ps)
        # self.simulation_time = params.time_step*params.num_steps*1e12/params.num_splits # ps
        self.simulation_time = params.time_step*self.frames_per_split*params.stride/self.ps # s to ps

    def _check_cell_vector(self, 
                           lattice
                           ):
        '''
        Check if the cell-vector from basis atom is correct
        '''
        fractional_cells = (
            self.cell_vecs @ np.linalg.inv(lattice.dir_lat)
        )
        nx, ny, nz = map(int, np.diagonal(lattice.supercell_matrix))
        expected_indices = np.array(
            [
                [k, j, i]
                for i in range(nz)
                for j in range(ny)
                for k in range(nx)
            ],
            dtype=float,
        )
        # print(fractional_cells[:5], expected_indices[:5])
        if not np.allclose(
            fractional_cells,
            expected_indices,
            atol=1.0e-10,
        ):
            raise ValueError(
                "\nERROR: cell_vecs is not the expected primitive-cell translation grid\n"
            )
            
    def _choose_q_batch(
        self,
        num_qpoints,
        num_frames,
        num_basis,
        memory_budget_gb=1.0,
        safety_factor=1.5
        ):
        """
        Estimate vectorized Q-point workspace and choose a batch size.

        Small workloads process all Q points at once. Large workloads are
        split so estimated temporary arrays stay within the memory budget.
        """
        budget_bytes = int(memory_budget_gb * 1024**3)

        # Per Q point, the main arrays are approximately:
        #   mode_vel: complex128 (T, Q, B, 3)
        #   spectrum: complex128 (T, Q, B, 3)
        #   abs(spectrum)**2: float64 (T, Q, B, 3)
        bytes_per_qpoint = (
            num_frames
            * num_basis
            * 3
            * (16 + 16 + 8)
        )

        estimated_full_bytes = int(
            bytes_per_qpoint * num_qpoints * safety_factor
        )

        if estimated_full_bytes <= budget_bytes:
            return num_qpoints, estimated_full_bytes

        q_batch = max(
            1,
            min(
                num_qpoints,
                budget_bytes // max(
                    1,
                    int(bytes_per_qpoint * safety_factor)
                )
            )
        )

        estimated_batch_bytes = int(
            bytes_per_qpoint * q_batch * safety_factor
        )

        return q_batch, estimated_batch_bytes
                
    def compute_sed(self, params, lattice, eigen_vectors):
        # note that velocites are in Ang/ps

        print('\nCAUTION: the FFT\'s have been scaled done, and the SED units is converted to J*s!',flush=True)  

        self.num_unit_cells = lattice.unit_cells.max()
        self.num_basis = lattice.basis_pos.max()
        if params.debug:
            self.num_loops = 1
        else:
            self.num_loops = params.num_splits   
        if params.with_eigs: # do the calculation with eigenvectors
            if self.num_basis != eigen_vectors.natom:
                print('\nERROR: number of basis atoms doesn\'t match the PHONOPY file!\n',flush=True)
                exit()
            self.sed_bands = np.zeros((params.num_splits,
                            eigen_vectors.natom*3,
                            self.frames_per_split,
                            sum(params.num_qpoints)))
            self.split_loop_with_eigs(params, lattice, eigen_vectors)
            self.sed_bands_avg = self.sed_bands.sum(axis=0) / self.num_loops
            self.sed_avg = self.sed_bands_avg.sum(axis=0)
            max_freq = len(self.thz) // 2
            self.sed_bands_avg = self.sed_bands_avg[:,:max_freq,:]
            self.sed_avg = self.sed_avg[:max_freq,:]
            self.thz = self.thz[:max_freq]
            
        else: # do the calculation without eigenvectors
            self.sed = np.zeros((params.num_splits,
                            self.frames_per_split,
                            sum(lattice.num_qpoints)))
            self.loop_over_splits(params, lattice)
            self.sed_avg = self.sed.sum(axis=0) / self.num_loops
            max_freq = len(self.thz) // 2
            self.sed_avg = self.sed_avg[:max_freq,:]
            self.thz = self.thz[:max_freq]

    ###############################################################
    ### without eigs
    def loop_over_splits(self, 
                         params, 
                         lattice):
        '''
        original code implementation seems incorrect since qdot is re-accumulating among splits
        which makes split-average meaningless because the last split contains all-splits averaged result!
        '''
        # self.qdot = np.zeros((self.frames_per_split, sum(lattice.num_qpoints)))
        # for i in range(self.num_loops):
        #     print('\nNow on split {} out {}...\n'.format(i+1, self.num_loops),flush=True)
        #     self.loop_index = i
        #     self.get_simulation_data(params, lattice) 
        #     self.loop_over_qpoints(params, lattice) 
        #     self.sed[i,:,:] = self.qdot/(4*np.pi*self.simulation_time*self.ps*self.num_unit_cells) # ps--s # scale to J*s
        
        # speed-up code implementation
        for i in range(self.num_loops):

            self.qdot = np.zeros((self.frames_per_split, sum(lattice.num_qpoints)))
            
            print('\nNow on split {} out {}...\n'.format(i+1, self.num_loops),flush=True)
            self.loop_index = i
            self.get_simulation_data(params, lattice) 
            
            # cell vectors test
            self._check_cell_vector(lattice)
                
            self.loop_over_qpoints_vectorized(params, lattice)
            self.sed[i,:,:] = self.qdot/(4*np.pi*self.simulation_time*self.ps*self.num_unit_cells) # ps--s # scale to J*s            
        
    def loop_over_qpoints(self, 
                          params, 
                          lattice):
        for q in range(sum(lattice.num_qpoints)):
            self.q_index = q
            print('\tNow on q-point {} out of {}:\t'
                  'q=({:.3f}, {:.3f}, {:.3f})'.format(q+1,
                                                      sum(lattice.num_qpoints),
                                                      lattice.qpoints[q,0],
                                                      lattice.qpoints[q,1],
                                                      lattice.qpoints[q,2]),
                  flush=True)
            
            self.exp_fac = np.tile(lattice.qpoints[q, :], (self.num_unit_cells, 1))
            self.exp_fac = np.exp(1j * np.multiply(self.exp_fac, 
                                                   self.cell_vecs).sum(axis=1))
            self.loop_over_basis(params,lattice)
    
    def loop_over_basis(self, 
                            params, 
                            lattice):
            for i in range(self.num_basis):
                basis_ids = np.argwhere(lattice.basis_pos == (i+1)).reshape(
                        self.num_unit_cells)
                mass = lattice.masses[i]
                vx = np.fft.fft(self.vels[:,basis_ids,0].reshape(self.frames_per_split, self.num_unit_cells)*self.exp_fac, axis=0) # scale
                vy = np.fft.fft(self.vels[:,basis_ids,1].reshape(self.frames_per_split, self.num_unit_cells)*self.exp_fac, axis=0) # scale
                vz = np.fft.fft(self.vels[:,basis_ids,2].reshape(self.frames_per_split, self.num_unit_cells)*self.exp_fac, axis=0) # scale
                self.qdot[:,self.q_index] = (self.qdot[:,self.q_index]+ \
                                            (abs(vx.sum(axis=1))**2 + \
                                             abs(vy.sum(axis=1))**2 + \
                                             abs(vz.sum(axis=1))**2)* \
                                            (mass*self.g/self.mol)) # *params.time_step) # scale
        
    def loop_over_qpoints_vectorized(self, 
                                     params, 
                                     lattice,
                                     memory_budget_gb=2.0):
        """
        Vectorized non-eigenvector SED calculation.

        Shapes:
            vels:       (T, B, C, 3)
            phase:      (Q, C)
            mode_vel:   (T, Q, B, 3)
            spectrum:   (T, Q, B, 3)
        """
        num_qpoints = sum(lattice.num_qpoints)
        num_basis = self.num_basis
        num_cells = self.num_unit_cells
        
        q_batch, estimated_workspace = \
            self._choose_q_batch(
            num_qpoints=num_qpoints,
            num_frames=self.frames_per_split,
            num_basis=num_basis,
            memory_budget_gb=memory_budget_gb
        )
        if q_batch == num_qpoints:
            print(
                'Vectorized Q calculation: SMALL, no batching '
                '(estimated workspace {:.2f} GiB)'.format(
                    estimated_workspace / 1024**3
                ),
                flush=True
            )
        else:
            print(
                'Vectorized Q calculation: LARGE, auto-set batch size {}, '
                'estimated workspace per batch {:.2f} GiB'.format(
                    q_batch,
                    estimated_workspace / 1024**3
                ),
                flush=True
            )
    
        # Build basis -> cell atom mapping only once per split.
        basis_ids = np.empty((num_basis, num_cells), dtype=np.intp)

        for basis_index in range(num_basis):
            ids = np.argwhere(lattice.basis_pos == (basis_index + 1)).reshape(num_cells)
            basis_ids[basis_index, :] = ids
            
        # Advanced indexing produces:
        # (frames, basis, cells, Cartesian component)
        velocities = self.vels[:, basis_ids, :]
        
        all_masses = np.asarray(lattice.masses, dtype=float)
        basis_masses = all_masses[basis_ids[:, 0]] # basis atom 1
        
        # Determine the mass of basis atoms
        # if all_masses.size == num_basis:
        #     # lattice.masses already stores one mass per basis atom.
        #     basis_masses = all_masses

        # elif all_masses.size == self.vels.shape[1]:
        #     # lattice.masses stores one mass per atom in the full supercell.
        #     # basis_ids has shape (num_basis, num_unit_cells).
        #     atom_masses = all_masses[basis_ids]
            
        #     # print('atom_masses',atom_masses.shape,all_masses.shape)
        #     # print(basis_ids,basis_ids.shape)
            
        #     # Every periodic copy of the same basis atom should have the
        #     # same mass.
        #     if not np.allclose(
        #         atom_masses,
        #         atom_masses[:, :1],
        #         rtol=1e-12,
        #         atol=1e-12
        #     ):
        #         raise ValueError(
        #             'Masses of periodic copies differ for at least one '
        #             'basis atom.'
        #         )

        #     basis_masses = atom_masses[:, 0]

        # else:
        #     raise ValueError(
        #         'Unexpected number of masses: got {}, expected either '
        #         'num_basis={} or num_atoms={}.'.format(
        #             all_masses.size,
        #             num_basis,
        #             self.vels.shape[1]
        #         )
        #     )

        mass_factor = basis_masses * self.g / self.mol
        
        print()
        
        for q_start in range(0, num_qpoints, q_batch):
            q_end = min(q_start + q_batch, num_qpoints)
            q_cart = lattice.qpoints[q_start:q_end, :]
            
            # print q info
            for q in range(len(lattice.qpoints[q_start:q_end, :])):
                print('\tNow on q-point {} out of {}:\t'
                                'q=({:.3f}, {:.3f}, {:.3f}) '
                                '(cart.)'.format(q+1,
                                                        sum(lattice.num_qpoints),
                                                        lattice.qpoints[q_start+q,0],
                                                        lattice.qpoints[q_start+q,1],
                                                        lattice.qpoints[q_start+q,2]),
                                flush=True)
                
            # lattice.qpoints are Cartesian reciprocal vectors in the current code.
            # cell_vecs has shape (cells, 3).
            
            # print("qpoints shape:", q_cart.shape)
            # print("cell_vecs shape:", self.cell_vecs.shape)
            # print("q sample:", q_cart[:3])
            # print("R sample:", self.cell_vecs[:3])
            
            # phase_argument = (
            #     q_cart @ self.cell_vecs.T
            # )

            # phase = np.exp(
            #     1j * phase_argument
            # )

            # print(
            #     "phase argument min/max:",
            #     phase_argument.min(),
            #     phase_argument.max(),
            # )

            # print(
            #     "phase magnitude min/max:",
            #     np.abs(phase).min(),
            #     np.abs(phase).max(),
            # )
            
            phase = np.exp(
                1j * np.matmul(
                    q_cart,
                    self.cell_vecs.T
                )
            )
            
            # phase:      (Q, C)
            # velocities: (T, B, C, 3)
            # mode_vel:   (T, Q, B, 3)
        
            mode_vel = np.einsum(
                'qc,tbcd->tqbd',
                phase,
                velocities,
                optimize=True
            )
        
            # The original code performs FFT independently for each basis and
            # Cartesian component. This is the same operation as a batched FFT.
           
            # spectrum = np.fft.fft(mode_vel, axis=0)
            
            spectrum = fft(mode_vel, axis=0, workers=params.num_cores)

            self.qdot[:, q_start:q_end] += np.sum(
                np.abs(spectrum) ** 2
                * mass_factor[None, None, :, None],
                axis=(2, 3)
            ) 
                      
    ###############################################################
    ### with eigs
    def split_loop_with_eigs(self, 
                             params, 
                             lattice, 
                             eigen_vectors):
        for i in range(self.num_loops):
            print('\nNow on split {} out {}...\n'.format(i+1,self.num_loops),flush=True)
            self.loop_index = i
            self.get_simulation_data(params,lattice)
            self.qpoint_loop_with_eigs(params,lattice,eigen_vectors)
    
    def qpoint_loop_with_eigs(self, 
                              params, 
                              lattice, 
                              eigen_vectors):
        for q in range(sum(lattice.num_qpoints)):
            self.q_index = q
            print('\n\tNow on q-point {} out of {}:\tq=({:.3f}, {:.3f}, {:.3f})'
                    .format(q+1,sum(lattice.num_qpoints),lattice.qpoints[q,0],
                        lattice.qpoints[q,1],lattice.qpoints[q,2]),flush=True)
            self.exp_fac = np.tile(lattice.qpoints[q,:],(self.num_unit_cells,1))
            self.exp_fac = np.exp(-1j*np.multiply(self.exp_fac, self.cell_vecs).sum(axis=1))
            self.band_loop_with_eigs(params,lattice,eigen_vectors)
    
    def band_loop_with_eigs(self, 
                            params, 
                            lattice, 
                            eigen_vectors):
        for b in range(eigen_vectors.natom*3): # loop over bands
            print('\t\tband {} out of {}'.format(b+1,self.num_basis*3),flush=True)
            self.qdot = np.zeros(self.frames_per_split).astype(complex)
            self.band_index = b
            self.ex = eigen_vectors.eig_vecs[self.q_index,b,:,0]
            self.ey = eigen_vectors.eig_vecs[self.q_index,b,:,1]
            self.ez = eigen_vectors.eig_vecs[self.q_index,b,:,2]
            self.basis_loop_with_eigs(params,lattice)
            self.sed_bands[self.loop_index,b,:,self.q_index] = abs(
                    np.fft.fft(self.qdot))**2*params.time_step/self.ps # !!!
    
    def basis_loop_with_eigs(self, 
                             params, 
                             lattice):
        for n in range(self.num_basis):
            basis_ids = np.argwhere(lattice.basis_pos == (n+1)).reshape(
                    self.num_unit_cells)
            mass = lattice.masses[n]
            vx = ((self.exp_fac*self.vels[:,basis_ids,0]
                    .reshape(self.frames_per_split,self.num_unit_cells))
                    .sum(axis=1)) 
            vy = ((self.exp_fac*self.vels[:,basis_ids,1]
                    .reshape(self.frames_per_split,self.num_unit_cells))
                    .sum(axis=1)) 
            vz = ((self.exp_fac*self.vels[:,basis_ids,2]
                    .reshape(self.frames_per_split,self.num_unit_cells))
                    .sum(axis=1))
            self.qdot[:] = (self.qdot[:]+(vx*self.ex[n].conj()+vy*self.ey[n].conj()+
                vz*self.ez[n].conj())/self.num_unit_cells*mass) # !!!

    ##################################################################
    ### read vels and pos from the hdf5 file
    def get_simulation_data(self, params, lattice):
        # if self.frames_cross_chunks == True:
        self.vels = params.database['vels'][self.loop_index*self.move_frames:
                                            (self.frames_per_split + self.loop_index*(self.move_frames)),:,:] * self.vc # scale to m/s
        self.pos = params.database['pos'][self.loop_index*self.move_frames:
                                            (self.frames_per_split + self.loop_index*(self.move_frames)),:,:]
        # else:
        # self.vels = params.database['vels'][self.loop_index*self.frames_per_split:
        #                                     (self.loop_index+1)*self.frames_per_split,:,:] * self.vc # scale to m/s
        # self.pos = params.database['pos'][self.loop_index*self.frames_per_split:
        #                                     (self.loop_index+1)*self.frames_per_split,:,:]

        # time average the positions (for now, maybe can do corr. between pos and vels)
        
        # self.cell_vecs = self.pos[:, lattice.cell_ref_ids, :].mean(axis=0) 
        self.cell_vecs = Lattice.build_cell_translation_vectors(lattice)

class dynamic_structure_factor:
    def __init__(self, lattice):
        print('\nDynamic Structure Factor (DSF) Calculation...\n')
        self._validate_inputs()
        
    def _validate_inputs(self):
        print('\nPreparation for DSF Done!\n')
    
    def compute_dsf(self, lattice):
        pass
    
def validate_lattice_structure(lattice, num_atoms=None):
    unit_cells = np.asarray(
        lattice.unit_cells,
        dtype=int
    ).reshape(-1)

    basis_pos = np.asarray(
        lattice.basis_pos,
        dtype=int
    ).reshape(-1)

    masses = np.asarray(
        lattice.masses,
        dtype=float
    ).reshape(-1)

    if num_atoms is not None:
        if len(unit_cells) != num_atoms:
            raise ValueError(
                'unit_cells contains {} atoms, but velocities contain {}'.format(
                    len(unit_cells),
                    num_atoms
                )
            )

    if len(unit_cells) != len(basis_pos):
        raise ValueError(
            'unit_cells and basis_pos have different lengths'
        )

    if len(unit_cells) != len(masses):
        raise ValueError(
            'unit_cells and masses have different lengths'
        )

    cell_ids = np.unique(unit_cells)
    basis_ids = np.unique(basis_pos)

    if len(cell_ids) == 0:
        raise ValueError('No unit cell was read from lattice.dat')

    if len(basis_ids) == 0:
        raise ValueError('No basis atom was read from lattice.dat')

    expected_basis = set(
        range(1, int(basis_ids.max()) + 1)
    )

    atoms_per_cell = []

    for cell_id in cell_ids:
        mask = unit_cells == cell_id
        cell_basis = set(basis_pos[mask].tolist())

        atoms_per_cell.append(np.sum(mask))

        if cell_basis != expected_basis:
            raise ValueError(
                'Cell {} has basis {}, expected {}'.format(
                    cell_id,
                    sorted(cell_basis),
                    sorted(expected_basis)
                )
            )

    unique_atoms_per_cell = np.unique(atoms_per_cell)

    if len(unique_atoms_per_cell) != 1:
        raise ValueError(
            'Different cells contain different numbers of atoms: {}'.format(
                unique_atoms_per_cell
            )
        )

    print(
        'lattice.dat validation passed:',
        'num_atoms = {}, num_cells = {}, num_basis = {}, '
        'atoms_per_cell = {}'.format(
            len(unit_cells),
            len(cell_ids),
            len(basis_ids),
            unique_atoms_per_cell[0]
        ),
        flush=True
    )

    return cell_ids, basis_ids

def build_basis_ids(lattice):
    unit_cells = np.asarray(lattice.unit_cells).reshape(-1)
    basis_pos = np.asarray(lattice.basis_pos).reshape(-1)

    cell_labels = np.unique(unit_cells)
    cell_labels.sort()

    num_cells = len(cell_labels)
    num_basis = int(np.max(basis_pos))

    basis_ids = np.full(
        (num_basis, num_cells),
        -1,
        dtype=np.intp
    )

    cell_to_index = {
        int(cell_label): cell_index
        for cell_index, cell_label in enumerate(cell_labels)
    }

    for atom_index, (cell_label, basis_index) in enumerate(
        zip(unit_cells, basis_pos)
    ):
        cell_index = cell_to_index[int(cell_label)]
        basis_index = int(basis_index) - 1

        if basis_index < 0 or basis_index >= num_basis:
            raise ValueError(
                'Invalid basis index {} at atom {}'.format(
                    basis_index + 1,
                    atom_index
                )
            )

        if basis_ids[basis_index, cell_index] != -1:
            raise ValueError(
                'Duplicate atom for cell {}, basis {}'.format(
                    cell_label,
                    basis_index + 1
                )
            )

        basis_ids[basis_index, cell_index] = atom_index

    if np.any(basis_ids < 0):
        missing = np.argwhere(basis_ids < 0)[0]
        raise ValueError(
            'Missing atom for basis {}, cell index {}'.format(
                int(missing[0]) + 1,
                int(missing[1])
            )
        )

    return basis_ids, cell_labels
