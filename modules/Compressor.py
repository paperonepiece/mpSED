import h5py 
import numpy as np
import os

def compress(params):
    # check if files exist
    if not os.path.exists(params.pos_vels_file):
                    print('\nERROR: file {} not found\n'.format(params.vels_file),flush=True)
                    exit()
    #if not os.path.exists(params.pos_file):
    #                print('\nERROR: file {} not found\n'.format(params.pos_file),flush=True)
    #                exit()

    print('\nCompressing velocity and position data into .hdf5 database\n'
            '\nThis may take a while...\n',flush=True)

    if params.file_format == 'xyz':

        print('\nDEV NOTE: before this routine can be used with \'xyz\', it needs to be\n'
                'modified such that it only saves the steps selected by the stride\n',flush=True)
        exit() # dev

        # write to hdf5 database file
        with h5py.File(params.database_file,'w') as fout:
            with open(params.vels_file,'r') as fin: # velocities
                vels_dset = fout.create_dataset('vels',
                        (params.num_steps,params.num_atoms,3))
                # look them up in .xyz file
                for i in range(params.num_steps):
                    vels = np.zeros((params.num_atoms,3))
                    for j in range(2):
                       fin.readline()
                    for j in range(params.num_atoms):
                       vels[j,:] = fin.readline().strip().split()[1:]
                    vels_dset[i,:,:] = vels

            with open(params.pos_file,'r') as fin: # positions
                pos_dset = fout.create_dataset('pos',
                        (params.num_steps,params.num_atoms,3))
                # look them up in .xyz file
                for i in range(params.num_steps):
                    pos = np.zeros((params.num_atoms,3))
                    for j in range(2):
                        fin.readline()
                    for j in range(params.num_atoms):
                       pos[j,:] = fin.readline().strip().split()[1:]
                    pos_dset[i,:,:] = pos                   

    elif params.file_format == 'lammps':
        # write to hdf5 database
        num_frames = params.num_steps//params.stride # number of times actually printed
        vels = np.zeros((params.num_atoms, 3))
        pos  = np.zeros((params.num_atoms, 3))
        with h5py.File(params.database_file,'w') as fout:
            vels_dset = fout.create_dataset('vels',(num_frames, params.num_atoms, 3))
            pos_dset  = fout.create_dataset('pos',(num_frames, params.num_atoms, 3))
            with open(params.pos_vels_file,'r') as fin: # vels
                # look them up in lammps output file
                for i in range(num_frames):
                    # for j in range(9):
                    #     fin.readline()
                    # for j in range(params.num_atoms):
                    #     line = fin.readline()
                    #     pos[j,:]  = line.strip().split()[2:5]
                    #     vels[j,:] = line.strip().split()[5:] 
                    # vels_dset[i,:,:] = vels
                    # pos_dset[i,:,:]  = pos
                      
                    for _ in range(9):
                        if not fin.readline():
                            raise EOFError(
                                f"Dump file ended before frame {i + 1}/{num_frames}."
                            )

                    data = np.loadtxt(
                        fin,
                        dtype=np.float64,
                        usecols=(2, 3, 4, 5, 6, 7),
                        max_rows=params.num_atoms,
                        ndmin=2,
                    )

                    if data.shape != (params.num_atoms, 6):
                        raise ValueError(
                            "Frame {}: expected {} atom rows, "
                            "but read array with shape {}.".format(i,
                                                                   params.num_atoms,
                                                                   data.shape)
                        )

                    pos_dset[i] = data[:, :3]
                    vels_dset[i] = data[:, 3:]
                    
                    if i / num_frames % 0.2 == 0.0:
                        print('Compression finished {:d}%'.format(int(i/num_frames*100)),flush=True)
                                         
                
#             with open(params.pos_vels_file,'r') as fin: # pos
#                 pos_dset  = fout.create_dataset('pos',(num_steps,params.num_atoms,3))
#                 for i in range(num_steps):
#                     pos  = np.zeros((params.num_atoms,3))
#                     for j in range(9):
#                         fin.readline()
#                     for j in range(params.num_atoms):
#                        pos[j,:]  = fin.readline().strip().split()[2:5]
#                     pos_dset[i,:,:]  = pos

    print('Done compressing {} into .hdf5 format.\n\n'
         'The compressed file is \'{}\' (DON\'T CHANGE IT!)\n\n'
         'Set COMPRESS = 0 in the input file and run the code again to continue'
         .format(params.pos_vels_file,params.database_file),flush=True)


