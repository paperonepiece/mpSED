# mpSED
Modified Phonon Spectral Energy Density Calculation&Processing Module

Original SED code is from https://github.com/tyst3273/phonon-sed, Ty Sterling. I made a few modifications on it to improve its performance (Zheyi Zhang, zhangzy3033@163.com)

**New Features:**

    1. Compression Speed x3

    2. SED calculation Speed x20

    3. Allow more steps(data frames) per split than NUM_STEPS/NUM_SPLITS to get finer average

    4. Auto-detecting available q-points on q-path from the dimension of unit-cell replication

    5. Automatic mode for Lorentz fitting, i.e., the generation of phonon lifetime
   
    6. Assign number of cores used for fft calculation to speed up SED further (slightly)

The other settings and functions almost remains the same to Styling's original version

**Need:**

    1. python
  
    2. numpy
  
    3. scipy
    
    4. hdf5
    
    5. matplotlib

**Usage Instruction:**

  1. Copy the modules/mpSED/psed.in/replicate_unitcell.py/in_vels.lmp to your desired work directory

  2. Prepare your unitcell(lammps-format) and generate supercell structure using 'replicate_unitcell.py':

      python replicate_unitcell.py

  3. Run lammps-MD simulation with 'in_vels.lmp' in your familiar way

  4. Modify parameters in 'psed.in' according to your lammps-MD
  
  5. Run 'mpSED' to do SED calculation (Set the plot and lorentz args as 0):
      python mpSED

  6. Reset the plot and lorentz args and run again to do postprocessing, e.g., phonon-band plotting or lifetime fitting


ALL DONE! Hope it helps your work and enjoy your SED analyzation ~
