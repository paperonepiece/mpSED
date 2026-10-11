import os,sys

root = os.path.dirname(os.path.abspath(__file__))
__all__ = []
for file in os.listdir(root):
    if '__init__' not in file and '.py' in file:
        __all__.append(file.split('.py')[0])
        
print('=============================================================================================================\n'+
      ' Welcome to use this module for phonon Spectral Energy Density calculation\n\n'+
      ' It is called pSED and was originally developed by Ty Sterling, Uni. of Colorado Boulder, Dept. of Physics\n\n'+
      ' A few modifications on it been made to improve its performance and universality\n\n'+
      ' But the second SED method based on eigenvector is still not available now...\n'+
      '=============================================================================================================',
flush=True)
