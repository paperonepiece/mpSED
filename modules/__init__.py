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
# print(' Welcome to use this module for SED calculation\n\n This pSED module was originally developed by Ty Sterling, University of Colorado Boulder, Dept. of Physics\n',flush=True)
# print(' Zheyi, from Dalian University of Technology, made a few modifications on it\n\n The other SED method based on eigenvector is not available in this module for now...',flush=True)
# print('=============================================================================================================',flush=True)
