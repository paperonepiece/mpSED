# -*- coding:utf-8 -*-
'''
This code is used to generate supercell for the given structure:

python3 cellexp.py -i filename -d '2 2 2'
python3 cellexp.py -i /base/path/path1/filename -d '2 2 2'
python3 cellexp.py -i POSCAR -d '2 2 2' -c lammps -s -m
python3 cellexp.py -i POSCAR -c lammps -o model.txt

Only support lammps/vasp-type of structure for now
@ Author: Zheyi Zhang(DUT, China)
@ Email : zhangzy3033@163.com
'''
import os
import sys
import numpy as np

def error(word):
    print('%s!\n'%(word),flush=True)
    exit()

def dir2cart(lattice,positions,seq='normal'):
    newpos = None
    if seq == 'normal':
        newpos = positions.dot(lattice)
        
    elif seq == 'reverse': # cart2dir
        newpos = np.array(positions.dot(np.matrix(lattice).I))
        
    else:
        generate.error('Unsupported command "%s" when converting atomic coordinates'%(seq))
    
    return newpos

class Readmodel:
    def __init__(self,path,file,times='all'):
        self.path    = path
        self.file    = file
        self.atoms   = 8
        self.lattice = np.zeros((3,3))
        self.type    = None
        self.masss   = []
        self.frames  = times
        Readmodel.getype(self)
        self.postions = np.zeros((self.atoms,3))

        if self.type == 'lammps':
            Readmodel.readlammps(self)
        elif self.type == 'vasp':
            Readmodel.readvasp(self)
        elif self.type == 'dump':
            Readmodel.readdump(self)         
#         elif self.type == 'hit':
#             Readmodel.readhit(self)
#         elif self.type == 'gala':
#             Readmodel.readgala(self)
#         elif self.type == 'gpumd':
#             pass
#         else:
#             pass
        
    def getype(self):
        lines = open(os.path.join(self.path,self.file),'r+').readlines()[:9]
        supportype = ['lammps','vasp','hit','gala','gpumd','dump']
        
        for line in lines:
            if 'atoms' in line and '#' not in line[:2]:
                self.type  = 'lammps'
                break
            if 'ITEM: TIMESTEP' in line:
                self.type  = 'dump'
                break
#             if '%Uni-cell vector' in line or '%Atom site' in line:
#                 self.type  = 'hit'
#                 break    
#             if 'galamost' in line or 'box lx=' in line:
#                 self.type = 'gala'
#                 break
        if not self.type:        
            if 'd' == lines[7][0].lower() or 'c'== lines[7][0].lower():
                if len(lines[2].split()) >= 3 and list(map(float,lines[2].split()[:3])):
                    self.type  = 'vasp'
                    
        if not self.type or not any(self.type in i for i in supportype):
            error('Unsupported format "%s" of %s'%(self.type,self.file))
                    
        del lines
        return None
    
    def readlammps(self):
        self.style = 'full'
        self.vels  = None
        self.bonds = np.zeros((3,4))
        self.angls = np.zeros((3,5))
        self.dihes = np.zeros((3,6))
        self.imprs = np.zeros((3,6))
        
        with open(os.path.join(self.path,self.file),'r+') as fr:
            while True:
                line = fr.readline()
                if not line:break
                if not line.split():continue
                if 'atoms' in line:
                    self.atoms = int(line.split()[0])
                    self.posit = np.zeros((self.atoms,7))
                    continue
                if 'bonds' in line:
                    self.bonds = [int(line.split()[0])]
                    continue
                if 'angles' in line:
                    self.angls = [int(line.split()[0])]
                    continue
                if 'dihedrals' in line:
                    self.dihes = [int(line.split()[0])]
                    continue
                if 'impropers' in line:
                    self.imprs = [int(line.split()[0])]
                    continue
                if 'atom types' in line:
                    continue
                if 'bond types' in line:
                    self.bonds.append(int(line.split()[0]))
                    continue
                if 'angle types' in line:
                    self.angls.append(int(line.split()[0]))
                    continue
                if 'dihedral types' in line:
                    self.dihes.append(int(line.split()[0]))
                    continue
                if 'improper types' in line:
                    self.imprs.append(int(line.split()[0]))
                    continue
                if 'xlo' in line and 'xhi' in line:
                    
                    # xhi-xlo means converting lammps-box to conventional lattice
                    self.lattice[0,0] = float(line.split()[1])-float(line.split()[0])
                    
                    # store the xlo
                    self.lattice[0,1] = float(line.split()[0])
                    
                    while True:
                        line = fr.readline().split('#')[0]
                        if not line.split():continue
                        if 'ylo' in line:
                            self.lattice[1,1] = float(line.split()[1])-float(line.split()[0])
                            self.lattice[1,2] = float(line.split()[0])
                        if 'zlo' in line:
                            self.lattice[2,2] = float(line.split()[1])-float(line.split()[0])
                            self.lattice[0,2] = float(line.split()[0])
                            break
                    continue
                if 'xy' in line and 'xz' in line and 'yz' in line:
                    self.lattice[1,0] = float(line.split()[0])
                    self.lattice[2,0] = float(line.split()[1])
                    self.lattice[2,1] = float(line.split()[2])
                    continue
                if 'Mass' in line:
                    while True:
                        line = fr.readline()
                        if not line.split():continue
                        if 'Atoms' in line:break
                        self.masss.append(line)
                    #print(self.masss)
                    for atom in range(len(self.masss)):
                        self.masss[atom] = atomass(float(self.masss[atom].split()[1]))[0]
                if 'Atoms' in line:
                    try:self.style = line.split('#')[1].split()[0]
                    except:error('Please specify the proper atomstyle at the line of "Atoms" in your lammps model "%s"'%self.file)
                    while True:
                        line = fr.readline().split('#')[0]
                        if not line.split():continue
                        #print(line)
                        self.posit[0,4:] = list(map(float,line.split()[-3:]))
                        if self.style == 'full':
                            self.posit[0,:4] = list(map(float,line.split()[:4]))
                            self.posit[0,4:] = list(map(float,line.split()[4:7]))
                            for i in range(1,self.atoms):
                                line = fr.readline().split('#')[0]
                                self.posit[i,:4] = list(map(float,line.split()[:4]))
                                self.posit[i,4:] = list(map(float,line.split()[4:7]))
                                
                        elif self.style == 'atomic':
                            self.posit[0,:4:2] = list(map(float,line.split()[:2]))
                            for i in range(1,self.atoms):
                                line = fr.readline().split('#')[0]
                                self.posit[i,4:]   = list(map(float,line.split()[-3:]))
                                self.posit[i,:4:2] = list(map(float,line.split()[:2]))
                            self.posit[:,1] = 1
                        
                        elif self.style == 'charge':
                            self.posit[0,:3] = list(map(float,line.split()[:3]))
                            for i in range(1,self.atoms):
                                line = fr.readline().split('#')[0]
                                self.posit[i,4:] = list(map(float,line.split()[-3:]))
                                self.posit[i,:3] = list(map(float,line.split()[:3]))
                            self.posit[:,2:4] = self.posit[:,1:3]
                            self.posit[:,1]   = 1
                        
                        else:
                            error('Unsupported atomstyle "%s" for BMD in LAMMPS model "%s"'%(self.style,self.file))
                        break
                    
                    self.posit[:,4] -= self.lattice[0,1]
                    self.posit[:,5] -= self.lattice[1,2]
                    self.posit[:,6] -= self.lattice[0,2]
                    
                    # remove the stored los
                    self.lattice[0,1] = 0.0
                    self.lattice[1,2] = 0.0
                    self.lattice[0,2] = 0.0
                    
                    if self.masss:
                        for atom in range(len(self.masss)):
                            self.masss.append(np.sum(self.posit[:,2] == atom+1))
                    continue
                
                if 'Velocities' in line:
                    self.vels = np.zeros((self.atoms,4))
                    while True:
                        line = fr.readline().split('#')[0]
                        if not line.split():continue
                        self.vels[0,:] = list(map(float,line.split()))
                        for i in range(1,self.atoms):
                            line = fr.readline().split('#')[0]
                            self.vels[i,:] = list(map(float,line.split()))
                        break
                    continue
                
                if 'Bonds' in line:
                    bonds = self.bonds[0]
                    self.bonds = np.zeros((bonds,4))
                    while True:
                        line = fr.readline().split('#')[0]
                        if not line.split():continue
                        self.bonds[0,:] = list(map(float,line.split()))
                        for i in range(1,bonds):
                            line = fr.readline().split('#')[0]
                            self.bonds[i,:] = list(map(float,line.split()))
                        break
                    continue
                
                if 'Angles' in line:
                    angls = self.angls[0]
                    self.angls = np.zeros((angls,5))
                    while True:
                        line = fr.readline().split('#')[0]
                        if not line.split():continue
                        self.angls[0,:] = list(map(float,line.split()))
                        for i in range(1,angls):
                            line = fr.readline().split('#')[0]
                            self.angls[i,:] = list(map(float,line.split()))
                        break
                    continue
                
                if 'Dihedrals' in line:
                    dihes = self.dihes[0]
                    self.dihes = np.zeros((dihes,6))
                    while True:
                        line = fr.readline().split('#')[0]
                        if not line.split():continue
                        self.dihes[0,:] = list(map(float,line.split()))
                        for i in range(1,dihes):
                            line = fr.readline().split('#')[0]
                            self.dihes[i,:] = list(map(float,line.split()))
                        break
                    continue
                
                if 'Impropers' in line:
                    imprs = self.imprs[0]
                    self.imprs = np.zeros((imprs,6))
                    while True:
                        line = fr.readline().split('#')[0]
                        if not line.split():continue
                        self.imprs[0,:] = list(map(float,line.split()))
                        for i in range(1,imprs):
                            line = fr.readline().split('#')[0]
                            self.imprs[i,:] = list(map(float,line.split()))
                        break
                    continue
        return None
    
    def readvasp(self):
        self.style = 'Direct'
        const = 1.0
        with open(os.path.join(self.path,self.file),'r+') as fr:
            fr.readline()
            const = float(fr.readline().split()[0])
            for i in range(3):
                self.lattice[i,:] = list(map(float,fr.readline().split()[:3]))
            for i in range(2):
                atoms = fr.readline().split()
                for atom in atoms:
                    try:self.masss.append(int(atom))
                    except ValueError:self.masss.append(atom)
            types = int(len(self.masss)/2)
            self.lattice *= const
            self.atoms = sum(self.masss[types:])
            self.posit = np.zeros((self.atoms,7))
            if 'c' == fr.readline()[0].lower():
                self.style = 'Cart'
                
            for i in range(self.atoms):
                self.posit[i,0]  = int(i)
                self.posit[i,4:] = list(map(float,fr.readline().split()[:3]))
            self.posit[:,1] = 1
            for i in range(types):
                self.posit[sum(self.masss[types:types+i]):sum(self.masss[types:types+i+1]),2] = i+1
        return None
    
    def readdump(self):
        self.style   = 'full'
        self.lattice = np.zeros((3,3))
        self.atoms   = None
        self.posits  = None
        self.pindex  = None
        self.vindex  = None
        self.findex  = None
        self.typedx  = None
        self.scaled  = False
        with open(os.path.join(self.path,self.file)) as fr:
            count = 0
            while True:
                line = fr.readline()
                if not line:
                    self.posits = np.array(self.posits)
                    break
                if not line.split():continue
                if 'NUMBER OF ATOMS' in line and not self.atoms:
                    self.atoms = int(fr.readline().split()[0])
                    self.posit = np.zeros((self.atoms,7))
                    continue
                if 'ITEM: BOX BOUNDS' in line and np.all(self.lattice == 0):
                    data = list(map(float,fr.readline().split()))
                    self.lattice[0,0] = data[1] - data[0] # x
                    self.lattice[0,1] = data[0] # x
                    if len(data) > 2:
                        self.lattice[1,0] = data[2] # xy
                    data = list(map(float,fr.readline().split())) 
                    self.lattice[1,1] = data[1] - data[0] # y
                    self.lattice[0,2] = data[0] # y
                    if len(data) > 2:
                        self.lattice[2,0] = data[2] # xz
                    data = list(map(float,fr.readline().split())) 
                    self.lattice[2,2] = data[1] - data[0] # z
                    self.lattice[1,2] = data[0] # z
                    if len(data) > 2:
                        self.lattice[2,1] = data[2] # yz
                    continue
                if 'ITEM: ATOMS' in line:
                    if not self.posits:
                        data = line.split('ITEM: ATOMS')[-1].split()
                        self.column = len(data)
                        if isinstance(self.frames,int):
                            self.posits = np.zeros((self.frames,self.atoms,self.column))
                        elif self.frames=='all':
                            self.posits = []
                        else:
                            raise
                        for i in range(self.column):
                            if 'x' == data[i][0]:
                                self.pindex = i
                                if data[i] == 'xs':
                                    #print('yes')
                                    self.scaled = True
                            elif 'vx' == data[i]:
                                self.vindex = i
                            elif 'fx' == data[i]:
                                self.findex = i
                            elif 'type' == data[i]:
                                self.typedx = i
                    tmp = np.zeros((self.atoms,self.column))
                    count += 1
                    for i in range(self.atoms):
                        tmp[i,:] = list(map(float,fr.readline().split()))
                    if self.pindex != None:
                        tmp[:,self.pindex] -= self.lattice[0,1] # x
                        tmp[:,self.pindex+1] -= self.lattice[0,2] # y
                        tmp[:,self.pindex+2] -= self.lattice[1,2] # z
                    if self.scaled:
                        tmp[:,self.pindex:self.pindex+3] = dir2cart(self.lattice,tmp[:,self.pindex:self.pindex+3])
                    
                    if isinstance(self.frames,int):
                        self.posits[count,:,:] == tmp
                    else:
                        self.posits.append(tmp)
                    if self.frames == count:
                        break
                    else:
                        continue
            if self.pindex != None:
                self.posit[:,1]  = 1
                self.posit[:,4:] = tmp[:,self.pindex:self.pindex+3]
            if self.typedx != None:
                self.posit[:,2]  = tmp[:,self.typedx]
        self.lattice[0,1] = 0.0
        self.lattice[0,2] = 0.0
        self.lattice[1,2] = 0.0
        self.posits = np.array(self.posits)
        return None
    
def atomass(symbol):
    masses = {'H':'1.00794','He':'4.002602','Li':'6.941','Be':'9.012182',
              'B':'10.811','C':'12.0107','N':'14.0067','O':'15.9994',
              'F':'18.9984','Ne':'20.1797','Na':'22.98977','Mg':'24.3050',
              'Al':'26.9815','Si':'28.0855','P':'30.973762','S':'32.065',
              'Cl':'35.453','Ar':'39.948','K':'39.0983','Ca':'40.078',
              'Sc':'44.955912','Ti':'47.867','V':'50.9415','Cr':'51.9961',
              'Mn':'54.938','Fe':'55.845','Co':'58.933195','Ni':'58.6934',
              'Cu':'63.546','Zn':'65.38','Ga':'69.723','Ge':'72.64',
              'As':'74.92160','Se':'78.96','Br':'79.904','Kr':'83.798',
              'Rb':'85.4678','Sr':'87.62','Y':'88.90585','Zr':'91.224',
              'Nb':'92.90638','Mo':'95.96','Tc':'98.9063','Ru':'101.07',
              'Rh':'102.90550','Pd':'106.42','Ag':'107.8682','Cd':'112.411',
              'In':'114.818','Sn':'118.710','Sb':'121.760','Te':'127.60',
              'I':'126.90447','Xe':'131.293','Cs':'132.9054519','Ba':'137.327',
              'W':'183.84'}
    
    mass = []
    if isinstance(symbol,str):
        mass.append(symbol)
        try:mass.append(masses[symbol])
        except KeyError:error('Unsupported type of element "%s" in BMD'%symbol)
    
    elif isinstance(symbol,float):
        for atom in masses:
            if round(float(masses[atom])) == round(symbol):
                mass.append(atom)
                mass.append(masses[atom])
                try:mass.append(masses[atom])
                except KeyError:error('Unsupported type of element "%s" in BMD'%atom)
    
    return mass

def lattice2lammps(lattice):
    lammps = np.zeros((3,3))
    Ahat    = lattice[0]/np.sqrt(np.sum(np.square(lattice[0])))
    AcrossB = np.cross(lattice[0],lattice[1])
    AcrossBh= AcrossB/np.sqrt(np.sum(np.square(AcrossB)))
    lammps[0,0] = np.sqrt(np.sum(np.square(lattice[0])))
    lammps[1,1] = np.sqrt(np.sum(np.square(np.cross(Ahat,lattice[1]))))
    lammps[2,2] = np.sqrt(np.sum(np.square(lattice[2].dot(AcrossBh))))
    lammps[1,0] = lattice[1].dot(Ahat) # xy
    lammps[2,0] = lattice[2].dot(Ahat) # xz
    lammps[2,1] = lattice[2].dot(np.cross(AcrossBh,Ahat)) # yz
    
    return lammps

def sed_lattice(path,position,masses,scale):
    #posi = [j for i in position for j in i]
    with open(os.path.join(path,'lattice.dat'),'w') as fw:
        count = 0
        for i in range(int(scale[0]*scale[1]*scale[2])):
            for j in range(len(position)):
                count += 1
                fw.write('%d %d %d %s\n'%(count,i+1,j+1,atomass(masses[int(position[j][2]-1)])[1]))
    return None

def gen_super(path,file,scale,
              strs=None,
              outype=None,
              outfile=None,
              mode='generate',
              showinfo=True,
              sed=False):
    # outype: lammps/vasp, default to be the same with input file
    # sed   = False
    molecule = False
    # read shell command
    if strs:
        helps = 'Options:\n'+\
                '  -h              Show help info\n'+\
                '  -i file         Submit file to be running\n'+\
                '  -d \'3 3 3\'      Expand unitcell for 3 times in each axis\n'+\
                '  -c lammps       Generate lammps-format file(lammps/vasp)\n'+\
                '  -s              Generate SED lattice file\n'+\
                '  -m              Consider molecule structure in lammps-format file\n'
        for command in strs:
            if '-h' == command:
                print(helps,flush=True)
                exit()
            elif '-i' == command:
                file = os.path.basename(strs[strs.index(command)+1])
                if os.path.dirname(strs[strs.index(command)+1]):
                    path = os.path.dirname(strs[strs.index(command)+1])
            elif '-d' == command:
                scale = list(map(int,strs[strs.index(command)+1].split()))
            elif '-s' == command:
                sed = True
            elif '-m' == command:
                molecule = True
            elif '-c' == command:
                outype = strs[strs.index(command)+1]
            elif '-o' == command:
                outfile = strs[strs.index(command)+1]
                
    if not all(isinstance(i,int) for i in scale):
        error('All scaling times in three axis should be int: %s'%str(scale))
    if any(i == 0 for i in scale):
        error('What do you mean by setting scale numbers as 0')

    # read structure file    
    model = Readmodel(path,file)
    unitcell = model.posit[:,4:].copy()
    lattice  = model.lattice.copy()
    if model.style == 'Direct':
        if outype == 'lammps':lattice = lattice2lammps(model.lattice)
        unitcell = unitcell.dot(lattice)
    if not outype:
        outype = model.type
        if outype == 'dump': outype = 'lammps'
    if outype not in ['lammps','vasp']:
        error('Unsupported conversion type "%s"'%outype)
    if not outfile:
        outfile = file + '_'+str(scale[0])+str(scale[1])+str(scale[2])
        if '.' in file:
            outfile = file.rsplit('.',1)[0]+'_'+str(scale[0])+str(scale[1])+str(scale[2])+'.'+file.rsplit('.',1)[-1]
        if outype == 'vasp' and 'POSCAR' not in outfile:
            outfile = outfile.rsplit('.',1)[0]+'.vasp'
    new_lattice   = np.zeros((3,3))
    new_positions = np.zeros((scale[0]*scale[1]*scale[2],unitcell.shape[0],unitcell.shape[1]))
    new_molecules = {'bonds':[],'angles':[],'dihedrals':[],'impropers':[]} # lammps

    if mode == 'generate':
        if showinfo:
            print('Running code with settings: %s to %d*%d*%d(%s)'%(model.file,scale[0],scale[1],scale[2],outfile),flush=True)
            print('Format conversion: %s to %s'%(model.type,outype),flush=True)
        
        # expand unitcell
        new_lattice = lattice*scale
        count = 0
        for n3 in range(scale[2]):
            for n2 in range(scale[1]):
                for n1 in range(scale[0]):
                    for axis in range(3):
                        new_positions[count,:,axis] = unitcell[:,axis] + np.sum(lattice[:,axis]*np.array([n1,n2,n3]))
                    count += 1
                
        if molecule and model.type == 'lammps':
            for unit in range(len(new_positions)):
                for key in new_molecules.keys():
                    new_molecules[key].append([])
            if np.any(model.bonds):
                new_molecules['bonds'][0] = model.bonds.copy()
                for unit in range(1,len(new_positions)):
                    new_molecules['bonds'][unit] = model.bonds.copy()
                    new_molecules['bonds'][unit][:,2:] += unit*model.atoms
            if np.any(model.angls):
                new_molecules['angles'][0] = model.angls.copy()
                for unit in range(1,len(new_positions)):
                    new_molecules['angles'][unit] = model.angls.copy()
                    new_molecules['angles'][unit][:,2:] += unit*model.atoms
            if np.any(model.dihes):
                new_molecules['dihedrals'][0] = model.dihes.copy()
                for unit in range(1,len(new_positions)):
                    new_molecules['dihedrals'][unit] = model.dihes.copy()
                    new_molecules['dihedrals'][unit][:,2:] += unit*model.atoms
            if np.any(model.imprs):
                new_molecules['impropers'] = model.imprs.copy()
                for unit in range(1,len(new_positions)):
                    new_molecules['impropers'][unit] = model.imprs.copy()
                    new_molecules['impropers'][unit][:,2:] += unit*model.atoms
                    
        # generate supercell
        if outype == 'lammps':
            if model.type != 'lammps':model.style = 'full'
                #new_lattice = lattice2lammps(new_lattice)
            with open(os.path.join(path,outfile),'w') as fw:
                fw.write('# LAMMPS data file written by Zheyi\'s python code\n')
                fw.write('%d atoms\n'%(model.atoms*scale[0]*scale[1]*scale[2]))
                if molecule:
                    model.style = 'full'
                    fw.write('%d bonds\n'%(len(new_molecules['bonds'])*len(new_molecules['bonds'][0])))
                    fw.write('%d angles\n'%(len(new_molecules['angles'])*len(new_molecules['angles'][0])))
                    fw.write('%d dihedrals\n'%(len(new_molecules['dihedrals'])*len(new_molecules['dihedrals'][0])))
                    fw.write('%d impropers\n\n'%(len(new_molecules['impropers'])*len(new_molecules['impropers'][0])))
                fw.write('%d atom types\n'%(np.max(model.posit[:,2])))
                if molecule:
                    if np.any(model.bonds):fw.write('%d bond types\n'%(np.max(model.bonds[:,1])))
                    else:fw.write('%d bond types\n'%(0))
                    if np.any(model.angls):fw.write('%d angle types\n'%(np.max(model.angls[:,1])))
                    else:fw.write('%d angle types\n'%(0))
                    if np.any(model.dihes):fw.write('%d dihedral types\n'%(np.max(model.dihes[:,1])))
                    else:fw.write('%d dihedral types\n'%(0))
                    if np.any(model.imprs):fw.write('%d improper types\n'%(np.max(model.imprs[:,1])))
                    else:fw.write('%d improper types\n'%(0))
                fw.write('\n0.0 %12.8f xlo xhi\n'%(new_lattice[0,0]))
                fw.write('0.0 %12.8f ylo yhi\n'%(new_lattice[1,1]))
                fw.write('0.0 %12.8f zlo zhi\n'%(new_lattice[2,2]))
                if np.any([new_lattice[1,0],new_lattice[2,0],new_lattice[2,1]]):
                    fw.write('%11.8f %11.8f %11.8f xy xz yz\n'%(new_lattice[1,0],new_lattice[2,0],new_lattice[2,1]))
                if model.masss:
                    fw.write('\nMasses\n\n')
                    for i in range(int(len(model.masss)/2)):
                        fw.write('%d %s # %s\n'%(i+1,atomass(model.masss[i])[1],model.masss[i]))
                fw.write('\nAtoms # %s\n\n'%(model.style))
                count = 0
                if model.style == 'full':
                    for unit in new_positions:
                        for atom in range(len(unit)):
                            fw.write('%2d %2d %2d %7.4f %12.8f %12.8f %12.8f\n'%(count+1,model.posit[atom][1],model.posit[atom][2],model.posit[atom][3],unit[atom][0],unit[atom][1],unit[atom][2]))
                            count += 1
                elif model.style == 'atomic':
                    for unit in new_positions:
                        for atom in range(len(unit)):
                            fw.write('%2d %2d %12.8f %12.8f %12.8f\n'%(count+1,model.posit[atom][2],unit[atom][0],unit[atom][1],unit[atom][2]))
                            count += 1
                elif model.style == 'charge':
                    for unit in new_positions:
                        for atom in range(len(unit)):
                            fw.write('%2d %2d %7.4f %12.8f %12.8f %12.8f\n'%(count+1,model.posit[atom][2],model.posit[atom][3],unit[atom][0],unit[atom][1],unit[atom][2]))
                            count += 1
                if molecule:
                    if np.any(model.bonds):
                        count = 0
                        fw.write('\nBonds\n\n')
                        for unit in new_molecules['bonds']:
                            for line in unit:
                                fw.write('%d %d %d %d\n'%(count+1,line[1],line[2],line[3]))
                                count += 1
                    if np.any(model.angls):
                        count = 0
                        fw.write('\nAngles\n\n')
                        for unit in new_molecules['angles']:
                            for line in unit:
                                fw.write('%d %d %d %d %d\n'%(count+1,line[1],line[2],line[3],line[4]))
                                count += 1
                    if np.any(model.dihes):
                        count = 0
                        fw.write('\nDihedrals\n\n')
                        for unit in new_molecules['dihedrals']:
                            for line in unit:
                                fw.write('%d %d %d %d %d %d\n'%(count+1,line[1],line[2],line[3],line[4],line[5]))
                                count += 1          
                    if np.any(model.imprs):
                        count = 0
                        fw.write('\nImpropers\n\n')
                        for unit in new_molecules['impropers']:
                            for line in unit:
                                fw.write('%d %d %d %d %d %d\n'%(count+1,line[1],line[2],line[3],line[4],line[5]))
                                count += 1  
        elif outype == 'vasp':
            if not model.masss:
                error('Unable to convert %s to vasp file due to the lack of elements type'%model.file)
            with open(os.path.join(path,outfile),'w') as fw:
                fw.write('# POSCAR written by Zheyi\'s python code\n')
                fw.write('%f\n'%(1.0))
                for axis in range(3):
                    fw.write('%12.8f %12.8f %12.8f\n'%(new_lattice[axis,0],new_lattice[axis,1],new_lattice[axis,2]))
                for types in range(int(len(model.masss)/2)):
                    fw.write('%s '%(model.masss[types]))
                fw.write('\n')
                for types in range(int(len(model.masss)/2)):
                    num = model.masss[types+int(len(model.masss)/2)]*scale[0]*scale[1]*scale[2]
                    fw.write('%s '%(num))
                fw.write('\nDirect\n')
                for types in range(int(len(model.masss)/2)):
                    index = np.argwhere(model.posit[:,2] == types+1)
                    for cell in new_positions:
                        cell = np.array(cell.dot(np.matrix(new_lattice).I))
                        for atom in cell[index]:
                            fw.write('%12.8f %12.8f %12.8f\n'%(atom[0][0],atom[0][1],atom[0][2]))

        if sed and model.masss:
            sed_lattice(path,model.posit,model.masss[:int(len(model.masss)/2)],scale)
        elif sed and not model.masss:
            error('Not able to generate SED lattice file due to the lack of masses info(Masses)')

        if showinfo:
            print('Supercell is generated done ~',flush=True)
        
    elif mode == 'read':
        pass

    return model,new_lattice,new_positions

if __name__ == '__main__':
    shell = sys.argv[1:]
    path  = os.getcwd()
    file  = 'diamond-exp-con.txt'  # support lammps/vasp type for now
    size  = [4,4,50]   # along the three axis
    write_sed_lattice = True
    unit, super_lattice, super_positions = gen_super(path,file,size,strs=shell,sed=write_sed_lattice)
    