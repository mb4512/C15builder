# initialise
import os, sys, glob
import numpy as np

import scipy
from scipy.spatial import cKDTree 

from numba import jit

import time
import sys


@jit(nopython=True)
def jcost_function_optnebs (delta, radius, nebvecs): 

    alattice = radius
    sigma = 0.25*alattice
    half_over_sigma_sq = 0.5/(sigma*sigma)
    
    vac_density = 0.0
    for i in range(len(nebvecs)):
        _dsq =  (nebvecs[i,0]+delta[0])*(nebvecs[i,0]+delta[0])
        _dsq += (nebvecs[i,1]+delta[1])*(nebvecs[i,1]+delta[1])
        _dsq += (nebvecs[i,2]+delta[2])*(nebvecs[i,2]+delta[2])
        vac_density += np.exp(-half_over_sigma_sq*_dsq)
        

    # penalise large offsets    
    penalty = 0.0
    c = 1.0
    for i in range(3):
        if np.abs(delta[i]) > c*sigma:    
            penalty += 100*(delta[i] - c*sigma)*(delta[i] - c*sigma)

    return vac_density + penalty



@jit(nopython=True)
def jcost_function_optnebs_deriv (delta, radius, nebvecs): 

    alattice = radius 
    sigma = 0.25*alattice
    half_over_sigma_sq = 0.5/(sigma*sigma)
    
    vac_deriv = np.zeros(3, dtype=float)
    for i in range(len(nebvecs)):
        _dsq =  (nebvecs[i,0]+delta[0])*(nebvecs[i,0]+delta[0])
        _dsq += (nebvecs[i,1]+delta[1])*(nebvecs[i,1]+delta[1])
        _dsq += (nebvecs[i,2]+delta[2])*(nebvecs[i,2]+delta[2])
        vac_deriv += -2.*half_over_sigma_sq * (nebvecs[i]+delta) * np.exp(-half_over_sigma_sq*_dsq)

    # penalise large offsets    
    c = 1.0
    for i in range(3):
        if np.abs(delta[i]) > c*sigma:    
            vac_deriv[i] += 2*100*(delta[i] - c*sigma)

    return vac_deriv

# compile functions
jcost_function_optnebs(np.random.rand(3), 3.5, np.random.rand(10,3))
jcost_function_optnebs_deriv(np.random.rand(3), 3.5, np.random.rand(10,3))

class EmptyClass:
    def __init__ (self):
        pass

class ReadFile:
    def __init__ (self, fpath, filetype="auto", my_rank=0, importing_rank=0, mpi_comm=None):
        '''Class for importing, processing, and storing atomic dump and data LAMMPS files'''
        self.fpath = fpath
        self.filetype = filetype
 
        # if given MPI comm: execute on one thread, then bcast to others to not waste i/o bandwidth
        # if not given MPI comm, construct a fake comm instance that does nothing, i.e. monkey patching
        if mpi_comm is not None:
            self.comm = mpi_comm
        else:
            self.comm = EmptyClass ()
            self.comm.bcast = lambda data, root=None: data 
            self.comm.barrier = lambda *args: args 

        # store MPI information, i.e. thread rank and MPI comm instance as attributes
        self.me = my_rank
        self.importing_rank = importing_rank 

    def load (self):
        '''Import dump file and save content as attributes.'''

        clock = time.time()

        if self.me == self.importing_rank:
            # detect file type from first line
            if self.filetype == "auto":
                with open(self.fpath, 'r') as fopen:
                    line = fopen.readline()
                    if "ITEM: TIMESTEP" in line:
                        self.filetype = "dump"
                    elif "LAMMPS data file" in line:
                        self.filetype = "data"
                print ("Detected file type:", self.filetype)

            if self.filetype == "dump":
                self.xyz, self.cell = self.read_dump (self.fpath)
            elif self.filetype == "data":
                self.xyz, self.cell = self.read_data (self.fpath)
            else:
                print ("Error: unknown file type, only data or dump are accepted.")

            self.natoms = len(self.xyz)           
        else:
            self.xyz  = None
            self.cell = None
            self.natoms = None 
        
        self.xyz = self.comm.bcast(self.xyz, root=self.importing_rank)
        self.cell = self.comm.bcast(self.cell, root=self.importing_rank)
        self.natoms = self.comm.bcast(self.natoms, root=self.importing_rank)

        clock = time.time() - clock
        if self.me == self.importing_rank:
            print ("Imported %d atoms from %s file: %s in %f seconds" % (self.natoms, self.filetype, self.fpath, clock))

    def read_dump (self, fpath):
        with open(fpath, 'r') as _dfile:
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            natoms = int(_dfile.readline())

            # read in box dimensions 
            if 'xy xz yz' in _dfile.readline():
                # Triclinic case
                xlb,xhb,xy = np.array(_dfile.readline().split(), dtype=float)
                ylb,yhb,xz = np.array(_dfile.readline().split(), dtype=float)
                zlb,zhb,yz = np.array(_dfile.readline().split(), dtype=float)
                self.ortho = False
            else:
                # Orthogonal case 
                xlb,xhb = np.array(_dfile.readline().split(), dtype=float)
                ylb,yhb = np.array(_dfile.readline().split(), dtype=float)
                zlb,zhb = np.array(_dfile.readline().split(), dtype=float)
                xy, xz, yz = 0.,0.,0.
                self.ortho = True 

            # this will be passed to AffineTransform to construct transformation matrices etc
            _cell = np.array([[xlb,xhb,xy], [ylb,yhb,xz], [zlb,zhb,yz]], dtype=float)

            # determine in which columns the atomic coordinates are stored 
            colstring = _dfile.readline()
            colstring = colstring.split()[2:] # first two entries typically are 'ITEM:' and 'ATOMS'
            
            try:
                ix = colstring.index('x') 
                iy = colstring.index('y') 
                iz = colstring.index('z') 
            except:
                raise RuntimeError("Could not find coordinates 'x', 'y', 'z' in the dump file!") 
            
            # read in atomic coordinates
            ixyz = np.array([ix,iy,iz])
            _xyz = [np.array(_dfile.readline().rstrip("\n").split(" "))[ixyz] for i in range(natoms)]
            _xyz = np.array(_xyz, dtype=float)
            
        return _xyz, _cell

    
    def read_data(self, fpath):
        # currently only supports orthogonal boxes
        with open(fpath, 'r') as _dfile:
            _dfile.readline()
            _dfile.readline()
            natoms = int(_dfile.readline().split()[0])

            _dfile.readline()
            _dfile.readline()

            xlo,xhi = np.array(_dfile.readline().split()[:2], dtype=float)
            ylo,yhi = np.array(_dfile.readline().split()[:2], dtype=float)
            zlo,zhi = np.array(_dfile.readline().split()[:2], dtype=float)
            _line = _dfile.readline()
            if "xy xz yz" in _line:
                self.ortho = False
                xy,xz,yz = np.array(_line.split()[:3], dtype=float)
                _dfile.readline()
            else:
                self.ortho = True
                xy,xz,yz = 0.,0.,0.

            # in contrast to the dump format, where xlo_bound, xhi_bound etc is stored, 
            # the read format stores xlo, xhi, etc., hence we need to convert
            # Relevant documentation: https://docs.lammps.org/Howto_triclinic.html 
            xlb = xlo + min(0.0,xy,xz,xy+xz)
            xhb = xhi + max(0.0,xy,xz,xy+xz)
            ylb = ylo + min(0.0,yz)
            yhb = yhi + max(0.0,yz)
            zlb, zhb = zlo, zhi

            # this will be passed to AffineTransform to construct transformation matrices etc
            _cell = np.array([[xlb,xhb,xy], [ylb,yhb,xz], [zlb,zhb,yz]], dtype=float)

            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()
            _dfile.readline()

            _atomdat = [_dfile.readline().rstrip('\n').split(' ') for i in range(natoms)]
            _atomdat = np.array(_atomdat, dtype=float)
            _xyz = _atomdat[:,2:5]

        return _xyz, _cell





def write_dump(cell, types, occ, xyz, path, frame):

    # exctract cell data
    N = len(xyz) 
    xlo_bound, xhi_bound, xy = cell[0]
    ylo_bound, yhi_bound, xz = cell[1]
    zlo_bound, zhi_bound, yz = cell[2]

    # write file header
    wfile = open(path, 'w')

    wfile.write("ITEM: TIMESTEP\n")
    wfile.write("%d\n" % frame)
    wfile.write("ITEM: NUMBER OF ATOMS\n")
    wfile.write("%d\n" % N)

    # simulation box info specific for orthogonal box with PBC
    wfile.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
    wfile.write("%f %f %f\n" % (xlo_bound, xhi_bound, xy))
    wfile.write("%f %f %f\n" % (ylo_bound, yhi_bound, xz))
    wfile.write("%f %f %f\n" % (zlo_bound, zhi_bound, yz))

    wfile.write("ITEM: ATOMS id type x y z occ\n")

    for _i,_xyz in enumerate(xyz):
        wfile.write("%5d %3d %14.8f %14.8f %14.8f %3d\n" % (_i+1, types[_i], _xyz[0], _xyz[1], _xyz[2], occ[_i])) 

    wfile.close()

    return 0


class AffineTransform:
    '''Define the affine transformation between two sytems.'''
    
    def __init__(self, celldim1, celldim2):
        # initialise affine transformation matrix and vector
        
        # get offset vectors
        self.r1 = self.get_origin(celldim1)
        self.r2 = self.get_origin(celldim2)
        
        # get basis matrices
        self.c1mat = self.get_cell_vectors(celldim1).T
        self.c2mat = self.get_cell_vectors(celldim2).T

        # get inverse matrices for lattice vector transformation
        self.c1mati = np.linalg.inv(self.c1mat)
        self.c2mati = np.linalg.inv(self.c2mat)

        # build transformation matrices
        self.amatrix  = np.matmul(self.c2mat, np.linalg.inv(self.c1mat))
        self.amatrixi = np.matmul(self.c1mat, np.linalg.inv(self.c2mat))
    
    def get_cell_dimensions(self, celldim):
        xlo_bound, xhi_bound, xy = celldim[0]
        ylo_bound, yhi_bound, xz = celldim[1]
        zlo_bound, zhi_bound, yz = celldim[2]

        xlo = xlo_bound - min(0., xy, xz, xy+xz)
        xhi = xhi_bound - max(0., xy, xz, xy+xz)
        ylo = ylo_bound - min(0., yz)
        yhi = yhi_bound - max(0., yz)
        zlo = zlo_bound
        zhi = zhi_bound

        return (xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz)

    
    def get_origin(self, celldim):
        (xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz) = self.get_cell_dimensions(celldim)
        return np.r_[xlo, ylo, zlo]

    
    def get_cell_vectors(self, celldim):
        (xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz) = self.get_cell_dimensions(celldim)

        c1 = np.r_[xhi-xlo, 0., 0.]
        c2 = np.r_[xy, yhi-ylo, 0.]
        c3 = np.r_[xz, yz, zhi-zlo]

        return np.c_[[c1,c2,c3]]
    
    def go1to2(self, xyz):
        '''Transform vector xyz from reference frame 1 to reference frame 2.
        
        Only for testing single transformations! It is slow.'''
        return  self.amatrix.dot(xyz - self.r1) + self.r2
    
    def go2to1(self, xyz):
        '''Transform vector xyz from reference frame 2 to reference frame 1.
        
        Only for testing single transformations! It is slow.'''
        return self.amatrixi.dot(xyz - self.r2) + self.r1
 
    def go1to2_all(self, xyz):
        '''Transform all vectors xyz from reference frame 1 to reference frame 2.
        
        This is reasonably efficient.'''
        return np.r_[[self.amatrix[0][0]*(xyz[:,0] - self.r1[0]) +
                      self.amatrix[0][1]*(xyz[:,1] - self.r1[0]) +
                      self.amatrix[0][2]*(xyz[:,2] - self.r1[0]) + self.r2[0]],
                     [self.amatrix[1][0]*(xyz[:,0] - self.r1[1]) +
                      self.amatrix[1][1]*(xyz[:,1] - self.r1[1]) +
                      self.amatrix[1][2]*(xyz[:,2] - self.r1[1]) + self.r2[1]],
                     [self.amatrix[2][0]*(xyz[:,0] - self.r1[2]) +
                      self.amatrix[2][1]*(xyz[:,1] - self.r1[2]) +
                      self.amatrix[2][2]*(xyz[:,2] - self.r1[2]) + self.r2[2]]].T
    
    def go2to1_all(self, xyz):
        '''Transform all vectors xyz from reference frame 2 to reference frame 1.
        
        This is reasonably efficient.'''
        return np.r_[[self.amatrixi[0][0]*(xyz[:,0] - self.r2[0]) +
                      self.amatrixi[0][1]*(xyz[:,1] - self.r2[0]) +
                      self.amatrixi[0][2]*(xyz[:,2] - self.r2[0]) + self.r1[0]],
                     [self.amatrixi[1][0]*(xyz[:,0] - self.r2[1]) +
                      self.amatrixi[1][1]*(xyz[:,1] - self.r2[1]) +
                      self.amatrixi[1][2]*(xyz[:,2] - self.r2[1]) + self.r1[1]],
                     [self.amatrixi[2][0]*(xyz[:,0] - self.r2[2]) +
                      self.amatrixi[2][1]*(xyz[:,1] - self.r2[2]) +
                      self.amatrixi[2][2]*(xyz[:,2] - self.r2[2]) + self.r1[2]]].T

    def pbcwrap1(self, xyz):
        '''Check if a vector falls outside the box, and if so, wrap it back inside.'''

        fcoords = np.matmul(self.c1mati, xyz - self.r1)
        gcoords = fcoords - np.floor(fcoords)

        return self.r1 + np.matmul(self.c1mat, gcoords)

    def pbcwrap2(self, xyz):
        '''Check if a vector falls outside the box, and if so, wrap it back inside.'''

        fcoords = np.matmul(self.c2mati, xyz - self.r2)
        gcoords = fcoords - np.floor(fcoords)

        return self.r2 + np.matmul(self.c2mat, gcoords)

    def cart_to_frac1_all(self,xyz):
        '''Transform all vectors xyz to fractional coordinates in reference frame 1.'''
        return np.r_[[self.c1mati[0][0]*(xyz[:,0] - self.r1[0]) +
                      self.c1mati[0][1]*(xyz[:,1] - self.r1[0]) +
                      self.c1mati[0][2]*(xyz[:,2] - self.r1[0])],
                     [self.c1mati[1][0]*(xyz[:,0] - self.r1[1]) +
                      self.c1mati[1][1]*(xyz[:,1] - self.r1[1]) +
                      self.c1mati[1][2]*(xyz[:,2] - self.r1[1])],
                     [self.c1mati[2][0]*(xyz[:,0] - self.r1[2]) +
                      self.c1mati[2][1]*(xyz[:,1] - self.r1[2]) +
                      self.c1mati[2][2]*(xyz[:,2] - self.r1[2])]].T        
        
    def cart_to_frac2_all(self,xyz):
        '''Transform all vectors xyz to fractional coordinates in reference frame 2.'''
        return np.r_[[self.c2mati[0][0]*(xyz[:,0] - self.r2[0]) +
                      self.c2mati[0][1]*(xyz[:,1] - self.r2[0]) +
                      self.c2mati[0][2]*(xyz[:,2] - self.r2[0])],
                     [self.c2mati[1][0]*(xyz[:,0] - self.r2[1]) +
                      self.c2mati[1][1]*(xyz[:,1] - self.r2[1]) +
                      self.c2mati[1][2]*(xyz[:,2] - self.r2[1])],
                     [self.c2mati[2][0]*(xyz[:,0] - self.r2[2]) +
                      self.c2mati[2][1]*(xyz[:,1] - self.r2[2]) +
                      self.c2mati[2][2]*(xyz[:,2] - self.r2[2])]].T   
    
    def frac_to_cart1_all(self,frac):
        '''Transform all vectors fractional coordinates to cartesian in reference frame 1.'''
        return np.r_[[self.c1mat[0][0]*frac[:,0] +
                      self.c1mat[0][1]*frac[:,1] +
                      self.c1mat[0][2]*frac[:,2] + self.r1[0]],
                     [self.c1mat[1][0]*frac[:,0] +
                      self.c1mat[1][1]*frac[:,1] +
                      self.c1mat[1][2]*frac[:,2] + self.r1[1]],
                     [self.c1mat[2][0]*frac[:,0] +
                      self.c1mat[2][1]*frac[:,1] +
                      self.c1mat[2][2]*frac[:,2] + self.r1[2]]].T  

    def frac_to_cart2_all(self,frac):
        '''Transform all vectors fractional coordinates to cartesian in reference frame 2.'''
        return np.r_[[self.c2mat[0][0]*frac[:,0] +
                      self.c2mat[0][1]*frac[:,1] +
                      self.c2mat[0][2]*frac[:,2] + self.r2[0]],
                     [self.c2mat[1][0]*frac[:,0] +
                      self.c2mat[1][1]*frac[:,1] +
                      self.c2mat[1][2]*frac[:,2] + self.r2[1]],
                     [self.c2mat[2][0]*frac[:,0] +
                      self.c2mat[2][1]*frac[:,1] +
                      self.c2mat[2][2]*frac[:,2] + self.r2[2]]].T  
    
    def pbcdistance (self, xyz1, xyz2):
        '''Compute the distance between two vectors, taking into account pbc.
        Only for testing single transformations! It is slow.'''

        fcoords1 = np.matmul(self.c1mati, xyz1 - self.r1)
        fcoords2 = np.matmul(self.c1mati, xyz2 - self.r1)
        
        df = fcoords2 - fcoords1
        dg = df - np.sign(df)*(np.abs(df) > .5).astype(int)

        return np.linalg.norm(np.matmul(self.c1mat, dg))


    def pbcdistance_all(self, xyz, xyz_all):
        '''Compute the distance between one vector and a list of vectors, taking into account pbc.'''

        fcoords = np.matmul(self.c1mati, xyz - self.r1)
        
        xyzr_all = xyz_all - self.r1
        fcoords_all = np.r_[[self.c1mati[0][0]*(xyzr_all[:,0]) +
                             self.c1mati[0][1]*(xyzr_all[:,1]) +
                             self.c1mati[0][2]*(xyzr_all[:,2])], 
                            [self.c1mati[1][0]*(xyzr_all[:,0]) +
                             self.c1mati[1][1]*(xyzr_all[:,1]) +
                             self.c1mati[1][2]*(xyzr_all[:,2])],
                            [self.c1mati[2][0]*(xyzr_all[:,0]) +
                             self.c1mati[2][1]*(xyzr_all[:,1]) +
                             self.c1mati[2][2]*(xyzr_all[:,2])]].T
         
        df = fcoords_all - fcoords
        dg = df - np.sign(df)*(np.abs(df) > .5).astype(int)

        dxyz_all = np.r_[[self.c1mat[0][0]*dg[:,0] +
                          self.c1mat[0][1]*dg[:,1] +
                          self.c1mat[0][2]*dg[:,2]], 
                         [self.c1mat[1][0]*dg[:,0] +
                          self.c1mat[1][1]*dg[:,1] +
                          self.c1mat[1][2]*dg[:,2]],
                         [self.c1mat[2][0]*dg[:,0] +
                          self.c1mat[2][1]*dg[:,1] +
                          self.c1mat[2][2]*dg[:,2]]]
        
        dxyz_norm = np.sqrt(dxyz_all[0]*dxyz_all[0] + dxyz_all[1]*dxyz_all[1] + dxyz_all[2]*dxyz_all[2])
        return dxyz_norm 



def announce(string):
    print ("\n=====================================================")
    print (string)
    print ("=====================================================\n")
    return 0


class WignerSeitz:
    def __init__(self, reference_path, my_rank=0, exporting_rank=0, mpi_comm=None):

        # if not given MPI comm, construct a fake comm instance that does nothing, i.e. monkey patching
        if mpi_comm is not None:
            self.comm = mpi_comm
            self.nprocs = self.comm.Get_size()
        else:
            self.comm = EmptyClass ()
            self.comm.bcast = lambda data, root=None: data 
            self.comm.barrier = lambda *args: args
            self.nprocs = 1 
 
        # store MPI information, i.e. thread rank and MPI comm instance as attributes
        self.me = my_rank
        self.exporting_rank = exporting_rank

        # import reference cell
        if self.me == self.exporting_rank:
            print ("Importing reference cell data %s... " % reference_path, end='')

        self.data0 = ReadFile (reference_path, my_rank=self.me, importing_rank=self.exporting_rank, mpi_comm=self.comm)
        self.data0.load ()
           
        # initial affine transformation instance (needed here to get the cell vectors)
        self.atrans = AffineTransform (self.data0.cell, self.data0.cell)

        # build KDTree of periodically repeated reference structure for nearest neighbour search
        clock = time.time()
        if self.me == self.exporting_rank:
            print ("Building k-tree of reference cell atoms... ", end='')

        self.ktree0 = Triclinic_KDTree_Buffer (self.data0.xyz, self.atrans.c1mat, 
                                               r0=self.atrans.r1, balanced_tree=False)
        #self.ktree0 = Triclinic_KDTree_Buffer_MPI (self.data0.xyz, self.atrans.c1mat, 
        #                                           r0=self.atrans.r1, 
        #                                           my_rank=self.me, exporting_rank=self.exporting_rank, mpi_comm=self.comm,
        #                                           balanced_tree=False)

        self.dcopy = self.ktree0.dcopy

        clock = time.time() - clock
        if self.me == self.exporting_rank:
            print ("Finished in %f seconds." % clock)
   
 
    def set_state (self, distorted_state):
 
        # assume path to a distorted file is given, otherwise assume it is a lammps instance
        if type(distorted_state) == str:
            assert os.path.isfile(distorted_state), "File %s not found." % distorted_state
            self.datap = ReadFile (distorted_state, my_rank=self.me, importing_rank=self.exporting_rank, mpi_comm=self.comm)
            self.datap.load ()
        else:
            lmp = distorted_state
            assert hasattr(lmp, "extract_global"), "Given state %s is not a LAMMPS instance!" % str(distorted_state)

            natoms = lmp.extract_global("natoms", 0)
            xlo, xhi = lmp.extract_global("boxxlo", 2), lmp.extract_global("boxxhi", 2)
            ylo, yhi = lmp.extract_global("boxylo", 2), lmp.extract_global("boxyhi", 2)
            zlo, zhi = lmp.extract_global("boxzlo", 2), lmp.extract_global("boxzhi", 2)
            xy, yz, xz = lmp.extract_global("xy", 2), lmp.extract_global("yz", 2), lmp.extract_global("xz", 2)

            # types = np.ctypeslib.as_array(lmp.gather_atoms("type", 1, 1)).reshape(natoms)
            x = np.ctypeslib.as_array(lmp.gather_atoms("x", 1, 3)).reshape(natoms, 3)
            
            # Relevant documentation: https://docs.lammps.org/Howto_triclinic.html 
            xlb = xlo + min(0.0,xy,xz,xy+xz)
            xhb = xhi + max(0.0,xy,xz,xy+xz)
            ylb = ylo + min(0.0,yz)
            yhb = yhi + max(0.0,yz)
            zlb, zhb = zlo, zhi

            cell = np.zeros((3,3))
            cell[0] = xlb, xhb, xy
            cell[1] = ylb, yhb, xz
            cell[2] = zlb, zhb, yz

            self.datap = EmptyClass ()
            self.datap.xyz = x
            self.datap.cell = cell
            self.datap.natoms = natoms

        # update affine transformation instance to include the distorted state
        self.atrans = AffineTransform (self.data0.cell, self.datap.cell)
 
 
    def __call__ (self, void_check=None, export_path=None):
        if self.me == self.exporting_rank:
            announce ("Wigner-Seitz analysis")

        clock = time.time()
        
        amax = 8.0

        # bring distorted atoms into the frame of the reference W-S structure
        xref = self.atrans.go2to1_all(self.datap.xyz) # fairly quick, no need to parallelise
        neb_indices = self.ktree0.query (xref, k=1, distance_upper_bound=amax)[1] # runs in parallel

        # distribute indices into chunks across all threads
        nnebs = len(neb_indices)
        chunk = int(np.ceil(nnebs/self.nprocs))
        i0 = self.me*chunk
        ie = min((self.me+1)*chunk, nnebs) 

        occ_array_rank = np.zeros(self.data0.natoms, dtype=int)
        for _ix in np.array(neb_indices[i0:ie])[:,0]:
            occ_array_rank[_ix%self.data0.natoms] += 1

        if self.nprocs > 1:
            occ_array = self.comm.allreduce (occ_array_rank)
        else:
            occ_array = occ_array_rank 

        vac_indices = np.where(occ_array == 0)[0]
        int_indices = np.where(occ_array > 1)[0]
        int_occs = occ_array[occ_array > 1]

        clock = time.time() - clock
        if self.me == self.exporting_rank:
            print("Found W-S neighbours in %f seconds." % clock)
           
        nvacs = len(vac_indices)
        nints = np.sum(int_occs-1)

        if self.me == self.exporting_rank:
            print("Identified n_vac n_int = %d %d" % (nvacs, nints))

        # optionally reject identified vacancies if they are not inside a void region
        export_cell = self.data0.cell
        accepted_vacs = self.data0.xyz[vac_indices]  

        if void_check is not None:

            if self.me == self.exporting_rank:
                announce ("Checking if W-S vacancies are real vacancies.")

            # build kd-tree of distorted cell atoms
            clock = time.time()
            if self.me == self.exporting_rank:
                print ("Building kd-tree of distorted cell atoms... ", end='')

            #x_kdtree = Triclinic_KDTree_Buffer (self.datap.xyz, self.atrans.c2mat, r0=self.atrans.r2, balanced_tree=False)
            x_kdtree = Triclinic_KDTree_Buffer_MPI (self.datap.xyz, self.atrans.c2mat, r0=self.atrans.r2, 
                                                    my_rank=self.me, exporting_rank=self.exporting_rank, mpi_comm=self.comm,
                                                    balanced_tree=False)

            clock = time.time() - clock
            if self.me == self.exporting_rank:
                print ("Finished in %f seconds." % clock)

            # fetch vacancy candidate coordinates and transform to distorted cell space
            vac_pos = self.atrans.go1to2_all (accepted_vacs)

            # find all atomic neighbours to all candidate vacancies
            _sigma = 0.95*0.25*void_check
            kdnebs = x_kdtree.query_ball_point (vac_pos, void_check + 3*_sigma, return_distance_vectors=True)[1]

            # optimise all vacancy positions
            clock = time.time()
            if self.me == self.exporting_rank:
                print ("Optimising vacancy positions... ", end='')

            # distribute vacancies into chunks across all threads
            chunk = int(np.ceil(len(kdnebs)/self.nprocs))
            i0 = self.me*chunk
            ie = min((self.me+1)*chunk, len(kdnebs))

            allres_rank = [scipy.optimize.minimize(jcost_function_optnebs, np.zeros(3, dtype=float), 
                                                   args=(0.95*void_check, _knebs), method='CG', jac=jcost_function_optnebs_deriv,
                                                   options={'gtol': 1e-3}) for _knebs in kdnebs[i0:ie]]
            clock = time.time() - clock

            '''
            print ("Optimisation information:")
            for i,res in enumerate(allres_rank):
                if res.fun < 0.1:
                    print ("Vacancy Number %d:" % i)
                    print ("initial density:  %12.6f" % jcost_function_optnebs(np.zeros(3, dtype=float), 0.95*void_check, kdnebs[i]))
                    print ("optimal density:  %12.6f" % res.fun)
                    print ("initial position: %12.6f %12.6f %12.6f" % tuple(vac_pos[i]))
                    print ("optimal position: %12.6f %12.6f %12.6f" % tuple(vac_pos[i] - res.x))
                    print ()
            '''
 
            if self.me == self.exporting_rank:
                print ("Finished in %f seconds." % clock)

            # only accept vacancies lying in low-density regions
            accepted_vacs_rank = np.array([vac_pos[i0+i] - res.x for i,res in enumerate(allres_rank) if res.fun < 0.15])
            if accepted_vacs_rank.shape == (0,):
                accepted_vacs_rank = np.zeros((0, 3), dtype=float)

            # distribute all accepted vacancies to all cores 
            if self.nprocs > 1:
                accepted_vacs = self.comm.allgather (accepted_vacs_rank)
                accepted_vacs = np.concatenate (accepted_vacs)
            else:
                accepted_vacs = accepted_vacs_rank

            # we must also check if any two vacancies ended up in the same location, and then delete them
            # this may happen if W-S creates false positive vacs, that end up in the same few real vacs during optimisation
            # we do not need a triclinic KDtree for this as we did not do any PBC rewrapping
            vac_kdtree = cKDTree (accepted_vacs)

            unique_vacs = [] 
            for _vi,_vac in enumerate(accepted_vacs):
                nebs = vac_kdtree.query_ball_point(_vac, 1.0, return_sorted=True)
                if len(nebs) == 1:
                    unique_vacs += nebs 
                else:
                    skip = False 
                    for nn in nebs:
                        if nn in unique_vacs:
                            skip = True
                            break
                    if not skip:
                        unique_vacs += [nebs[0]]

            if self.me == self.exporting_rank:
                print ("Found %d vacancies, of which %d are unique." % (len(accepted_vacs), len(unique_vacs)))
            accepted_vacs = accepted_vacs[unique_vacs] 

            if self.me == self.exporting_rank:
                announce ("Identified N=%d vacancies as lying in void regions" % len(accepted_vacs))
            sys.stdout.flush()
 
            export_cell = self.datap.cell

        if len(accepted_vacs) == 0:
            accepted_vacs = np.empty((0,3))

        xyz_array = np.r_[self.data0.xyz[int_indices], accepted_vacs]
        occ_array = np.r_[int_occs, [0]*len(accepted_vacs)]

        sys.stdout.flush()

        if export_path:
            if self.me == self.exporting_rank:
                print ()
                print ("Exporting W-S output to file %s... " % export_path, end="")
                write_dump (export_cell, np.ones(len(xyz_array)).astype(int), occ_array, xyz_array, export_path, 0)
                print ("done.")
                print ()

        self.comm.barrier()

        return xyz_array, np.ones(len(xyz_array)).astype(int), occ_array
    

class Triclinic_KDTree_Buffer_MPI:
    def __init__(self, xyz, cellvecs, amax=10.0, r0=None, my_rank=0, exporting_rank=0, mpi_comm=None, **kwargs):

        # if not given MPI comm, construct a fake comm instance that does nothing, i.e. monkey patching
        if mpi_comm is not None:
            self.comm = mpi_comm
            self.nprocs = self.comm.Get_size()
        else:
            self.comm = EmptyClass ()
            self.comm.bcast = lambda data, root=None: data 
            self.comm.barrier = lambda *args: args
            self.nprocs = 1 

        # store MPI information, i.e. thread rank and MPI comm instance as attributes
        self.me = my_rank
        self.exporting_rank = exporting_rank 

        self.r0 = r0
        self.cmat = cellvecs
        if cellvecs is not None:
            self.cmati = np.linalg.inv(self.cmat)
            
        npts = len(xyz)

        # create a buffer region
        ids = np.r_[:npts]
         
        # distribute points into chunks across all threads
        chunk = int(np.ceil(npts/self.nprocs))
        i0 = self.me*chunk
        ie = min((self.me+1)*chunk, npts) 
        npt = ie-i0

        # split points amongst chunks, convert into fractional coordinates, and wrap back into box
        xyz_rank = xyz[i0:ie]
        ids_rank = ids[i0:ie]
        frac = self._cart_to_frac_all(xyz_rank)
        frac = frac - np.floor(frac)

        # estimating buffer region width in fractional coordinates        
        limmat = self._cart_to_frac_all(amax*np.identity(3) + self.r0)
        flims = np.linalg.norm(limmat, axis=0)

        fboolsL = frac < flims
        fboolsR = frac > (1.-flims)

        fbuffer = []
        ibuffer = []
        
        # cube sides
        fbuffer += [frac[fboolsL[:,0]] + np.r_[ 1, 0, 0]] # (0,y,z)
        fbuffer += [frac[fboolsL[:,1]] + np.r_[ 0, 1, 0]] # (x,0,z)
        fbuffer += [frac[fboolsL[:,2]] + np.r_[ 0, 0, 1]] # (x,y,0)

        fbuffer += [frac[fboolsR[:,0]] + np.r_[-1, 0, 0]] # (1,y,z)
        fbuffer += [frac[fboolsR[:,1]] + np.r_[ 0,-1, 0]] # (x,1,z)
        fbuffer += [frac[fboolsR[:,2]] + np.r_[ 0, 0,-1]] # (x,y,1)

        ibuffer += [ids_rank[fboolsL[:,0]]]
        ibuffer += [ids_rank[fboolsL[:,1]]]
        ibuffer += [ids_rank[fboolsL[:,2]]]

        ibuffer += [ids_rank[fboolsR[:,0]]]
        ibuffer += [ids_rank[fboolsR[:,1]]]
        ibuffer += [ids_rank[fboolsR[:,2]]]
        
        # cube edges
        fbuffer += [frac[fboolsL[:,0]*fboolsL[:,1]] + np.r_[ 1, 1, 0]] # (0,0,z)
        fbuffer += [frac[fboolsR[:,0]*fboolsL[:,1]] + np.r_[-1, 1, 0]] # (1,0,z)
        fbuffer += [frac[fboolsL[:,0]*fboolsR[:,1]] + np.r_[ 1,-1, 0]] # (0,1,z)
        fbuffer += [frac[fboolsR[:,0]*fboolsR[:,1]] + np.r_[-1,-1, 0]] # (1,1,z)

        fbuffer += [frac[fboolsL[:,0]*fboolsL[:,2]] + np.r_[ 1, 0, 1]] # (0,y,0)
        fbuffer += [frac[fboolsR[:,0]*fboolsL[:,2]] + np.r_[-1, 0, 1]] # (1,y,0)
        fbuffer += [frac[fboolsL[:,0]*fboolsR[:,2]] + np.r_[ 1, 0,-1]] # (0,y,1)
        fbuffer += [frac[fboolsR[:,0]*fboolsR[:,2]] + np.r_[-1, 0,-1]] # (1,y,1)

        fbuffer += [frac[fboolsL[:,1]*fboolsL[:,2]] + np.r_[ 0, 1, 1]] # (x,0,0)
        fbuffer += [frac[fboolsR[:,1]*fboolsL[:,2]] + np.r_[ 0,-1, 1]] # (x,1,0)
        fbuffer += [frac[fboolsL[:,1]*fboolsR[:,2]] + np.r_[ 0, 1,-1]] # (x,0,1)
        fbuffer += [frac[fboolsR[:,1]*fboolsR[:,2]] + np.r_[ 0,-1,-1]] # (x,1,1)

        ibuffer += [ids_rank[fboolsL[:,0]*fboolsL[:,1]]]
        ibuffer += [ids_rank[fboolsR[:,0]*fboolsL[:,1]]]
        ibuffer += [ids_rank[fboolsL[:,0]*fboolsR[:,1]]]
        ibuffer += [ids_rank[fboolsR[:,0]*fboolsR[:,1]]]
   
        ibuffer += [ids_rank[fboolsL[:,0]*fboolsL[:,2]]]
        ibuffer += [ids_rank[fboolsR[:,0]*fboolsL[:,2]]]
        ibuffer += [ids_rank[fboolsL[:,0]*fboolsR[:,2]]]
        ibuffer += [ids_rank[fboolsR[:,0]*fboolsR[:,2]]]
        
        ibuffer += [ids_rank[fboolsL[:,1]*fboolsL[:,2]]]
        ibuffer += [ids_rank[fboolsR[:,1]*fboolsL[:,2]]]
        ibuffer += [ids_rank[fboolsL[:,1]*fboolsR[:,2]]]
        ibuffer += [ids_rank[fboolsR[:,1]*fboolsR[:,2]]]

        # cube corners
        fbuffer += [frac[fboolsL[:,0]*fboolsL[:,1]*fboolsL[:,2]] + np.r_[ 1, 1, 1]] # (0,0,0)
        fbuffer += [frac[fboolsL[:,0]*fboolsL[:,1]*fboolsR[:,2]] + np.r_[ 1, 1,-1]] # (0,0,1)
        fbuffer += [frac[fboolsL[:,0]*fboolsR[:,1]*fboolsL[:,2]] + np.r_[ 1,-1, 1]] # (0,1,0)
        fbuffer += [frac[fboolsL[:,0]*fboolsR[:,1]*fboolsR[:,2]] + np.r_[ 1,-1,-1]] # (0,1,1)
        fbuffer += [frac[fboolsR[:,0]*fboolsL[:,1]*fboolsL[:,2]] + np.r_[-1, 1, 1]] # (1,0,0)
        fbuffer += [frac[fboolsR[:,0]*fboolsL[:,1]*fboolsR[:,2]] + np.r_[-1, 1,-1]] # (1,0,1)
        fbuffer += [frac[fboolsR[:,0]*fboolsR[:,1]*fboolsL[:,2]] + np.r_[-1,-1, 1]] # (1,1,0)
        fbuffer += [frac[fboolsR[:,0]*fboolsR[:,1]*fboolsR[:,2]] + np.r_[-1,-1,-1]] # (1,1,1)

        ibuffer += [ids_rank[fboolsL[:,0]*fboolsL[:,1]*fboolsL[:,2]]]
        ibuffer += [ids_rank[fboolsL[:,0]*fboolsL[:,1]*fboolsR[:,2]]]
        ibuffer += [ids_rank[fboolsL[:,0]*fboolsR[:,1]*fboolsL[:,2]]]
        ibuffer += [ids_rank[fboolsL[:,0]*fboolsR[:,1]*fboolsR[:,2]]]
        ibuffer += [ids_rank[fboolsR[:,0]*fboolsL[:,1]*fboolsL[:,2]]]
        ibuffer += [ids_rank[fboolsR[:,0]*fboolsL[:,1]*fboolsR[:,2]]]
        ibuffer += [ids_rank[fboolsR[:,0]*fboolsR[:,1]*fboolsL[:,2]]]
        ibuffer += [ids_rank[fboolsR[:,0]*fboolsR[:,1]*fboolsR[:,2]]]

        # merge together
        fbuffer = np.concatenate(fbuffer)
        ibuffer = np.concatenate(ibuffer)
        
        # convert back to cartesian xyz
        xyzbuffer = self._frac_to_cart_all(fbuffer)
        
        # join together
        dcopy_rank = np.r_[xyz_rank, xyzbuffer]
        ids_rank = np.r_[ids_rank, ibuffer]

        # gather data together on all ranks
        self.dcopy = self.comm.allgather (dcopy_rank)
        self.dcopy = np.concatenate (self.dcopy) 

        self.ids = self.comm.allgather (ids_rank)
        self.ids = np.concatenate (self.ids) 

        self.npts = npts
        self.amax = amax

        # build a kd-tree for neighbour search on each thread
        # this is not parallelised! it would need a dedicated MPI aware KD-Tree construction routine
        self.kdtree = cKDTree (self.dcopy, **kwargs)
        
    def query_ball_point (self, xyz, rcut, return_distance_vectors=False, *args, **kwargs):

        # distribute points into chunks across all threads
        npts = len(xyz)
        chunk = int(np.ceil(npts/self.nprocs))
        i0 = self.me*chunk
        ie = min((self.me+1)*chunk, npts) 
        npt = ie-i0

        # split points amongst chunks and find neighbours
        xyz_rank = xyz[i0:ie]
        pnebs_rank = self.kdtree.query_ball_point (xyz_rank, rcut, *args, **kwargs)

        nebs_rank = [self.ids[pi] for pi in pnebs_rank]

        if return_distance_vectors:
            dvecs_rank = [self.dcopy[pnebs_rank[i]] - xi for i,xi in enumerate(xyz_rank)]

        # gather data together on all ranks
        if self.nprocs > 1:
            nebs = self.comm.allgather (nebs_rank)
            nebs = sum (nebs, [])
        else:
            nebs = nebs_rank

        if return_distance_vectors:
            if self.nprocs > 1:
                dvecs = self.comm.allgather (dvecs_rank)
                dvecs = sum (dvecs, []) 
            else:
                dvecs = dvecs_rank 

            return nebs, dvecs
        else: 
            return nebs

    def query (self, xyz, return_distance_vectors=False, *args, **kwargs):

        # distribute points into chunks across all threads
        npts = len(xyz)
        chunk = int(np.ceil(npts/self.nprocs))
        i0 = self.me*chunk
        ie = min((self.me+1)*chunk, npts) 
        npt = ie-i0

        # split points amongst chunks and find neighbours
        xyz_rank = xyz[i0:ie]
        dist_rank,pnebs_rank = self.kdtree.query (xyz_rank, *args, **kwargs) 

        # unsqueeze the last dimension if k=1 (kdtree.query behaviour)
        if 'k' in kwargs:
            if kwargs['k'] == 1:
                dist_rank = np.expand_dims(dist_rank, axis=1)
                pnebs_rank = np.expand_dims(pnebs_rank, axis=1)

        pnebs_rank[np.isinf(dist_rank)] = 0 # cap infinite distance indices to avoid index error
        nebs_rank = self.ids[pnebs_rank]
        nebs_rank[np.isinf(dist_rank)] = len(nebs_rank) # restore infinite distance indices 
      
        # obtain neighbour vectors 
        if return_distance_vectors:
            dvecs_rank = np.array([xnebs - xyz_rank[i] for i,xnebs in enumerate(self.dcopy[pnebs_rank])])
            dvecs_rank[np.isinf(dist_rank)] = np.inf
 
        # gather data together on all ranks
        dist = self.comm.allgather (dist_rank)
        dist = np.concatenate (dist)

        nebs = self.comm.allgather (nebs_rank)
        nebs = np.concatenate (nebs)
      
        if return_distance_vectors:
            dvecs = self.comm.allgather (dvecs_rank)
            dvecs = np.concatenate (dvecs)
            return dist, nebs, dvecs
        else:
            return dist, nebs
    
    
    def _cart_to_frac_all(self,xyz):
        '''Transform all xyz vectors to fractional coordinates.'''
        return np.r_[[self.cmati[0][0]*(xyz[:,0] - self.r0[0]) +
                      self.cmati[0][1]*(xyz[:,1] - self.r0[0]) +
                      self.cmati[0][2]*(xyz[:,2] - self.r0[0])],
                     [self.cmati[1][0]*(xyz[:,0] - self.r0[1]) +
                      self.cmati[1][1]*(xyz[:,1] - self.r0[1]) +
                      self.cmati[1][2]*(xyz[:,2] - self.r0[1])],
                     [self.cmati[2][0]*(xyz[:,0] - self.r0[2]) +
                      self.cmati[2][1]*(xyz[:,1] - self.r0[2]) +
                      self.cmati[2][2]*(xyz[:,2] - self.r0[2])]].T        
        
    
    def _frac_to_cart_all(self,frac):
        '''Transform all fractional coordinates to cartesian vectors.'''
        return np.r_[[self.cmat[0][0]*frac[:,0] +
                      self.cmat[0][1]*frac[:,1] +
                      self.cmat[0][2]*frac[:,2] + self.r0[0]],
                     [self.cmat[1][0]*frac[:,0] +
                      self.cmat[1][1]*frac[:,1] +
                      self.cmat[1][2]*frac[:,2] + self.r0[1]],
                     [self.cmat[2][0]*frac[:,0] +
                      self.cmat[2][1]*frac[:,1] +
                      self.cmat[2][2]*frac[:,2] + self.r0[2]]].T  





class Triclinic_KDTree_Buffer:
    def __init__(self, xyz, cellvecs, amax=10.0, r0=None, **kwargs):

        self.r0 = r0
        self.cmat = cellvecs
        if cellvecs is not None:
            self.cmati = np.linalg.inv(self.cmat)
            
        npts = len(xyz)

        # create a buffer region
        ids = np.r_[:npts]
         
        frac = self._cart_to_frac_all(xyz)
        frac = frac - np.floor(frac) # wrap back into box
        
        limmat = self._cart_to_frac_all(amax*np.identity(3) + self.r0)
        flims = np.linalg.norm(limmat, axis=0)
        #flims = self._cart_to_frac_all(np.array([amax*np.ones(3) + self.r0]))

        fboolsL = frac < flims
        fboolsR = frac > (1.-flims)

        fbuffer = []
        ibuffer = []
        
        origin = [["original"]*npts]
        
        # cube sides
        fbuffer += [frac[fboolsL[:,0]] + np.r_[ 1, 0, 0]] # (0,y,z)
        fbuffer += [frac[fboolsL[:,1]] + np.r_[ 0, 1, 0]] # (x,0,z)
        fbuffer += [frac[fboolsL[:,2]] + np.r_[ 0, 0, 1]] # (x,y,0)

        fbuffer += [frac[fboolsR[:,0]] + np.r_[-1, 0, 0]] # (1,y,z)
        fbuffer += [frac[fboolsR[:,1]] + np.r_[ 0,-1, 0]] # (x,1,z)
        fbuffer += [frac[fboolsR[:,2]] + np.r_[ 0, 0,-1]] # (x,y,1)

        ibuffer += [ids[fboolsL[:,0]]]
        ibuffer += [ids[fboolsL[:,1]]]
        ibuffer += [ids[fboolsL[:,2]]]

        ibuffer += [ids[fboolsR[:,0]]]
        ibuffer += [ids[fboolsR[:,1]]]
        ibuffer += [ids[fboolsR[:,2]]]
        
        origin += [["0yx"]*len(ids[fboolsL[:,0]])]
        origin += [["x0z"]*len(ids[fboolsL[:,1]])]
        origin += [["xy0"]*len(ids[fboolsL[:,2]])]
        
        origin += [["1yz"]*len(ids[fboolsR[:,0]])]
        origin += [["x1z"]*len(ids[fboolsR[:,1]])]
        origin += [["xy1"]*len(ids[fboolsR[:,2]])]
                    
        # cube edges
        fbuffer += [frac[fboolsL[:,0]*fboolsL[:,1]] + np.r_[ 1, 1, 0]] # (0,0,z)
        fbuffer += [frac[fboolsR[:,0]*fboolsL[:,1]] + np.r_[-1, 1, 0]] # (1,0,z)
        fbuffer += [frac[fboolsL[:,0]*fboolsR[:,1]] + np.r_[ 1,-1, 0]] # (0,1,z)
        fbuffer += [frac[fboolsR[:,0]*fboolsR[:,1]] + np.r_[-1,-1, 0]] # (1,1,z)

        fbuffer += [frac[fboolsL[:,0]*fboolsL[:,2]] + np.r_[ 1, 0, 1]] # (0,y,0)
        fbuffer += [frac[fboolsR[:,0]*fboolsL[:,2]] + np.r_[-1, 0, 1]] # (1,y,0)
        fbuffer += [frac[fboolsL[:,0]*fboolsR[:,2]] + np.r_[ 1, 0,-1]] # (0,y,1)
        fbuffer += [frac[fboolsR[:,0]*fboolsR[:,2]] + np.r_[-1, 0,-1]] # (1,y,1)

        fbuffer += [frac[fboolsL[:,1]*fboolsL[:,2]] + np.r_[ 0, 1, 1]] # (x,0,0)
        fbuffer += [frac[fboolsR[:,1]*fboolsL[:,2]] + np.r_[ 0,-1, 1]] # (x,1,0)
        fbuffer += [frac[fboolsL[:,1]*fboolsR[:,2]] + np.r_[ 0, 1,-1]] # (x,0,1)
        fbuffer += [frac[fboolsR[:,1]*fboolsR[:,2]] + np.r_[ 0,-1,-1]] # (x,1,1)

        ibuffer += [ids[fboolsL[:,0]*fboolsL[:,1]]]
        ibuffer += [ids[fboolsR[:,0]*fboolsL[:,1]]]
        ibuffer += [ids[fboolsL[:,0]*fboolsR[:,1]]]
        ibuffer += [ids[fboolsR[:,0]*fboolsR[:,1]]]
   
        ibuffer += [ids[fboolsL[:,0]*fboolsL[:,2]]]
        ibuffer += [ids[fboolsR[:,0]*fboolsL[:,2]]]
        ibuffer += [ids[fboolsL[:,0]*fboolsR[:,2]]]
        ibuffer += [ids[fboolsR[:,0]*fboolsR[:,2]]]
        
        ibuffer += [ids[fboolsL[:,1]*fboolsL[:,2]]]
        ibuffer += [ids[fboolsR[:,1]*fboolsL[:,2]]]
        ibuffer += [ids[fboolsL[:,1]*fboolsR[:,2]]]
        ibuffer += [ids[fboolsR[:,1]*fboolsR[:,2]]]

        origin += [["00z"]*len(ids[fboolsL[:,0]*fboolsL[:,1]])]
        origin += [["10z"]*len(ids[fboolsR[:,0]*fboolsL[:,1]])]
        origin += [["01z"]*len(ids[fboolsL[:,0]*fboolsR[:,1]])]
        origin += [["11z"]*len(ids[fboolsR[:,0]*fboolsR[:,1]])]

        origin += [["0y0"]*len(ids[fboolsL[:,0]*fboolsL[:,2]])]
        origin += [["1y0"]*len(ids[fboolsR[:,0]*fboolsL[:,2]])]
        origin += [["0y1"]*len(ids[fboolsL[:,0]*fboolsR[:,2]])]
        origin += [["1y1"]*len(ids[fboolsR[:,0]*fboolsR[:,2]])]

        origin += [["x00"]*len(ids[fboolsL[:,1]*fboolsL[:,2]])]
        origin += [["x10"]*len(ids[fboolsR[:,1]*fboolsL[:,2]])]
        origin += [["x01"]*len(ids[fboolsL[:,1]*fboolsR[:,2]])]
        origin += [["x11"]*len(ids[fboolsR[:,1]*fboolsR[:,2]])]
        
        # cube corners
        fbuffer += [frac[fboolsL[:,0]*fboolsL[:,1]*fboolsL[:,2]] + np.r_[ 1, 1, 1]] # (0,0,0)
        fbuffer += [frac[fboolsL[:,0]*fboolsL[:,1]*fboolsR[:,2]] + np.r_[ 1, 1,-1]] # (0,0,1)
        fbuffer += [frac[fboolsL[:,0]*fboolsR[:,1]*fboolsL[:,2]] + np.r_[ 1,-1, 1]] # (0,1,0)
        fbuffer += [frac[fboolsL[:,0]*fboolsR[:,1]*fboolsR[:,2]] + np.r_[ 1,-1,-1]] # (0,1,1)
        fbuffer += [frac[fboolsR[:,0]*fboolsL[:,1]*fboolsL[:,2]] + np.r_[-1, 1, 1]] # (1,0,0)
        fbuffer += [frac[fboolsR[:,0]*fboolsL[:,1]*fboolsR[:,2]] + np.r_[-1, 1,-1]] # (1,0,1)
        fbuffer += [frac[fboolsR[:,0]*fboolsR[:,1]*fboolsL[:,2]] + np.r_[-1,-1, 1]] # (1,1,0)
        fbuffer += [frac[fboolsR[:,0]*fboolsR[:,1]*fboolsR[:,2]] + np.r_[-1,-1,-1]] # (1,1,1)

        ibuffer += [ids[fboolsL[:,0]*fboolsL[:,1]*fboolsL[:,2]]]
        ibuffer += [ids[fboolsL[:,0]*fboolsL[:,1]*fboolsR[:,2]]]
        ibuffer += [ids[fboolsL[:,0]*fboolsR[:,1]*fboolsL[:,2]]]
        ibuffer += [ids[fboolsL[:,0]*fboolsR[:,1]*fboolsR[:,2]]]
        ibuffer += [ids[fboolsR[:,0]*fboolsL[:,1]*fboolsL[:,2]]]
        ibuffer += [ids[fboolsR[:,0]*fboolsL[:,1]*fboolsR[:,2]]]
        ibuffer += [ids[fboolsR[:,0]*fboolsR[:,1]*fboolsL[:,2]]]
        ibuffer += [ids[fboolsR[:,0]*fboolsR[:,1]*fboolsR[:,2]]]

        origin += [["000"]*len(ids[fboolsL[:,0]*fboolsL[:,1]*fboolsL[:,2]])]
        origin += [["001"]*len(ids[fboolsL[:,0]*fboolsL[:,1]*fboolsR[:,2]])]
        origin += [["010"]*len(ids[fboolsL[:,0]*fboolsR[:,1]*fboolsL[:,2]])]
        origin += [["011"]*len(ids[fboolsL[:,0]*fboolsR[:,1]*fboolsR[:,2]])]
        origin += [["100"]*len(ids[fboolsR[:,0]*fboolsL[:,1]*fboolsL[:,2]])]
        origin += [["101"]*len(ids[fboolsR[:,0]*fboolsL[:,1]*fboolsR[:,2]])]
        origin += [["110"]*len(ids[fboolsR[:,0]*fboolsR[:,1]*fboolsL[:,2]])]
        origin += [["111"]*len(ids[fboolsR[:,0]*fboolsR[:,1]*fboolsR[:,2]])]
        
        # merge together
        fbuffer = np.concatenate(fbuffer)
        ibuffer = np.concatenate(ibuffer)
        origin = np.concatenate(origin)
        
        # convert back to cartesian xyz
        xyzbuffer = self._frac_to_cart_all(fbuffer)
        
        # join together
        self.dcopy = np.r_[xyz, xyzbuffer]
        self.ids = np.r_[ids, ibuffer]
        self.origin = origin
        
        # build kd-tree for neighbour search
        self.npts = npts
        self.amax = amax
        self.kdtree = cKDTree (self.dcopy, **kwargs)
        
    def query_ball_point (self, x, rcut, return_distance_vectors=False, *args, **kwargs):

        # unsqueeze the last dimension if k=1 (kdtree.query behaviour)
        x = np.array(x)
        if len(x.shape) == 1:
            unsqueeze = True
            x = [x] 

        pnebs = self.kdtree.query_ball_point (x, rcut, *args, **kwargs)
        nebs = [self.ids[pi] for pi in pnebs]

        # squeeze the last dimension again if k=1
        if unsqueeze: 
            nebs = nebs[0]

        if return_distance_vectors:
            dvecs = [self.dcopy[pnebs[i]] - xi for i,xi in enumerate(x)]
            if unsqueeze: 
                dvecs = dvecs[0]                
            return nebs, dvecs

        return nebs

    def query (self, x, return_distance_vectors=False, *args, **kwargs):
        dist,pnebs = self.kdtree.query (x, *args, **kwargs) 

        # unsqueeze the last dimension if k=1 (kdtree.query behaviour)
        if 'k' in kwargs:
            if kwargs['k'] == 1:
                dist = np.expand_dims(dist, axis=1)
                pnebs = np.expand_dims(pnebs, axis=1)

        pnebs[np.isinf(dist)] = 0 # cap infinite distance indices to avoid index error
        nebs = self.ids[pnebs]
        nebs[np.isinf(dist)] = len(nebs) # restore infinite distance indices 
        
        if return_distance_vectors:
            dvecs = np.array([xnebs - x[i] for i,xnebs in enumerate(self.dcopy[pnebs])])
            dvecs[np.isinf(dist)] = np.inf
            return dist, nebs, dvecs
        
        return dist,nebs
    
    
    def _cart_to_frac_all(self,xyz):
        '''Transform all xyz vectors to fractional coordinates.'''
        return np.r_[[self.cmati[0][0]*(xyz[:,0] - self.r0[0]) +
                      self.cmati[0][1]*(xyz[:,1] - self.r0[0]) +
                      self.cmati[0][2]*(xyz[:,2] - self.r0[0])],
                     [self.cmati[1][0]*(xyz[:,0] - self.r0[1]) +
                      self.cmati[1][1]*(xyz[:,1] - self.r0[1]) +
                      self.cmati[1][2]*(xyz[:,2] - self.r0[1])],
                     [self.cmati[2][0]*(xyz[:,0] - self.r0[2]) +
                      self.cmati[2][1]*(xyz[:,1] - self.r0[2]) +
                      self.cmati[2][2]*(xyz[:,2] - self.r0[2])]].T        
        
    
    def _frac_to_cart_all(self,frac):
        '''Transform all fractional coordinates to cartesian vectors.'''
        return np.r_[[self.cmat[0][0]*frac[:,0] +
                      self.cmat[0][1]*frac[:,1] +
                      self.cmat[0][2]*frac[:,2] + self.r0[0]],
                     [self.cmat[1][0]*frac[:,0] +
                      self.cmat[1][1]*frac[:,1] +
                      self.cmat[1][2]*frac[:,2] + self.r0[1]],
                     [self.cmat[2][0]*frac[:,0] +
                      self.cmat[2][1]*frac[:,1] +
                      self.cmat[2][2]*frac[:,2] + self.r0[2]]].T  




def cluster_analysis (xyz, rcut, cellvecs=None, r0=np.zeros(3), my_rank=0, exporting_rank=0, mpi_comm=None):

    # build kd-tree for neighbour search
    npts = len(xyz)
    kdtree = Triclinic_KDTree_Buffer (xyz, cellvecs, r0=r0, balanced_tree=False)

    '''
    # define unique atomic ids of periodically repeated atoms
    npts = len(xyz)
    repeated_ids = np.r_[[i for i in range(npts)]]

    # duplicate atoms for periodic images    
    dcopy = np.copy(xyz)
    ids = np.copy(repeated_ids)

    if cellvecs is not None:
        for ix in [-1,0,1]:
            for iy in [-1,0,1]:
                for iz in [-1,0,1]:
                    if ix == iy == iz == 0:
                        continue
                    _dcopy = xyz + ix*cellvecs[:,0] + iy*cellvecs[:,1] + iz*cellvecs[:,2]
                    dcopy = np.r_[dcopy, _dcopy]
                    repeated_ids = np.r_[repeated_ids, ids]
    else:
        repeated_ids = ids

    # build kd-tree for neighbour search
    kdtree = cKDTree(dcopy)
    '''

    # start looking for clusters
    assigned_ids = []
    unassigned_ids = [i for i in range(npts)]
    clusters = []   

    for _id in range(npts):#unassigned_ids:
        if _id in assigned_ids:
            continue
        else:
            assigned_ids += [_id]
            cluster = [_id]

        to_check = [_id]
        while len(to_check) > 0:

            # look for neighbours, fetch true indices and remove self-neighbour
            nebs = kdtree.query_ball_point (xyz[to_check[0]], rcut, return_sorted=True)
            #print ("nebs before:", nebs)
            #print ("repeated_ids:", repeated_ids)
            nebs = [repeated_ids[neb] for neb in nebs if neb != to_check[0]]
            #print ("nebs after:", nebs)

            _unassigned = list(np.setdiff1d(nebs, assigned_ids))
            #print ("unassigned:", _unassigned)

            assigned_ids += _unassigned 
            to_check += _unassigned
            cluster += _unassigned
            to_check.pop(0)

            #print ("assigned_ids:", assigned_ids)
            #print ("to_check:", to_check)
            #print ("cluster:", cluster)

            if len(assigned_ids) == npts:
                break

        clusters += [cluster]

    return clusters
    
 

