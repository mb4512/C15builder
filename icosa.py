import os
import sys
import json
import glob

import numpy as np
import scipy

from lammps import lammps
from lib.eaminfo import Import_eamfs
from lib.lmp_ws import WignerSeitz


# template to replace MPI functionality for single-threaded use
class MPI_to_serial():
    def bcast(self, *args, **kwargs):
        return args[0]
    def barrier(self):
        return 0

# try running in parallel, otherwise single thread
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    me = comm.Get_rank()
    nprocs = comm.Get_size()
    mode = 'MPI'
except:
    me = 0
    nprocs = 1
    comm = MPI_to_serial()
    mode = 'serial'

def mpiprint(*arg):
    if me == 0:
        print(*arg)
        sys.stdout.flush()
    return 0


def announce(string):
    mpiprint ()
    mpiprint ("=================================================")
    mpiprint (string)
    mpiprint ("=================================================")
    mpiprint ()
    return 0 



def main():

    def full_ico (sign):
        # int and vac coordinates for an icosahedron centred on an atom at position (0,0,0) 

        # network node positions spanned by interstitials (in units of a) 
        interstitials = np.array([[   1/4,  -1/4, 1-1/4],
                                  [  -1/4,   1/4, 1-1/4],
                                  [  -1/4,  -1/4,-1+1/4],
                                  [   1/4,   1/4,-1+1/4],
                                  [ 1-1/4,  -1/4,   1/4],
                                  [ 1-1/4,   1/4,  -1/4],
                                  [-1+1/4,   1/4,   1/4],
                                  [-1+1/4,  -1/4,  -1/4],
                                  [  -1/4, 1-1/4,   1/4],
                                  [   1/4, 1-1/4,  -1/4],
                                  [   1/4,-1+1/4,   1/4],
                                  [  -1/4,-1+1/4,  -1/4]])

        # vacancy coordinates
        vacancies = np.array([[ 0, 0, 1],
                              [ 0, 0,-1],
                              [ 1, 0, 0],
                              [-1, 0, 0],
                              [ 0, 1, 0],
                              [ 0,-1, 0],
                              [ 1/2,-1/2, 1/2],
                              [-1/2, 1/2, 1/2],
                              [ 1/2, 1/2,-1/2],
                              [-1/2,-1/2,-1/2]])

        
        vacancies *= sign
        interstitials *= sign

        return vacancies, interstitials


 
    def fetch_motive (dvec):

        motiveA = np.array([[ 1, 1, 1],
                            [-1,-1, 1],
                            [ 1,-1,-1],
                            [-1, 1,-1]])
        motiveA = motiveA/np.sqrt(3)

        if (np.dot(motiveA, dvec/np.linalg.norm(dvec)) > 0.99).any():
            return -1
        else:
            return 1


    # wrap back into box
    def wrap_back_unique (ws, xyz, offset):
        frac = ws.atrans.cart_to_frac1_all (offset + xyz)
        for i in range(3):
            frac[frac[:,i]>1,i] -= 1
            frac[frac[:,i]<0,i] += 1

        # only keep unique
        frac = np.unique (frac, axis=0)

        return ws.atrans.frac_to_cart1_all (frac) - offset 

    def unique_xyz (xyz):
        kdtree = scipy.spatial.KDTree (xyz)
        unique_xyz = [] 
        for _xi,_xyz in enumerate(xyz):
            nebs = kdtree.query_ball_point([_xyz], 0.01, return_sorted=True)
            nebs = nebs[0]
            if len(nebs) == 1:
                unique_xyz += [nebs[0]] 
            else:
                skip = False 
                for nn in nebs:
                    if nn in unique_xyz:
                        skip = True
                        break
                if not skip:
                    unique_xyz += [nebs[0]]

        return xyz[unique_xyz] 


    mpiprint ('''Constructing C15 clusters.''')

    # -------------------
    #  IMPORT PARAMETERS    
    # -------------------

    # here, input parameters are read, dump files are imported etc.
    inputfile = sys.argv[1] 
  
    all_input = None 
    if (me == 0):
        with open(inputfile) as fp:
            all_input = json.loads(fp.read())
    all_input = comm.bcast(all_input, root=0)

    nint = int(sys.argv[2])

    # -----------------------
    #  JOB INPUT PARAMETERS 
    # -----------------------

    job_name = all_input['job_name']
    
    potdir  = all_input['potential_path']
    potname = all_input['potential']
    atype   = all_input['atomtype']

    scrdir = all_input["scratch"]

    mpiprint ("Running in %s mode." % mode)
    mpiprint ("Job %s running on %s cores.\n" % (job_name, nprocs))
    
    mpiprint ("Parameter input file %s:\n" % inputfile)
    for key in all_input:
        mpiprint ("    %s: %s" % (key, all_input[key]))
    mpiprint()

    comm.barrier()

    # -------------------
    #  SIMULATION POTENTIAL    
    # ------------------- 

    potfile = potdir + potname

    # Any EAM fs or alloy potential file will be scraped for lattice parameters etc
    if (me == 0): 
        potential = Import_eamfs(potfile)
    else:
        potential = None
    
    # broadcast imported data to all cores 
    potential = comm.bcast(potential, root=0)

    mpiprint ('''\nPotential information:
    Elements, %s,
    mass: %s,
    Z number: %s,
    lattice: %s,
    crystal: %s,
    cutoff: %s
    ''' % ( 
        tuple(potential.elements), tuple(potential.mass.values()), tuple(potential.znum.values()), 
        tuple(potential.alat.values()), tuple(potential.atyp.values()), 
        potential.cutoff)
    )   

    # fetch potential values for host lattice
    ele      = potential.elements[atype-1] # element name
    mass     = potential.mass[atype-1] # mass
    znum     = potential.znum[atype-1] # z number
    lattice  = potential.atyp[atype-1] # crystal structure
    alattice = potential.alat[atype-1] # lattice constant

    # -----------------------
    #  SIMULATION PARAMETERS
    # -----------------------

    # energy convergence tolerance (if energy minimisation is done)
    etol = float(all_input["etol"])
    etolstring = "%.5e" % etol

    # path to reference file for W-S analysis
    if "reference" in all_input: 
        reference = all_input["reference"]
    else:
        reference = None

    # all INTEGER lattice vectors for LAMMPS lattice orientation
    ix = np.r_[1,0,0]
    iy = np.r_[0,1,0]
    iz = np.r_[0,0,1]

    nx = all_input['nx']
    ny = all_input['ny']
    nz = all_input['nz']

    # lattice vector norms
    sx = np.linalg.norm(ix)
    sy = np.linalg.norm(iy)
    sz = np.linalg.norm(iz)

    # Check for right-handedness in basis
    if np.r_[ix].dot(np.cross(np.r_[iy],np.r_[iz])) < 0:
        mpiprint ("Left Handed Basis!\n\n y -> -y:\t",iy,"->",)
        for i in range(3):
            iy[i] *= -1
        mpiprint (iy,"\n\n")

    # -----------------------
    # INITIALISE LAMMPS  
    # -----------------------

    # Start LAMMPS instance
    lmp = lammps()

    lmp.command('# Lammps input file')
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('atom_modify map array sort 0 0.0')

    # initialise lattice
    lmp.command('lattice %s %f orient x %d %d %d orient y %d %d %d orient z %d %d %d' % (
                lattice,
                alattice,
                ix[0], ix[1], ix[2],
                iy[0], iy[1], iy[2],
                iz[0], iz[1], iz[2]))

    # cubic simulation cell region
    lmp.command('region rsimbox block 0 %d 0 %d 0 %d units lattice' % (nx, ny, nz))
    lmp.command('create_box 1 rsimbox')
    lmp.command('create_atoms 1 region rsimbox')
  
    # potential
    lmp.command('mass 1 %f' % mass)
    lmp.command('pair_style eam/fs')
    lmp.command('pair_coeff * * %s %s' % (potfile, ele))
    lmp.command('neighbor 1.0 bin')

    lmp.command('thermo 100')
    lmp.command('thermo_style custom step pe pxx pyy pzz pxy pxz pyz')
    lmp.command("thermo_modify line one format line '%8d %9.3f %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e'")

    # keep box centre of mass from drifting
    lmp.command("fix frecenter all recenter INIT INIT INIT")

    lmp.command("run 0")

    # write reference file if none is given
    if reference is None: 
        lmp.command("write_data %s/%s.0.data" % (scrdir, job_name))
        reference = "%s/%s.0.data" % (scrdir, job_name)

    # initialise W-S code 
    ws = WignerSeitz (reference, my_rank=me, exporting_rank=0, mpi_comm=comm) 

    def fetch_id (xyz, ws):
        _dis,_id = ws.ktree0.query (np.array([xyz]), k=1)
        if _dis[0][0] < 0.1:
            return _id[0][0] # remember to increment by 1 for lammps convention 
        else:
            return None

    # fetch coordinate of most central atom
    centre = ws.atrans.r1 + 0.5*(ws.atrans.c1mat[:,0] + ws.atrans.c1mat[:,1] + ws.atrans.c1mat[:,2])
    cid = fetch_id (centre, ws)
    r0 = ws.data0.xyz[cid]

    # target number of interstitials to introduce
    IC15n = nint
    if IC15n%2 == 0:
        complete = True
    else:
        complete = False

    # approximate number of isocahedral centres to introduce
    nicocentres = int(np.round(IC15n/2 + .25))

    # on average, the bcc unit cell in a C15 structure contains 1 icosahedral centre 
    # the search radius spans a sphere in which we try to generate the C15 structure
    search_radius = np.power(3/(4*np.pi)*nicocentres, 1/3.)
    
    # add a buffer of a few unit cells 
    search_radius = int(1.3*search_radius + 5)

    # create diamond lattice for icosahedron centres
    diamond = np.array([[0,0,0],
                        [0,2,2],
                        [2,0,2],
                        [2,2,0],
                        [3,3,3],
                        [3,1,1],
                        [1,3,1],
                        [1,1,3]])

    signs = np.array([-1,-1,-1,-1,1,1,1,1])

    # place centre at [3,1,1] (motive A)
    diamond -= np.array ([3,1,1])
 
    # the diamond unit cell fits 2x2x2 bcc unit cells, so we 
    # rescale lattice positions to units of bcc unit cells
    diamond = 0.5 * diamond
   
    # create sublattice of icosahedron centres 
    ico_centre_lattice = []
    ico_centre_signs = [] 
    for i in range(-search_radius, search_radius+1): 
        for j in range(-search_radius, search_radius+1): 
            for k in range(-search_radius, search_radius+1):
                if i*i + j*j + k*k >= search_radius*search_radius:
                    continue
                ico_centre_lattice += [diamond + np.array([2*i,2*j,2*k])]
                ico_centre_signs += [signs]

    # concatenate lattice coordinates
    ico_centre_lattice = np.concatenate(ico_centre_lattice).astype(float)
    ico_centre_signs = np.concatenate(ico_centre_signs) 

    # filter out any icosahedra centres that fall outside the box
    frac = ws.atrans.cart_to_frac1_all (r0 + alattice*ico_centre_lattice)
    inboxbool = (frac[:,0]>=0)*(frac[:,0]<1) * (frac[:,1]>=0)*(frac[:,1]<1) * (frac[:,2]>=0)*(frac[:,2]<1)

    ico_centre_lattice = ico_centre_lattice[inboxbool]
    ico_centre_signs = ico_centre_signs[inboxbool]

    # sort ico centres by distance to origin of C15 structure
    ico_centre_distances = np.linalg.norm (ico_centre_lattice, axis=1)

    ordering = np.argsort (ico_centre_distances)
    ico_centre_lattice = ico_centre_lattice[ordering]
    ico_centre_signs = ico_centre_signs[ordering]
    ico_centre_distances = ico_centre_distances[ordering]

    # insert C15 patterns until all targeted interstitials have been placed
    allvacs = np.zeros((0,3))
    allints = np.zeros((0,3))
    nint = len(allints) - len(allvacs)

    current_index = 0
    while (nint < IC15n):

        nremainder = IC15n - nint

        _centre = ico_centre_lattice[current_index]
        _sign   = ico_centre_signs[current_index]

        # fetch vacancy and interstitial coordinates belonging to this ico centre
        vxyz,ixyz = full_ico (_sign)
        vxyz += _centre
        ixyz += _centre

        if nremainder > 1:
            # if more than 1 int has to be placed, try placing an entire icosahedron (i.e. up to 2 net ints)

            # merge new vac,int coordinates with all previously gathered ones, 
            allvacs = np.vstack ([allvacs, vxyz])
            allints = np.vstack ([allints, ixyz])

            # wrap back into periodic box
            allvacs = wrap_back_unique (ws, alattice*allvacs, r0)/alattice
            allints = wrap_back_unique (ws, alattice*allints, r0)/alattice

            # only keep unique coordinates
            allvacs = unique_xyz (allvacs)
            allints = unique_xyz (allints)


        if nremainder == 1:
            # otherwise, try adding one more vac + 2 of its close ints (i.e. up to 1 net int)
            for ix in [[0,0,1],[1,2,3],[2,4,5],[3,6,7],[4,8,9],[5,10,11]]:
                _vxyz = [vxyz[ix[0]]]
                _ixyz = [ixyz[ix[1]], ixyz[ix[2]]]

                # merge new vac,int coordinates with all previously gathered ones, 
                allvacs = np.vstack ([allvacs, _vxyz]) 
                allints = np.vstack ([allints, _ixyz])

                # wrap back into box
                allvacs = wrap_back_unique (ws, alattice*allvacs, r0)/alattice
                allints = wrap_back_unique (ws, alattice*allints, r0)/alattice

                # only keep unique coordinates
                allvacs = unique_xyz (allvacs)
                allints = unique_xyz (allints)

                # break if an additional interstitial was able to be placed
                if len(allints) - len(allvacs) > nint:
                    break


        # for the next iteration, consider the next closest ico centre
        current_index += 1
        
        # update net number of interstitials
        nint = len(allints) - len(allvacs)

        if current_index == len(ico_centre_lattice):
            mpiprint ("No more icosahedral lattice centres remaining! Cannot place more interstitials.")
            break 

        nremainder = IC15n - nint
        mpiprint ("Iteration %3d, remainder %3d: net, nint, nvac: %4d %4d %4d" % (current_index, nremainder, len(allints) - len(allvacs), len(allints) , len(allvacs)))


    mpiprint ("This C15 structure contains N_interstitials and N_vacancies:", len(allints), len(allvacs))
    mpiprint ("Extra atoms:", len(allints) - len(allvacs))

    if me == 0:
        with open("%s/%s.ico.%d.xyz" % (scrdir, job_name, IC15n), "w") as fopen:
            fopen.write ("%d\n" % (len(allints) + len(allvacs)))
            fopen.write ("# 0 are vacs, 1 are ints\n")
            for _int in allints:
                fopen.write ("1 %f %f %f\n" % tuple(_int))
            for _int in allvacs:
                fopen.write ("0 %f %f %f\n" % tuple(_int))


    # bring into lattice units
    allvacs = alattice * allvacs 
    allints = alattice * allints 

    comm.barrier ()

    # delete atoms for placing vacancies according to C15 prescription
    vacids = [fetch_id (r0 + _vxyz, ws) for _vxyz in allvacs]
    vacids = np.array([vacid for vacid in vacids if vacid is not None])

    vacstring = "%d "*len(vacids) % tuple(vacids+1)

    lmp.command ('group gvacs id %s' % vacstring)
    lmp.command ('delete_atoms group gvacs compress no')

    # place interstitials according to C15 prescription
    for _ixyz in allints:
        lmp.command ('create_atoms 1 single %f %f %f units box' % tuple(r0 + _ixyz)) 

    # delete overlapping atoms in case there is conflict due to PBC when inserting interstitials
    lmp.command('delete_atoms overlap 0.01 all all')   
 
    lmp.command("write_data %s/%s.unrelaxed.%d.data" % (scrdir, job_name, IC15n))

    # before minimisation: rescale box dimensions to lower energy
    scale = 1 + nint/(2*nx*ny*nz)/3
    lmp.command("change_box all x scale %f y scale %f z scale %f remap" % (scale, scale, scale))

    # also allow for box dimensions to be relaxed
    lmp.command("fix frelax all box/relax aniso 0.0 vmax 0.0001")

    lmp.command("min_modify dmax 0.001")
    lmp.command("minimize %s 0 100 100" % etolstring)

    lmp.command("min_modify dmax 0.01")
    lmp.command("minimize %s 0 100 100" % etolstring)

    lmp.command("min_modify dmax 0.01")
    lmp.command("minimize %s 0 1000 1000" % etolstring)
    
    lmp.command("min_modify dmax 0.1")
    lmp.command("minimize %s 0 10000 10000" % etolstring)

    lmp.command("write_data %s/%s.relaxed.%d.data" % (scrdir, job_name, IC15n))

    lmp.close()
    return 0


if __name__ == "__main__":
    main()

    if mode == 'MPI':
        MPI.Finalize()
