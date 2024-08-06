import numpy as np

class Import_eamfs:
    def __init__(self, path):
        print ("Importing eam/fs file at %s." % path)
        self.load(path)

    def load(self, path):
        with open(path, 'r') as fopen:
            eamfile = [line.split() for line in fopen]
            
        # print header
        for l in eamfile[:3]:
            print (" ".join(l))
        print ()

        elements = eamfile[3][1:]
        nelements = len(elements)
        Nrho, drho, Nr, dr, cutoff = np.array(eamfile[4], dtype=float)
        Nrho, Nr = int(Nrho), int(Nr)

        #print (Nrho, drho, Nr, dr, cutoff)
        _rawdata = [item for sublist in eamfile[5:] for item in sublist]

        self.znum = {}
        self.mass = {}
        self.alat = {}
        self.atyp = {}

        F_i_raw = {}
        rho_ij_raw = {}
        phi_ij_raw = {}

        c = 0
        for iele in range(nelements):
            # read in element info
            self.znum[iele] = int(_rawdata[c])
            self.mass[iele] = float(_rawdata[c+1])
            self.alat[iele] = float(_rawdata[c+2])
            self.atyp[iele] = _rawdata[c+3].lower()
            c+=4

            # read in embedding function
            F_i_raw[iele] = np.array(_rawdata[c:c+Nrho], dtype=float)
            c+=Nrho

            # read in density functions
            #for jele in range(nelements):
            #    rho_ij_raw[(iele, jele)] = np.array(_rawdata[c:c+Nrho], dtype=float)
            #    c += Nrho
            rho_ij_raw[(iele, iele)] = np.array(_rawdata[c:c+Nr], dtype=float)
            c += Nr

        rho_range = np.linspace(0,drho*(Nrho-1), Nrho)
        phi_range = np.linspace(0,dr*(Nr-1), Nr)

        phi_range_fix = phi_range[:]
        phi_range_fix[0] += 1.e-9 # avoid nan

        # read in pair potentials and rescale
        for iele in range(nelements):
            for jele in range(nelements):
                if iele>=jele:
                    phi_ij_raw[(iele, jele)] = np.array(_rawdata[c:c+Nr], dtype=float)/phi_range_fix
                    c += Nr

        # we do not need these unless we want to do some energy/force calculations outside of lammps 
        '''
        print ("Create interpolation functions...")

        # create interpolation functions
        self.F_i = {}
        for _key in F_i_raw:
            self.F_i[_key] = PyCubicSpline(rho_range, F_i_raw[_key])
        print ("F_i done.")

        self.rho_ij = {}
        for _key in rho_ij_raw:
            self.rho_ij[_key] = PyCubicSpline(phi_range, rho_ij_raw[_key])
        print ("rho_ij done.")

        self.phi_ij = {}
        for _key in phi_ij_raw:
            self.phi_ij[_key] = PyCubicSpline(phi_range, phi_ij_raw[_key])
        print ("phi_ij done.")
        '''

        self.elements = elements 
        self.nelements = nelements 
        self.cutoff = cutoff

        return 0

