# C15builder
Python script for generating coherent C15 phases of arbitrary size in a base-centered cubic crystal with LAMMPS

The construction roughly follows the prescription outlined in [M.-C. Marinica, F. Willaime, and J.-P. Crocombette, Phys. Rev. Lett. 108, 025501](https://doi.org/10.1103/PhysRevLett.108.025501).

## Getting started

The script requires LAMMPS to be built with the MANYBODY package to enable support for embedded atom model (EAM) potentials. LAMMPS must be compiled as a shared library linked to Python 3+ with the numpy, scipy, and mpi4py packages. See the [LAMMPS documentation](https://docs.lammps.org/Python_head.html) for more information on how to set this up.

The `json/*.json` files contains simulation settings and paths to important files and directories, such as the EAM potential file and the data directory. Update the paths as appropriate for your system. The `json` file acts as the input file for the simulation, enabling the running of multiple similar simulations without having to manually edit the Python script.

## Running the code

To build an example C15 cluster containing 10 interstitials in iron from the terminal, first ensure that an iron EAM potential is available, for example `M07_eam.fs` by [Malerba et al. (2010)](https://doi.org/10.1016/j.jnucmat.2010.05.017) available at the [NIST Interatomic Potentials repository](https://www.ctcms.nist.gov/potentials/), and then run the script as follows:

```
mpirun -n 8 python3 icosa.py json/test.json 10
```

As the simulation runs, the structure is initialised and relaxed. The following output files are written:
- `data/test.ico.10.xyz`: xyz file of interstitial and vacancy coordinates required to create the C15 phase inside the BCC crystal
- `data/test.unrelaxed.10.data`: LAMMPS data file of the C15 phase before structural relaxation
- `data/test.relaxed.10.data`: LAMMPS data file of the C15 phase after structural relaxation

In practice, the unrelaxed structure `test.unrelaxed.10.data` is constructed by adding and deleting atoms following the prescription of `test.ico.10.xyz`. This repository already comes with the `test.ico.*.xyz` files for sizes in the range between 2 and 100 interstitials. Note that these structures do not represent the global energy minimum of a C15 phase in BCC; for this purpose, some heuristic global minimisation scheme must be used, for example as outlined by [Alexander et al. (2016)](https://doi.org/10.1103/PhysRevB.94.024103). In principle, this script could be modified for this purpose.
