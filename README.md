EVspectrum
==========

Program for simulating electron vibronic coupling in molecules.

Simple Instructions for Installation:

Install PETSc, which can be found here

Install SLEPc, which can be found here

Define the two environment variables 
PETSC_DIR
SLEPC_DIR

In the main source file directory, run make

To execute a serial job, run ./EVspectrum

To execute a parallel job, run ${PETSc}/bin/petscmpiexec -n ${TOTCPU} ./EVspectrum0.2.1
where ${TOTCPU} is the number of MPI processes you would like to use. 
