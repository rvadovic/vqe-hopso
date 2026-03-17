
ENV_NAME="cwq"
rem --- Activate Conda Environment ---
call conda activate $ENV_NAME

rem --- Set MPI compiler for MPI4Py / mpiCC ---
set MPICC=mpicc

rem --- Go to project directory ---
cd /d %~dp0

echo Environment initialized.

cmd /k