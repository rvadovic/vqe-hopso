
rem --- Load Intel oneAPI ---
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

rem --- Activate Conda Environment ---
call conda activate cwq

rem --- Set MPI compiler for MPI4Py / mpiCC ---
set MPICC=mpicc

rem --- Go to project directory ---
cd /d %~dp0

echo Environment initialized.

cmd /k