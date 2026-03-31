call conda activate cwq

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

cd /d %~dp0

set MPICC=mpicc

echo Environment initialized.

cmd /k