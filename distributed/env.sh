# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
module load languages/gcc/10.4.0
module load languages/python/3.9.15
module load languages/intel/2020-u4

export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=28
export OMP_WAIT_POLICY=active
export OMP_DYANMIC=false
