# Run all the cram tests in the current working directory, which presumably
# is the directory containing this file.

# Define NO_MODULE_LOAD to avoid purging previously loaded modules and
# loading all default versions of modules needed to run the cram tests.

if [[ ! -v NO_MODULE_LOAD ]]; then
    echo "Loading dependency modules."
    module use /pbi/dept/primary/modulefiles
    module use /mnt/software/modulefiles
    module purge
    module load cram/0.7
    module load primary-toolkit
    unset WORKSPACE  # if this is set, then the following module will use it instead of the cwd
    module load pacbio-pa-mongo/workspace
fi

echo "Using bazviewer: $(which bazviewer)."
echo "Using mongo-basecaller: $(which mongo-basecaller)."
cram -v *.t