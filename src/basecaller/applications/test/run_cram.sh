# Run all the cram tests in the current working directory, which presumably
# is the directory containing this file.

# Define NO_MODULE_LOAD to avoid purging previously loaded modules and
# loading all default versions of modules needed to run the cram tests.

if [[ ! -v NO_MODULE_LOAD ]]; then
    echo "Loading dependency modules."
    module use /pbi/dept/primary/modulefiles
    module use /mnt/software/modulefiles
    module purge
    # temporarily required for running mongo, until some tbb issues get resolved
    module load composer_xe/2017.4.196
    module load cram/0.7
    module load primary-toolkit/1.0.7
    unset WORKSPACE  # if this is set, then the following module will use it instead of the cwd
    module load pacbio-pa-mongo/workspace
fi

echo "Using bazviewer: $(which bazviewer)."
echo "Using smrt-basecaller: $(which smrt-basecaller)."

pushd $(dirname "$0")
cram -v *.t
popd
