# This creates a Python3 virtual environment for use in testing.
# This script is meant to be "sourced" with the "." command from the parent bash script.
# It makes the assumption that the virtual environment either does not exist yet,
# or it has been properly installed. It then installs or updates the
# python modules it needs. To completely install the virtual environment
# from scratch, please do `rm -rf python_ve` before sourcing this script.
module purge
module use /mnt/software/modulefiles
module unload python
module load   python/3.9.6
if [ ! -e python_ve ]; then
  python3 -m venv python_ve
elif python_ve/bin/pip3 --help > /dev/null; then
  echo -n
  # pass
else
    rm -rfv python_ve
  python3 -m venv python_ve
fi

source python_ve/bin/activate
pip3 -q install --upgrade pip
pip3 -q install h5py junit_xml lxml requests simplejson pyzmq sysv_ipc
