# This creates a Python3 virtual environment for use by pa_end_to_end.py 
# This script is meant to be "sourced" with the "." command from the parent bash script.
# It makes the assumption that the virtual environment either does not exist yet,
# or it has been properly installed. It then installs or updates the
# python modules it needs. To completely install the virtual environment
# from scratch, please do `rm -rf e2e_ve` before sourcing this script.

. /usr/share/Modules/init/sh
module use /mnt/software/modulefiles
module unload python
module load   python/3.7.3
if [ ! -e e2e_ve ]
then
  python3 -m venv e2e_ve
fi
source e2e_ve/bin/activate
pip3 install --upgrade pip
pip3 install --isolated h5py junit_xml lxml requests simplejson pyyaml
