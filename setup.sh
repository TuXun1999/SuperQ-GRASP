CALL_DIR=$PWD
# Import several helper functions in bash
SCRIPT_DIR=$PWD
. "${SCRIPT_DIR}/tools.sh"

# Path to Spot workspace, relative to repository root;
# No begin or trailing slash.
SPOT_PATH=${SCRIPT_DIR}

# Add a few alias for pinging spot.
#------------- Main Logic  ----------------

# We have only tested Spot stack with Ubuntu 20.04.
if ! ubuntu_version_equal 20.04; then
    echo "Current SPOT development requires Ubuntu 20.04. Abort."
    return 1
fi


# create a dedicated virtualenv for all programs
if [ ! -d "${SPOT_PATH}/venv/spot" ]; then
    cd ${SPOT_PATH}/
    virtualenv -p python3 venv/spot
fi

# activate virtualenv; 
source ${SPOT_PATH}/venv/spot/bin/activate

## Install the dependencies
pip install empy
# pip install pyqt5
pip3 install pyqt5==5.12.2
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install numpy
pip install open3d
# scikit-image
pip install -U scikit-image
pip install pydot
pip install graphviz


## Dependencies for SPOT
pip install PySide2
pip install bosdyn-client
pip install bosdyn-mission
pip install bosdyn-api
pip install bosdyn-core
pip install bosdyn-choreography-client






    



