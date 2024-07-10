CALL_DIR=$PWD
# Import several helper functions in bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
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
    cd ..
fi

# activate virtualenv; 
source ${SPOT_PATH}/venv/spot/bin/activate

## Install the dependencies
pip uninstall -y em
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

## Install instant-NGP
# cuda 11.8 is already supported within the container

# Install cmake 3.21.0
# get and build CMake
wget https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0.tar.gz
tar -zvxf cmake-3.21.0.tar.gz
cd cmake-3.21.0
./bootstrap
make -j8

sudo apt-get install checkinstall
# this will take some time
sudo checkinstall --pkgname=cmake --pkgversion="3.20-custom" --default
# reset shell cache for tools paths
hash -r
rm cmake-3.21.0.tar.gz
rm -rf cmake-3.21.0

# Other python packages for instant-NGP
pip install commentjson
pip install imageio
pip install opencv-python-headless
pip install pybind11
pip install pyquaternion
pip install scipy
pip install tqdm

# Install instant-NGP
git clone --recursive https://github.com/nvlabs/instant-ngp
cd instant-ngp
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
cd ..

## Install GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
# Download the pretrained weights
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../../

## Install LoFTR
git clone https://github.com/zju3dv/LoFTR.git
pip install einops yacs kornia

## Dependencies for SPOT
pip install PySide2
pip install bosdyn-client
pip install bosdyn-mission
pip install bosdyn-api
pip install bosdyn-core
pip install bosdyn-choreography-client






    



