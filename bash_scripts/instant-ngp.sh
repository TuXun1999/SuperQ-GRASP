sudo apt-get install build-essential git python3-dev python3-pip libopenexr-dev libxi-dev \
                     libglfw3-dev libglew-dev libomp-dev libxinerama-dev libxcursor-dev
## Install instant-NGP
# cuda 12.1 is already supported within the container

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