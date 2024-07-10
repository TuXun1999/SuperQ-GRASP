# Install cmake 3.21.0
sudo apt-get install -y libssl-dev

# get and build CMake
wget https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0.tar.gz
tar -zvxf cmake-3.21.0.tar.gz
cd cmake-3.21.0
./bootstrap
make -j8

sudo apt-get install checkinstall
# this will take some time
sudo checkinstall --pkgname=cmake --pkgversion="3.21-custom" --default
# reset shell cache for tools paths
hash -r
cd ..
rm cmake-3.21.0.tar.gz
sudo rm -rf cmake-3.21.0