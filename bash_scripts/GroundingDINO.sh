## Prerequisite
if [ -z "${CUDA_HOME}" ]; then
    echo "CUDA_HOME is unset or set to the empty string"
    echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
    source ~/.bashrc
    echo $CUDA_HOME
fi


## Install GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
# Download the pretrained weights
source ../venv/spot/bin/activate

mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../

## Install Segment-Anything-Mask
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
mkdir sam_weights
cd sam_weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../../