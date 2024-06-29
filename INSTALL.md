## Installation

### Example conda environment setup
```bash
conda create --name UnSAM python=3.8 -y
conda activate UnSAM
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

python -m pip install 'git+https://github.com/MaureenZOU/detectron2-xyz.git'
pip install git+https://github.com/cocodataset/panopticapi.git
python -m pip install 'git+https://github.com/UX-Decoder/Semantic-SAM.git'

git clone git@github.com:frank-xwang/UnSAM.git
cd promptable_segmentation/model/body/encoder/ops
sh make.sh
cd whole_image_segmentation/mask2former/modeling/pixel_decoder/ops
sh make.sh

python -m pip install -r requirements.txt
```