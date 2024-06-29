# Segment Anything without Supervision

Unsupervised SAM (UnSAM) is a 'segment anything' model for promptable and automatic whole-image segmentation which does not require human annotations. 

<p align="center"> 
  <img width="1301" alt="teaser_unsam" src="https://github.com/frank-xwang/UnSAM/assets/58996472/0c53071c-bdc8-4424-9e9e-40b8c8c31a18" align="center" >
</p>


> [**Segment Anything without Supervision**](https://arxiv.org/abs/xxxx.xxxxx)            
> XuDong Wang, Jingfeng Yang, Trevor Darrell      
> UC Berkeley            
> Preprint            

[[`arxiv`](https://arxiv.org/abs/xxxx.xxxxx)] [[`colab (UnSAM)`](https://drive.google.com/file/d/1KyxbFb2JC76RZ1jg7F8Ee4TEmOlpYMe7/view?usp=sharing)] [[`colab (pseudo-label)`](https://drive.google.com/file/d/1aFObIt-xlQmCKk3G7dD8KQxaWhM_RTEd/view?usp=sharing)] [[`bibtex`](#citation)]             


## Updates
- 07/01/2024 Initial commit


## Features
- The performance gap between unsupervised segmentation models and SAM can be significantly reduced. UnSAM not only advances the state-of-the-art in unsupervised segmentation by 10% but also achieves comparable performance with the labor-intensive, fully-supervised SAM.
- The supervised SAM can also benefit from our self-supervised labels. By training UnSAM with only 1% of SA-1B images, a lightly semi-supervised UnSAM can often segment entities overlooked by supervised SAM, exceeding SAM’s AR by over 6.7% and AP by 3.9% on SA-1B. 


## Installation
See [installation instructions](INSTALL.md).

## Dataset Preparation
See [Preparing Datasets for UnSAM](datasets/README.md).

## Method Overview

UnSAM has two major stages: 1) generating pseudo-masks with divide-and-conquer and 2) learning unsupervised segmentation models from pseudo-masks of unlabeled data.

### 1. Multi-granular Pseudo-mask Generation with Divide-and-Conquer

Our Divide-and-Conquer approach can be used to provide multi-granular masks without human supervision.

### Divide-and-Conquer Demo

Try out the demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1aFObIt-xlQmCKk3G7dD8KQxaWhM_RTEd/view?usp=sharing)

If you want to run Divide-and-Conquer locally, we provide `demo_dico.py` that is able to visualize the pseudo-masks.
Please download the CutLER's checkpoint from [here](http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth), and then run it with:
```
cd divide_and_conquer
python demo_dico.py \
    --input /path/to/input/image \
    --output /path/to/save/output \
    --preprocess true \
    --postprocess true \ #postprocess requires gpu 
    --opts MODEL.WEIGHTS /path/to/cutler_checkpoint \
    MODEL.DEVICE gpu
```
We give a few demo images in docs/demos/. Following, we give some visualizations of the pseudo-masks on the demo images.
<p align="center">
  <img src="https://github.com/frank-xwang/UnSAM/assets/58996472/6ea40b0a-7fd3-436b-9b3f-37acbc122fc3" width=100%>
</p>


### 2. Segment Anything without Supervision

### Inference Demo for UnSAM with Pre-trained Models (whole image segmentation)
Try out the UnSAM demo using Colab (no GPU needed): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1KyxbFb2JC76RZ1jg7F8Ee4TEmOlpYMe7/view?usp=sharing)

If you want to run UnSAM or UnSAM+ demos locally, we provide `demo_whole_image.py` that is able to demo builtin configs. 
Please download UnSAM/UnSAM+'s checkpoints from the [model zoo](#model-zoo). 
Run it with:
```
cd whole_image_segmentation
python demo_whole_image.py \
    --input /path/to/input/image \
    --output /path/to/save/output \
    --opts \
    MODEL.WEIGHTS /path/to/UnSAM_checkpoint \
    MODEL.DEVICE cpu
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and save the results in the local path.
<!-- For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are: -->
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.

Following, we give some visualizations of the model predictions on the demo images.
<p align="center">
  <img src="https://github.com/frank-xwang/UnSAM/assets/58996472/83f9d9ee-0c2e-4b65-83f7-77852d169d2d" width=100%>
</p>


### Gradio Demo for UnSAM with Pre-trained Models (promptable image segmentation)

The following command will pops up a gradio website link in the terminal, on which users can interact with our model. 
Please download UnSAM/UnSAM+'s checkpoints from the [model zoo](#model-zoo). 
For details of the command line arguments, see `demo_promptable.py -h` or look at its source code
to understand its behavior.
* To run __on cpu__, add `cpu` after `--device`.
```
python demo_promptable.py \
    --ckpt /path/to/UnSAM_checkpoint \
    --conf_files configs/semantic_sam_only_sa-1b_swinT.yaml \
    --device gpu
```

Following, we give some visualizations of the model predictions on the demo images.
<p align="center">
  <img src="https://github.com/frank-xwang/UnSAM/assets/58996472/1b7eb492-2c3d-426f-9f90-bc117ea322eb" width=100%>
</p>


### Model Evaluation
To evaluate a model's performance on 7 different datasets, please refer to [datasets/README.md](datasets/README.md) for 
instructions on preparing the datasets. Next, select a model from the model zoo, specify the "model_weights", "config_file" 
and the path to "DETECTRON2_DATASETS" in `tools/eval.sh`, then run the script.
```
bash tools/{promptable, whole_image}_eval.sh
```

### Model Zoo

#### Whole image segmentation
UnSAM achieves the state-of-the-art results on unsupervised image segmentation, using a backbone of ResNet50 and training 
with only 1% of SA-1B data. We show zero-shot unsupervised image segmentation performance on 6 different datasets, 
including COCO, LVIS, ADE20K, Entity, SA-1B, Part-ImageNet and PACO.   
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Methods</th>
<th valign="bottom">Models</th>
<th valign="bottom">Backbone</th>
<th valign="bottom"># of Train Images</th>
<th valign="bottom">Avg.</th>
<th valign="bottom">COCO</th>
<th valign="bottom">LVIS</th>
<th valign="bottom">ADE20K</th>
<th valign="bottom">Entity</th>
<th valign="bottom">SA-1B</th>
<th valign="bottom">PtIn</th>
<th valign="bottom">PACO</th>
<!-- TABLE BODY -->
</tr>
<tr><td align="center">Prev. Unsup. SOTA</td>
<td valign="center">-</td>
<td valign="center">ViT-Base</th>
<td align="center">0.2M</td>
<td align="center">30.1</td>
<td align="center">30.5</td>
<td align="center">29.1</td>
<td align="center">31.1</td>
<td align="center">33.5</td>
<td align="center">33.3</td>
<td align="center">36.0</td>
<td align="center">17.1</td>
</tr>
<tr><td align="center">UnSAM (ours)</td>
<td valign="center">-</td>
<td valign="center">ResNet50</th>
<td align="center">0.1M</td>
<td align="center">39.2</td>
<td align="center">40.5</td>
<td align="center">37.7</td>
<td align="center">35.7</td>
<td align="center">39.6</td>
<td align="center">41.9</td>
<td align="center">51.6</td>
<td align="center">27.5</td>
</tr>
<tr><td align="center">UnSAM (ours)</td>
<td valign="center"><a href="https://drive.google.com/file/d/1qUdZ2ELU_5SNTsmx3Q0wSA87u4SebiO4/view?usp=drive_link">download</a></td>
<td valign="center">ResNet50</th>
<td align="center">0.4M</td>
<td align="center">41.1</td>
<td align="center">42.0</td>
<td align="center">40.5</td>
<td align="center">37.5</td>
<td align="center">41.0</td>
<td align="center">44.5</td>
<td align="center">52.7</td>
<td align="center">29.7</td>
</tr>
</tbody></table>

UnSAM+ can outperform SAM on most experimented benchmarks (including SA-1B), when training UnSAM on 1% of SA-1B with both 
ground truth masks and our unsupervised labels. This demonstrates that the supervised SAM can also benefit from our self-supervised labels.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Methods</th>
<th valign="bottom">Models</th>
<th valign="bottom">Backbone</th>
<th valign="bottom"># of Train Images</th>
<th valign="bottom">Avg.</th>
<th valign="bottom">COCO</th>
<th valign="bottom">LVIS</th>
<th valign="bottom">ADE20K</th>
<th valign="bottom">Entity</th>
<th valign="bottom">SA-1B</th>
<th valign="bottom">PtIn</th>
<th valign="bottom">PACO</th>
<!-- TABLE BODY -->
</tr>
<tr><td align="center">SAM</td>
<td valign="center">-</td>
<td valign="center">ViT-Base</td>
<td align="center">11M</td>
<td align="center">42.1</td>
<td align="center">49.6</td>
<td align="center">46.1</td>
<td align="center">45.8</td>
<td align="center">45.9</td>
<td align="center">60.8</td>
<td align="center">28.3</td>
<td align="center">18.1</td>
</tr>
<tr><td align="center">UnSAM+ (ours)</td>
<td valign="center"><a href="https://drive.google.com/file/d/1sCZM5j2pQr34-scSEkgG7VmUaHJc8n4d/view?usp=drive_link">download</a></td>
<td valign="center">ResNet50</td>
<td align="center">0.1M</td>
<td align="center">48.8</td>
<td align="center">52.2</td>
<td align="center">50.8</td>
<td align="center">45.3</td>
<td align="center">49.8</td>
<td align="center">64.8</td>
<td align="center">46.0</td>
<td align="center">32.3</td>
</tr>
</tbody></table>

#### Promptable image segmentation
Despite using a backbone that is 3× smaller and being trained on only 1% of SA-1B, our lightly semi-supervised UnSAM+ surpasses the fully-supervised SAM in promptable segmentation task on COCO.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Methods</th>
<th valign="bottom">Models</th>
<th valign="bottom">Backbone</th>
<th valign="bottom"># of Train Images</th>
<th valign="bottom">Point (Max)</th>
<th valign="bottom">Point (Oracle)</th>
<!-- TABLE BODY -->
</tr>
<tr><td align="center">SAM</td>
<td valign="center">-</td>
<td align="center">ViT-B/8 (85M)</td>
<td align="center">11M</td>
<td align="center">52.1</td>
<td align="center">68.2</td>
</tr>
<tr><td align="center">UnSAM (ours)</td>
<td valign="center"><a href="https://drive.google.com/file/d/10iPpraRoWE58mHPiv8Q1alsBekAQOxM9/view?usp=drive_link">download</a></td>
<td align="center">Swin-Tiny (25M)</td>
<td align="center">0.1M</td>
<td align="center">40.3</td>
<td align="center">59.5</td>
</tr>
<tr><td align="center">UnSAM+ (ours)</td>
<td valign="center"><a href="https://drive.google.com/file/d/12Z2pOASXEEMGz5-Svn1Fe7MNX41JkhHD/view?usp=sharing">download</a></td>
<td align="center">Swin-Tiny (25M)</td>
<td align="center">0.1M</td>
<td align="center">52.4</td>
<td align="center">69.5</td>
</tr>
</tbody></table>

## License
The majority of UnSAM, CutLER, Detectron2 and DINO are licensed under the [CC-BY-NC license](LICENSE), however portions of the project are available under separate license terms: Mask2Former, Semantic-SAM, CascadePSP, Bilateral Solver and CRF are licensed under the MIT license; If you later add other third party code, please keep this license info updated, and please let us know if that component is licensed under something other than CC-BY-NC, MIT, or CC0.

## Acknowledgement
This codebase is based on CutLER, SAM, Mask2Former, Semantic-SAM, CascadePSP, BFS, CRF, DINO and Detectron2. We appreciate the authors for open-sourcing their codes. 

## Ethical Considerations
UnSAM's wide range of detection capabilities may introduce similar challenges to many other visual recognition methods.
As the image can contain arbitrary instances, it may impact the model output.

## How to get support from us?
If you have any general questions, feel free to email us at [XuDong Wang](mailto:xdwang@eecs.berkeley.edu). If you have code or implementation-related questions, please feel free to send emails to us or open an issue in this codebase (We recommend that you open an issue in this codebase, because your questions may help others). 

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@inproceedings{wang2023cut,
  title={Segment Anything without Supervision},
  author={Wang, Xudong and Yang, Jingfeng and Darrell, Trevor},
  year={2024}
}
```

