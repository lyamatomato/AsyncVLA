# AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://asyncvla.github.io)


[Noriaki Hirose](https://sites.google.com/view/noriaki-hirose/)<sup>1, 2</sup>, [Catherine Glossop](https://catglossop.github.io/)<sup>1</sup>, [Dhruv Shah](https://robodhruv.github.io/)<sup>3</sup>, [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)<sup>1</sup>

<sup>1</sup> UC Berkeley (_Berkeley AI Research_),  <sup>2</sup> Toyota Motor North America, ,  <sup>3</sup> Princeton University

### Installation
Please set up a conda environment (see instructions in [SETUP.md](SETUP.md)).

### Inference
1. Download our checkpoints and place them in our directory. 
    ```
    git clone https://huggingface.co/NHirose/AsyncVLA_release
    ```
2. Run AsyncVLA using a sample current image, GPS pose, and language prompt. You can view the generated trajectory in the output figure 1_ex.jpg. (Run BaseVLA and Edge adapter in same PC)
    ```
    python inference/run_asyncvla.py
    ```
3. Change the goal modality: by default, our code generates actions based on the language prompt. To use a different modality, you can modify the settings around line 560. 
    
4. Run AsyncVLA to control the real robot. We split the AsyncVLA into the base VLA and the edge adapter. Then we run the base VLA in the remote workstation and run the edge adapter in the robot edge controller with ROS1. Details are shown in the paper appendix. 

### Training datasets
We provide training code that supports multiple public datasets. Before following the full training process, please first ensure that you can run the example training with the sample dataloader.

1. Downloading all datasets from the original website. ([GNM](https://github.com/robodhruv/visualnav-transformer), [LeLaN](https://github.com/NHirose/learning-language-navigation), [SACSoN(HuRoN)](https://sites.google.com/view/sacson-review/home)) Please verify that the downloaded datasets work properly in their original codebase.

2. Downloading the lerobot code base for the Frodobots dataset dataloader:
    ```
    git clone https://github.com/huggingface/lerobot.git 
    ```
3. Edit the data path in config_nav/dataset_config.yaml:

5. Training our policy from OpenVLA checkpoints (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_omnivla_dataset.py  --vla_path ./omnivla-original --dataset_name omnivla --wandb_entity "X"   --wandb_project "omnivla"
    ```
       
In our training setup, we use 5 Nvidia H200 GPUs (140 GB each) across 5 nodes. The batch sizes are configured as [LeLaN, GNM, SACSoN] = [6, 6, 6], with gradient accumulation set to 2 steps. 

### Training
We provide the training code along with a sample dataloader to help you quickly understand the required data loading structure. Since preparing the full training dataset is resource-intensive, we include this simplified code base for convenience.

1. Downloading MBRA project code base:
    ```
    git clone https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA.git
    ```
2. You can set the training mode at line 10 and 11 in vla-scripts/train_asyncvla.py.

3. You can configure visualization at line 12 in vla-scripts/train_asyncvla.py. During training, it should be set to False.
    
4. Training our policy from AsyncVLA checkpoints (Please fill X):
    ```
    torchrun --standalone --nnodes 1 --nproc-per-node X vla-scripts/train_asyncvla.py  --vla_path ./AsyncVLA_release --dataset_name omnivla --wandb_entity "X"   --wandb_project "asyncvla" --grad_accumulation_steps X
    ```
    
### Acknowledgement
We implement our ideas and design choices on top of the pretrained checkpoints. Our work builds upon the [OpenVLA-OFT](https://openvla-oft.github.io/), [OmniVLA](https://github.com/NHirose/OmniVLA) and [ViNT](https://github.com/robodhruv/visualnav-transformer) codebases, with additional code added to create AsyncVLA. As such, our implementation leverages many components of these codebases. We sincerely appreciate the effort and contributions of the OpenVLA-OFT, OmniVLA and ViNT team!

## Citing
```
@misc{hirose2026asyncvla,
      title={AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge}, 
      author={Noriaki Hirose and Catherine Glossop and Dhruv Shah and Sergey Levine},
      year={2026},
      eprint={xxxx.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/xxxx.xxxxx}, 
}
