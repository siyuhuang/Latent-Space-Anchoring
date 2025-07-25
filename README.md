# Latent-Space-Anchoring

PyTorch implementation of ***Domain-Scalable Unpaired Image Translation via Latent Space Anchoring***

[Siyu Huang<sup>*</sup>](https://siyuhuang.github.io) (Harvard), [Jie An<sup>*</sup>](https://www.cs.rochester.edu/u/jan6/) (Rochester), [Donglai Wei](https://donglaiw.github.io/) (BC), [Zudi Lin](https://zudi-lin.github.io/) (Amazon Alexa), [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/) (Rochester), [Hanspeter Pfister](https://vcg.seas.harvard.edu/people/hanspeter-pfister) (Harvard)  
*IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*

[[Paper]](https://ieeexplore.ieee.org/document/10158033)

<p align="center">
<img src="docs/intro.png" width="550px"/>
</p>

Given an **unpaired image-to-image translation (UNIT)** model trained on certain domains, it is challenging to incorporate new domains. This work includes a domain-scalable UNIT method, termed as *latent space anchoring*, anchors images of different domains to the same latent space of frozen GANs by learning lightweight encoder and regressor models to reconstruct single-domain images. In inference, the learned encoders and decoders of different domains can be arbitrarily combined to translate images between any two domains without fine-tuning:

<p align="center">
<img src="docs/4domain.jpg"/>
</p>

## Installation
We recommend installing using [Anaconda](https://docs.anaconda.com/anaconda/install/). All dependencies are provided in `env.yaml`.
```
conda env create -f env.yaml
conda activate lsa
```

## Pretrained Models
Please download the pre-trained models from the following links. 
| Name | Enc/Dec Domain | Generator Backbone 
| :--- | :---------- | :----------
|[seg2ffhq.pt](https://drive.google.com/file/d/19DtecBicfg6vBY5ovhcn_pmDpOpLbF4J/view?usp=drive_link)  | facial segmentation mask ([CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)) | StyleGAN2 trained on [FFHQ](https://github.com/NVlabs/ffhq-dataset) face.
|[sketch2ffhq.pt](https://drive.google.com/file/d/1Svbsk1a2nyiTKzi9RrB8bYOVC9kyvVFv/view?usp=drive_link)  | facial sketch ([CUFSF](http://mmlab.ie.cuhk.edu.hk/archive/cufsf/)) | StyleGAN2 trained on [FFHQ](https://github.com/NVlabs/ffhq-dataset) face.
|[cat2dog.pt](https://drive.google.com/file/d/1Vs9zgskcnNkmQoVF8iemLLf4dYwllynd/view?usp=drive_link)  | cat face ([AFHQ-cat](https://github.com/clovaai/stargan-v2/blob/master/README.md)) | StyleGAN2 trained on [AFHQ-dog](https://github.com/clovaai/stargan-v2/blob/master/README.md).

In addition, we provide the auxiliary pre-trained models used for training our models.
| Name | Description
| :--- | :----------
|[stylegan2-ffhq-config-f.pt](https://drive.google.com/file/d/1NghBNgK6x5E9d3yyq_VFns3Lsl9JSR1K/view?usp=drive_link) | StyleGAN2 generator on [FFHQ](https://github.com/NVlabs/ffhq-dataset) face.
|[psp_ffhq_encode.pt](https://drive.google.com/file/d/1jsaz6fgr_22jjSjaPmddkodbdohnZu0y/view?usp=drive_link)  | The encoder for [StyleGAN2-FFHQ](https://github.com/rosinality/stylegan2-pytorch) inversion.
|[model_ir_se50.pth](https://drive.google.com/file/d/1ZNKfkNg5LUK4aIjck-2Cg90XSW-7Vn3Z/view?usp=drive_link) | [IR-SE50](https://github.com/TreB1eN/InsightFace_Pytorch) model used for encoder's weight initialization.

## Testing
### CelebAMask-to-FFHQ
<p align="left">
<img src="docs/seg2ffhq.jpg" width="400px"/>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;regressor output&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generator output
</p>

Unpaired image translation from CelebAMask-HQ mask to FFHQ image.

Please download [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ), put it in `./data/`. Download pre-trained model [seg2ffhq.pt](https://drive.google.com/file/d/19DtecBicfg6vBY5ovhcn_pmDpOpLbF4J/view?usp=drive_link), put it in `./pretrained_models`. The folder structure is
```
Latent-Space-Anchoring
├── data
│   ├── CelebAMask-HQ
│   │   ├── face_parsing
│   │   │   ├── Data_preprocessing
│   │   │   ├── ├── train_img
│   │   │   ├── ├── train_label
│   │   │   ├── ├── test_img
│   │   │   ├── ├── test_label
├── pretrained_models
│   ├── seg2ffhq.pt
├── commands
│   ├── test_seg2ffhq.sh
```

Run:
```
bash commands/test_seg2ffhq.sh
```

### Sketch-to-FFHQ
<p align="left">
<img src="docs/sketch2ffhq.jpg" width="400px"/>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;regressor output&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generator output
</p>

Unpaired image translation from CUFSF facial sketch to FFHQ image. Figures: input, regressor output, generator output.

Please download [CUFSF dataset](http://mmlab.ie.cuhk.edu.hk/archive/cufsf/), put it in `./data/`. Manually split the dataset into training and test sets (we use the first 1k images as training set and the rest as test set). Download pre-trained model [sketch2ffhq.pt](https://drive.google.com/file/d/1Svbsk1a2nyiTKzi9RrB8bYOVC9kyvVFv/view?usp=drive_link), put it in `./pretrained_models`. The folder structure is
```
Latent-Space-Anchoring
├── data
│   ├── CUFSF
│   │   ├── train
│   │   ├── test
├── pretrained_models
│   ├── sketch2ffhq.pt
├── commands
│   ├── test_sketch2ffhq.sh
```

Run:
```
bash commands/test_sketch2ffhq.sh
```

### FFHQ-to-CelebAMask
<p align="left">
<img src="docs/ffhq2seg.jpg" width="400px"/>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;regressor output&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generator output
</p>

Unpaired image translation from FFHQ image to CelebAMask-HQ mask. Figures: input, regressor output, generator output.

Please download [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset), put it in `./data/`. Download pre-trained models [seg2ffhq.pt](https://drive.google.com/file/d/19DtecBicfg6vBY5ovhcn_pmDpOpLbF4J/view?usp=drive_link) and [psp_ffhq_encode.pt](https://drive.google.com/file/d/1WmG-Sfcubv3QNwdMsOwW3QUeLiW0nCrf/view?usp=sharing), put them in `./pretrained_models`. The folder structure is
```
Latent-Space-Anchoring
├── data
│   ├── ffhq
│   │   ├── images1024x1024
├── pretrained_models
│   ├── seg2ffhq.pt
│   ├── psp_ffhq_encode.pt
├── commands
│   ├── test_ffhq2seg.sh
```

Run:
```
bash commands/test_seg2ffhq.sh
```

### FFHQ-to-Sketch
<p align="left">
<img src="docs/ffhq2sketch.jpg" width="400px"/>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;regressor output&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generator output
</p>

Unpaired image translation from FFHQ image to CUFSF facial sketch. Figures: input, regressor output, generator output.

Please download [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset), put it in `./data/`. Download pre-trained models [sketch2ffhq.pt](https://drive.google.com/file/d/1Svbsk1a2nyiTKzi9RrB8bYOVC9kyvVFv/view?usp=drive_link) and [psp_ffhq_encode.pt](https://drive.google.com/file/d/1WmG-Sfcubv3QNwdMsOwW3QUeLiW0nCrf/view?usp=sharing), put them in `./pretrained_models`. The folder structure is
```
Latent-Space-Anchoring
├── data
│   ├── ffhq
│   │   ├── images1024x1024
├── pretrained_models
│   ├── sketch2ffhq.pt
│   ├── psp_ffhq_encode.pt
├── commands
│   ├── test_ffhq2sketch.sh
```

Run:
```
bash commands/test_ffhq2sketch.sh
```

### CelebAMask-to-Sketch
<p align="left">
<img src="docs/seg2sketch.jpg" width="400px"/>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;regressor output&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generator output
</p>

Unpaired image translation from CelebAMask-HQ mask to CUFSF facial sketch.

Please download [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ), put it in `./data/`. Download pre-trained model [seg2ffhq.pt](https://drive.google.com/file/d/19DtecBicfg6vBY5ovhcn_pmDpOpLbF4J/view?usp=drive_link) and [sketch2ffhq.pt](https://drive.google.com/file/d/1Svbsk1a2nyiTKzi9RrB8bYOVC9kyvVFv/view?usp=drive_link), put them in `./pretrained_models`. The folder structure is
```
Latent-Space-Anchoring
├── data
│   ├── CelebAMask-HQ
│   │   ├── face_parsing
│   │   │   ├── Data_preprocessing
│   │   │   ├── ├── train_img
│   │   │   ├── ├── train_label
│   │   │   ├── ├── test_img
│   │   │   ├── ├── test_label
├── pretrained_models
│   ├── seg2ffhq.pt
│   ├── sketch2ffhq.pt
├── commands
│   ├── test_seg2sketch.sh
```

Run:
```
bash commands/test_seg2sketch.sh
```

### Cat-to-Dog
<p align="left">
<img src="docs/cat2dog.jpg" width="400px"/>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;input&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;regressor output&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;generator output
</p>

Unpaired image translation from AFHQ-cat to AFHQ-dog.

Please download [AFHQ dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md), put it in `./data/`. Download pre-trained model [cat2dog.pt](https://drive.google.com/file/d/1Vs9zgskcnNkmQoVF8iemLLf4dYwllynd/view?usp=drive_link), put it in `./pretrained_models`. The folder structure is
```
Latent-Space-Anchoring
├── data
│   ├── AFHQ
│   │   ├── afhq
│   │   │   ├── train
│   │   │   ├── ├── cat
│   │   │   ├── ├── dog
│   │   │   ├── ├── wild
│   │   │   ├── test
│   │   │   ├── ├── cat
│   │   │   ├── ├── dog
│   │   │   ├── ├── wild
├── pretrained_models
│   ├── cat2dog.pt
├── commands
│   ├── test_cat2dog.sh
```

Run:
```
bash commands/test_cat2dog.sh
```


## Training
It requires a single GPU with at least 16GB memory. Less GPU memory with a smaller batch size is potentially feasible, although we have not tested it.

### CelebAMask-to-FFHQ
Train encoder and regressor for CelebAMask-HQ mask domain, by using StyleGAN2-FFHQ as the generator backbone. 

Please download [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ) and [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset), put them in `./data/`. Download pre-trained models [stylegan2-ffhq-config-f.pt](https://drive.google.com/file/d/1NghBNgK6x5E9d3yyq_VFns3Lsl9JSR1K/view?usp=drive_link) and [model_ir_se50.pth](https://drive.google.com/file/d/1ZNKfkNg5LUK4aIjck-2Cg90XSW-7Vn3Z/view?usp=drive_link), put them in `./pretrained_models`. The folder structure is
```
Latent-Space-Anchoring
├── data
│   ├── CelebAMask-HQ
│   │   ├── face_parsing
│   │   │   ├── Data_preprocessing
│   │   │   ├── ├── train_img
│   │   │   ├── ├── train_label
│   │   │   ├── ├── test_img
│   │   │   ├── ├── test_label
│   ├── ffhq
│   │   ├── images1024x1024
├── pretrained_models
│   ├── stylegan2-ffhq-config-f.pt
│   ├── model_ir_se50.pth
├── commands
│   ├── train_seg2ffhq.sh
```

Run:
```
bash commands/train_seg2ffhq.sh
```
The training results and model checkpoints will be saved in `./logs/seg2ffhq`.

### Sketch-to-FFHQ
Train encoder and regressor for CUFSF facial sketch domain, by using StyleGAN2-FFHQ as the generator backbone. 

Please download [CUFSF](http://mmlab.ie.cuhk.edu.hk/archive/cufsf/) and [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset), put them in `./data/`. Download pre-trained models [stylegan2-ffhq-config-f.pt](https://drive.google.com/file/d/1A3rXFwAl2Bkuf6Sd4LUqzrjFmgHT7oM-/view?usp=sharing) and [model_ir_se50.pth](https://drive.google.com/file/d/1lcrPgEA7PZk9mQCG2BS7DP6FUdl-4sm6/view?usp=sharing), put them in `./pretrained_models`. The folder structure is
```
Latent-Space-Anchoring
├── data
│   ├── CUFSF
│   │   ├── train
│   │   ├── test
│   ├── ffhq
│   │   ├── images1024x1024
├── pretrained_models
│   ├── stylegan2-ffhq-config-f.pt
│   ├── model_ir_se50.pth
├── commands
│   ├── train_sketch2ffhq.sh
```

Run:
```
bash commands/train_sketch2ffhq.sh
```
The training results and model checkpoints will be saved in `./logs/sketch2ffhq`.

## Diverse Generations
We support diverse model generations.
<p align="left">
<img src="docs/diverse.jpg"/>
</p>

Please follow ***Testing: CelebAMask-to-FFHQ** to prepare dataset and pre-trained models. Run:
```
bash commands/inference_seg2ffhq.sh
```

## High-Resolution Sampling
We support sampling high-resolution (i.e., 1024x1024) mask and images from a random noise.
<p align="left">
<img src="docs/sampling.jpg"/>
</p>

Please follow **Testing: CelebAMask-to-FFHQ** to prepare dataset and pre-trained models. Run:
```
bash commands/sampling_seg2ffhq.sh
```

## Acknowledgements
This implementation is built upon [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch) and [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel).

## Citation
```
@article{huang2023domain,
  author={Huang, Siyu and An, Jie and Wei, Donglai and Lin, Zudi and Luo, Jiebo and Pfister, Hanspeter},
  title={Domain-Scalable Unpaired Image Translation Via Latent Space Anchoring},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2023},
}
```

## Contact
[Siyu Huang](http://siyuhuang.github.io/) (huangsiyutc@gmail.com)



