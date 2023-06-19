# Latent-Space-Anchoring

PyTorch implementation of

**Domain-Scalable Unpaired Image Translation via Latent Space Anchoring**

[Siyu Huang<sup>*</sup>](https://siyuhuang.github.io) (Harvard), [Jie An<sup>*</sup>](https://www.cs.rochester.edu/u/jan6/) (Rochester), [Donglai Wei](https://donglaiw.github.io/) (BC), [Zudi Lin](https://zudi-lin.github.io/) (Amazon Alexa), [Jiebo Luo](https://www.cs.rochester.edu/u/jluo/) (Rochester), [Hanspeter Pfister](https://vcg.seas.harvard.edu/people/hanspeter-pfister) (Harvard)  
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

<p align="center">
<img src="docs/4domain.jpg"/>
</p>

Given an unpaired image-to-image translation (UNIT) model trained on certain domains, it is challenging to incorporate new domains. This work includes a domain-scalable UNIT method, termed as latent space anchoring, anchors images of different domains to the same latent space of frozen GANs by learning lightweight encoder and regressor models to reconstruct single-domain images. In inference, the learned encoders and decoders of different domains can be arbitrarily combined to translate images between any two domains without fine-tuning.

<p align="center">
<img src="docs/intro.png" width="600px"/>
</p>


