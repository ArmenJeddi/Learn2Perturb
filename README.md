## Learn2Perturb: a noise injection method for adversarial robustness

(Pytorch 1.0)

![image info](./teaser.png)

This repository contains an implementation corresponding to our CVPR 2020 paper: "[Learn2Perturb: an End-to-end Feature Perturbation Learning to Improve Adversarial Robustness](https://arxiv.org/abs/2003.01090)". A brief presentation of our work is available at [this youtube link](https://youtu.be/KUJIDZo8azo).

If you find our work useful, please cite it as follows:
```bibtex
@inproceedings{jeddi2020learn2perturb,
  title={Learn2Perturb: an End-to-end Feature Perturbation Learning to Improve Adversarial Robustness},
  author={Jeddi, Ahmadreza and Shafiee, Mohammad Javad and Karg, Michelle and Scharfenberger, Christian and Wong, Alexander},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={},
  year={2020}
}
```

This repository includes PyTorch implementation of:

- Adversarial attacks 
    - FGSM
    - PGD
    - EOT (Expectation Over Transformations [1])
- Baseline models used in experiments
- Learn2Perturb Modules

Pytorch implementations for other adversarial attacks used in this work: [C&W](https://github.com/kkew3/pytorch-cw2) and [few-pixel attack](https://github.com/DebangLi/one-pixel-attack-pytorch)


### References
- [1] Athalye, A., Engstrom, L., Ilyas, A., and Kwok, K. Syn- thesizing robust adversarial examples. arXiv preprint arXiv:1707.07397, 2017.
