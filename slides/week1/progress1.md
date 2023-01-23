---
marp: true
theme: uncover

---
# Progress - January 23th
Complex Valued Autoencoders for Object Discovery

---
### Restructure of Framework
- Existing codebase
    - Functional approach
    - Nested configurations
- New structure
    - Modular approach
    - Per-module configuration
    - More compatible with PyTorch
- Room for other complex-valued modules

---
### MNIST & Shape
| ![height:3in](assets/mnist_s.jpg) | ![height:3in](assets/mnist_r.jpg) | ![height:3in](assets/mnist_c.jpg) |
| :---: | :---: | :---: |
| Sample | Reconstruction |Phases |

---
### MNIST & Shape
| ![height:3in](assets/mnist_s.jpg) | ![height:3in](assets/mnist_p.jpg) | ![height:3in](assets/mnist_c.jpg) |
| :---: | :---: | :---: |
| Sample | Phases | Polar |

---
### CIFAR100 Example 1
| ![height:3in](assets/cifar1_s.jpg) | ![height:3in](assets/cifar1_r.jpg) | ![height:3in](assets/cifar1_c.jpg) |
| :---: | :---: | :---: |
| Sample | Reconstruction | Phases |

---
### CIFAR100 Example 1
| ![height:3in](assets/cifar1_s.jpg) | ![height:3in](assets/cifar1_p.jpg) | ![height:3in](assets/cifar1_c.jpg) |
| :---: | :---: | :---: |
| Sample | Phases | Polar |

---
### CIFAR100 Example 2
| ![height:3in](assets/cifar2_s.jpg) | ![height:3in](assets/cifar2_r.jpg) | ![height:3in](assets/cifar2_c.jpg) |
| :---: | :---: | :---: |
| Sample | Reconstruction | Phases |

---
### CIFAR100 Example 2
| ![height:3in](assets/cifar2_s.jpg) | ![height:3in](assets/cifar2_p.jpg) | ![height:3in](assets/cifar2_c.jpg) |
| :---: | :---: | :---: |
| Sample | Phases | Polar |

---
### ImageNet
| ![height:3in](assets/imagenet_s.jpg) | ![height:3in](assets/imagenet_r.jpg) | ![height:3in](assets/imagenet_c.jpg) |
| :---: | :---: | :---: |
| Sample | Reconstruction | Phases |

---
### ImageNet
| ![height:3in](assets/imagenet_s.jpg) | ![height:3in](assets/imagenet_p.jpg) | ![height:3in](assets/imagenet_c.jpg) |
| :---: | :---: | :---: |
| Sample | Phases | Polar |

---
### Challenges
- Poor reconstruction quality
    :heavy_check_mark: Larger model
    :heavy_check_mark: Larger images (ImageNet)
    &nbsp;&nbsp;&nbsp;&nbsp; ↳ More phase assignments ⟶ Less separability
- Poor phase assignments (Unseparability)
    :heavy_multiplication_x: Contrastive learning
- RGB seperation
    :heavy_multiplication_x: New evaluation of complex output

---
### Literature

- **DetCo**: Unsupervised Contrastive Learning
- **DINO** Emerging Properties in Self-Supervised ViT
- **PatchNet**: Unsupervised Object Discovery
- **DAIC**: Deep Adaptive Image Clustering

---
#### DetCo (Xie et al.)
![height:5in](assets/detco.jpg)
- Global and local augmentations

---
#### DINO (Caron et al.)
![height:4in](assets/dino.jpg)
- Also global and local augmentations
- EMA instead of contrastive learning

---
#### PatchNet (Moon et al.)
![height:3.5in](assets/patchnet.jpg)
- Global and local patches
- Dissimilarity matching
    - Pattern space

---
#### DAIC (Chang et al.)
![height:3.5in](assets/daic.jpg)
- Image embedding-based clustering
    - Cosine similarity ranking
    - Extend training set iteratively

---
### Continuation
- Patch-based approach
    - Local and global patches
    - Augmentations for robustness
- Pretrain CAE on dataset
- Obtain patch similarites
    - Using bottleneck features
    - Using patch pair overlap (IoU)
- Optimize phase-distance (**contrastive**)
    - Bind patches that are "similar"
    - Unbind patches that are "dissimilar"

<style>
    h1, h2, h3, h4, h5 {
        color: #6b32a8
    }

    ul {
        width: 100%;
        list-style: none;
    }

    ul li::before {
        content: "\2022";
        color: #6b32a8;
        font-weight: bold;
        display: inline-block;
        width: 1em;
        margin-left: -1em;
    }

    ul ul li::before {
        opacity: 0.5;
    }
</style>