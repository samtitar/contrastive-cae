---
marp: true
theme: uncover
---

# Progress - January 30th

Complex Valued Autoencoders for Object Discovery

---

### Recap

-   Restructured existing codebase
-   Reproduction of existing results
-   Experimentation with CIFAR & ImageNet
    -   Larger model ⟶ better reconstruction
-   Improve phase assignments
    -   Patch-based contrastive learning
    -   Pretrained features for similarity matching

---

### Progress

-   Implementation of new layers
    -   `ComplexMaxPool2d`
    -   `ComplexUnmaxPool2d`
-   Implementation of VGG-16 based autoencoder
    -   Used in U-Net/SegNet for segmentation
    -   Configurable number of features
    -   Pretraining on StanfordCars
-   Implementation of patching & matching

---

### Stanford Cars - Example 1

| ![height:3in width:3in](assets/cars1_s.jpg) | ![height:3in width:3in](assets/cars1_r.jpg) | ![height:3in width:3in](assets/cars1_c.jpg) |
| :-----------------------------------------: | :-----------------------------------------: | :-----------------------------------------: |
|                   Sample                    |               Reconstruction                |                   Phases                    |

---

### Stanford Cars - Example 1

| ![height:3in width:3in](assets/cars1_s.jpg) | ![height:3in width:3in](assets/cars1_p.jpg) | ![height:3in width:3in](assets/cars1_c.jpg) |
| :-----------------------------------------: | :-----------------------------------------: | :-----------------------------------------: |
|                   Sample                    |               Reconstruction                |                   Phases                    |

---

### Stanford Cars - Example 2

| ![height:3in width:3in](assets/cars2_s.jpg) | ![height:3in width:3in](assets/cars2_r.jpg) | ![height:3in width:3in](assets/cars2_c.jpg) |
| :-----------------------------------------: | :-----------------------------------------: | :-----------------------------------------: |
|                   Sample                    |               Reconstruction                |                   Phases                    |

---

### Stanford Cars - Example 2

| ![height:3in width:3in](assets/cars2_s.jpg) | ![height:3in width:3in](assets/cars2_p.jpg) | ![height:3in width:3in](assets/cars2_c.jpg) |
| :-----------------------------------------: | :-----------------------------------------: | :-----------------------------------------: |
|                   Sample                    |               Reconstruction                |                   Phases                    |

---

### Additional Research

- FreeSOLO
    - FreeMask: Coarse mask extraction using learned features
    - SOLO refinment using self-training
        - Weak supervision loss on coarse masks
- CutLER
    - Very recent (few days ago) - still reading
    - ViT ⟶ MaskCut ⟶ Detector (self-training)
        - MaskCut outperforms FreeMask

---

# Next up

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
