---
marp: true
theme: uncover
---

# Progress - Febuary 13th

Complex Valued Autoencoders for Object Discovery

---

### Progressive Growing

| ![height:3in width:3in](assets/pro_s.jpg) | ![height:3in width:3in](assets/pro_r.jpg) | ![height:3in width:3in](assets/pro_c.jpg) |
| :---------------------------------------: | :---------------------------------------: | :---------------------------------------: |
|                  Sample                   |              Reconstruction               |                  Phases                   |

---

### Progressive Growing

| ![height:3in width:3in](assets/pro_s.jpg) | ![height:3in width:3in](assets/pro_p.jpg) | ![height:3in width:3in](assets/pro_c.jpg) |
| :---------------------------------------: | :---------------------------------------: | :---------------------------------------: |
|                  Sample                   |                   Polar                   |                  Phases                   |

---

### Polar Interpolation and Pooling

-   Max(Un)Pooling on polar projection
-   Interpolation on polar projection

---

### Improved Clustering

-   Opening/Closing
-   Uniform kernel
-   Connected components

---

### Polar Interpolation

| ![height:3in width:3in](assets/interp_s.jpg) | ![height:3in width:3in](assets/interp_r.jpg) | ![height:3in width:3in](assets/interp_c.jpg) |
| :------------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|                    Sample                    |                Reconstruction                |                    Phases                    |

---

### Polar Interpolation

| ![height:3in width:3in](assets/interp_s.jpg) | ![height:3in width:3in](assets/interp_p.jpg) | ![height:3in width:3in](assets/interp_l.jpg) |
| :------------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|                    Sample                    |                    Polar                     |

---

### Polar Pooling

| ![height:3in width:3in](assets/pool_s.jpg) | ![height:3in width:3in](assets/pool_r.jpg) | ![height:3in width:3in](assets/pool_c.jpg) |
| :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
|                   Sample                   |               Reconstruction               |                   Phases                   |

---

### Polar Pooling

| ![height:3in width:3in](assets/pool_s.jpg) | ![height:3in width:3in](assets/pool_p.jpg) | ![height:3in width:3in](assets/pool_l.jpg) |
| :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
|                   Sample                   |                   Polar                    |                  Clusters                  |

---

### Contrastive Learning

-   Masked convolutions
    -   Partial convolution (Liu et al.)
-   Bipartite mask matching
    -   Mask IOU metric
-   Masked feature similarity
    -   Object centric representation

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
