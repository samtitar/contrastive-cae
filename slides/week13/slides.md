---
marp: true
theme: uncover
paginate: true
---

<!-- _paginate: skip -->

# Progress - April 17th

Complex Valued Autoencoders for Object Discovery

---

### Recap

-   Testing phase initialization
    -   Superpixels
    -   Positional encoding
-   Testing momentum-based contrastive learning
    -   Different augmentations
-   Rendering matte only CLEVR dataset

---

### Architecture (VGG16-based)

![width:12in](assets/CAE.jpg)

---

### Positional Phase Results - CLEVR

| ![height:3in width:3in](assets/clevr_s.jpg) | ![height:3in width:3in](assets/clevr_p_ns.jpg) | ![height:3in width:3in](assets/clevr_c_ns.jpg) |
| :-----------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
|                   Sample                    |                     Phases                     |                     Polar                      |

---

### Positional Phase Results - CLEVR

| ![height:3in width:3in](assets/clevr_s.jpg) | ![height:3in width:3in](assets/clevr_p.jpg) | ![height:3in width:3in](assets/clevr_c.jpg) |
| :-----------------------------------------: | :-----------------------------------------: | :-----------------------------------------: |
|                   Sample                    |                   Phases                    |                    Polar                    |

---

### Positional Phase Results - Faces

| ![height:3in width:3in](assets/celeb_s.jpg) | ![height:3in width:3in](assets/celeb_p.jpg) | ![height:3in width:3in](assets/celeb_c.jpg) |
| :-----------------------------------------: | :-----------------------------------------: | :-----------------------------------------: |
|                   Sample                    |                   Phases                    |                    Polar                    |

---

### CLEVR - Comparison

| ![height:2.5in width:2.5in](assets/clevr_p_np.jpg) | ![height:2.5in width:2.5in](assets/clevr_r_np.jpg) | ![height:2.5in width:2.5in](assets/clevr_p.jpg) | ![height:2.5in width:2.5in](assets/clevr_r.jpg) |
| :------------------------------------------------: | :------------------------------------------------: | :---------------------------------------------: | :---------------------------------------------: |
|                      0-Phase                       |                      0-Phase                       |                     P-Phase                     |                     P-Phase                     |

---

### CLEVR Training -> Faces Eval

| ![height:2.5in width:2.5in](assets/cf_s1.jpg) | ![height:2.5in width:2.5in](assets/cf_p1.jpg) | ![height:2.5in width:2.5in](assets/cf_s2.jpg) | ![height:2.5in width:2.5in](assets/cf_p2.jpg) |
| :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: |
|                   Sample 1                    |                   Result 1                    |                   Sample 2                    |                   Result 2                    |

---

### CLEVR Training -> Cars Eval

| ![height:2.5in width:2.5in](assets/cc_s1.jpg) | ![height:2.5in width:2.5in](assets/cc_p1.jpg) | ![height:2.5in width:2.5in](assets/cc_s2.jpg) | ![height:2.5in width:2.5in](assets/cc_p2.jpg) |
| :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: |
|                   Sample 1                    |                   Result 1                    |                   Sample 2                    |                   Result 2                    |

---

### Momentum-Based Contrastive

![width:9in](assets/MoCAE.jpg)

---

### Augmentations

-   2 teacher augmentations
-   8 student augmentations
-   Solarize, ColorJitter, Grayscale, GaussianBlur

---

### Results

| ![height:3in width:3in](assets/clevr_s.jpg) | ![height:3in width:3in](assets/clevr_p_mo.jpg) | ![height:3in width:3in](assets/clevr_c_mo.jpg) |
| :-----------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
|                   Sample                    |                     Phases                     |                     Polar                      |

---

### Next up

-   Contrastive approach
    -   Add crop to augmentation
    -   Add (histogram-based) discretization
        -   IOU matching instead of 1 to 1 pixel loss

<style>
    section {
        background: white;
    }

    h1, h2, h3, h4, h5 {
        color: #78588a;
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

    section::after {
        content: attr(data-marpit-pagination) '/' attr(data-marpit-pagination-total);
        background: None;
    }
</style>
