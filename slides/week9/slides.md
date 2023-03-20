---
marp: true
theme: uncover
paginate: true
---

<!-- _paginate: skip -->

# Progress - March 20th

Complex Valued Autoencoders for Object Discovery

---

### Recap - Current Challenges

-   Working towards RGB images
    -   Original paper: phases become meaningless
-   Small magnitudes
    -   Original paper: apply masks

---

### Recap - Proposed Solutions

-   Channel pooling: take mean over channels
    -   Magnitude-based pooling
        -   Magnitude-max
        -   Forcing larger magnitudes

---

### Channel Pooling for RGB

![width:10in](assets/Pooling.jpg)

---

### Architecture (VGG16-based)

![width:12in](assets/CAE.jpg)

---

### Magnitude MaxPooling Results

| ![height:3in width:3in](assets/cars1_s.jpg) | ![height:3in width:3in](assets/cars1_c.jpg) | ![height:3in width:3in](assets/cars1_p.jpg) |
| :-----------------------------------------: | :-----------------------------------------: | :-----------------------------------------: |
|                   Sample                    |                   Phases                    |                    Polar                    |

---

### Magnitude MaxPooling Results

-   CAE produces meaningful phases for RGB images
-   CAE produces larger magnitudes
    -   Small magnitudes are implicitly penalized
    -   $\gt 90\%$ of magnitudes are $\gt 0.1$
        ($0.1 =$ Mask threshold in original paper)

---

### Clustering Update

-   Clustering becomes more trivial
    -   All information encoded in phases
        -   Projection to eucledian space not necessary
        -   K-Means not necessary
    -   Apply histogram to phases directly
        -   Number of bins = number of clusters
        -   Better clustering resolution control
        -   K is more dynamic

---

### Clustering Update

| ![height:3in width:3in](assets/cars1_s.jpg) | ![height:3in width:3in](assets/cars1_p.jpg) | ![height:3in width:3in](assets/cars1_pf.jpg) |
| :-----------------------------------------: | :-----------------------------------------: | :------------------------------------------: |
|                   Sample                    |                   Phases                    |                     Bins                     |

---

### Clustering Update

-   Only classify which bins are "active"
    -   $k \times 224 \times 224$ to just $k$
    -   Histogram approximation maintains gradient
        -   Enables direct contrastive optimization
        -   Push bins away from each other

---

### Histogram-based Contrastive Learning

-   Push distance (angle) between "active" bins
    -   Collapses to few bin

---

### CelabA-HQ-Mask Dataset

-   Segmentations provided
-   $2 \times$ the number of images

---

### CelabA-HQ-Mask Results #1

| ![height:3in width:3in](assets/celeba1_s.jpg) | ![height:3in width:3in](assets/celeba1_r.jpg) | ![height:3in width:3in](assets/celeba1_c.jpg) |
| :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: |
|                    Sample                     |                Reconstruction                 |                    Phases                     |

---

### CelabA-HQ-Mask Results #1

| ![height:3in width:3in](assets/celeba1_s.jpg) | ![height:3in width:3in](assets/celeba1_c.jpg) | ![height:3in width:3in](assets/celeba1_p.jpg) |
| :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: |
|                    Sample                     |                    Phases                     |                     Polar                     |

---

### CelabA-HQ-Mask Results #2

| ![height:3in width:3in](assets/celeba2_s.jpg) | ![height:3in width:3in](assets/celeba2_r.jpg) | ![height:3in width:3in](assets/celeba2_c.jpg) |
| :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: |
|                    Sample                     |                Reconstruction                 |                    Phases                     |

---

### CelabA-HQ-Mask Results #2

| ![height:3in width:3in](assets/celeba2_s.jpg) | ![height:3in width:3in](assets/celeba2_c.jpg) | ![height:3in width:3in](assets/celeba2_p.jpg) |
| :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: |
|                    Sample                     |                    Phases                     |                     Polar                     |

---

### Histogram Count Maximization

-   With histogram-based clustering
    -   Less need for angle distance
    -   More control over "super-pixel" size
    -   Apply area size loss to histogram bins

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
