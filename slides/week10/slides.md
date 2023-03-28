---
marp: true
theme: uncover
paginate: true
---

<!-- _paginate: skip -->

# Progress - March 28th

Complex Valued Autoencoders for Object Discovery

---

### Architecture (VGG16-based)

![width:12in](assets/CAE.jpg)

---

### Recap

-   CAE works for RGB images with channel pooling
    -   Resulting phases clustered closely together
    -   Cluster distances are minimal
-   Why are the clusters distances so small?
    -   Pixel darkness seems to play a role
    -   Investiage CAE response to "uniform" signals
-   Why are the phases clustered so thightly?
    -   Small maginitudes could play a role

---

### Black to White - Untrained

![width:12in](assets/angles_u.jpg)

---

### Black to White - Trained

![width:12in](assets/angles_t.jpg)

---

### Black to White - Analysis

-   Untrained model uniformally distributed
-   Trained model angles between 135 and 225
    -   Or: between $0.2\pi$ and $-0.2\pi$
-   Phase-normalization could prove helpful
    -   Next experiments

---

### Small magnitudes

-   Subtract $0.1$ from reconstruction
    -   Forces last convolutional layer bias to be $\gt 0.1$
    -   Forcing output maginutdes to be $\gt 0.1$
    -   Forcing phases to be more valuable

---

### Small magnitudes - Results

| ![height:3in width:3in](assets/loss1.jpg) | ![height:3in width:3in](assets/loss2.jpg) |
| :---------------------------------------: | :---------------------------------------: |
|                $MSE$ loss                 |           $\%$ of $\rho > 0.1$            |

---

### Small magnitudes - Results

-   Only effective in early stages of training
    -   Up to about $100.000$ datapoints

---

### Channel Difference Regularization

-   Completely fails to optimize

| ![height:3in width:3in](assets/sample.jpg) | ![height:3in width:3in](assets/recon.jpg) | ![height:3in width:3in](assets/phase.jpg) |
| :----------------------------------------: | :---------------------------------------: | :---------------------------------------: |
|                   Sample                   |              Reconstruction               |                  Phases                   |

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
