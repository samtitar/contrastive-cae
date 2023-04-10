---
marp: true
theme: uncover
paginate: true
---

<!-- _paginate: skip -->

# Progress - April 10th

Complex Valued Autoencoders for Object Discovery

---

### Recap

-   Testing different pooling techniques
    -   Weighted average
    -   Softmax weighted average
    -   Gumbel softmax weighted average
-   Testing phase initialization
    -   Superpixels

---

### This Week

-   Code for ARI Evaluation on CLEVR dataset
-   Experiments with:
    -   Superpixel based phase initialization
    -   Hard forcing magnitude > 0.1
    -   Momentum encoder

---

### Superpixel Initialization

![width:12in](assets/superpix.jpg)

---

### Superpixel Initialization

| ![height:3in width:3in](assets/clevr_ps_s.jpg) | ![height:3in width:3in](assets/clevr_ps_p.jpg) | ![height:3in width:3in](assets/clevr_ps_c.jpg) |
| :--------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
|                     Sample                     |                     Phases                     |                     Polar                      |

---

### Forcing Magnitude > 0.1

| ![height:3in width:3in](assets/clevr_ps_s.jpg) | ![height:3in width:3in](assets/clevr_fm_p.jpg) | ![height:3in width:3in](assets/clevr_fm_c.jpg) |
| :--------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
|                     Sample                     |                     Phases                     |                     Polar                      |

---

### Momentum Encoder

![width:9in](assets/MoCAE.jpg)

---

### Momentum Encoder

| ![height:3in width:3in](assets/clevr_ps_s.jpg) | ![height:3in width:3in](assets/clevr_me_p.jpg) | ![height:3in width:3in](assets/clevr_me_c.jpg) |
| :--------------------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
|                     Sample                     |                     Phases                     |                     Polar                      |

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
