---
marp: true
theme: uncover
paginate: true
---

<!-- _paginate: skip -->

# Progress - April 25th

Complex Valued Autoencoders for Object Discovery

---

### Recap

-   Momentum-based contrastive learning
    -   Different augmentations
    -   Discretize segmentations

---

### Architecture (VGG16-based)

![width:12in](assets/CAE.jpg)

---

### Momentum-based Contrastive Learning

![width:9in](assets/MoCAE.jpg)

---

### Discrete Masks

![width:8in](assets/SegCAE.jpg)

```python
seg = einsum("bchw, bcs -> bshw", (bins, probs))
```

---

### MoCAE Discrete Masks - Results

| ![height:3in width:3in](assets/mc_s.jpg) | ![height:3in width:3in](assets/mc_c.jpg) | ![height:3in width:3in](assets/mc_p.jpg) |
| :--------------------------------------: | :--------------------------------------: | :--------------------------------------: |
|                  Sample                  |                  Phases                  |                  Polar                   |

---

### Discrete Masks BCE - Results

![width:9in](assets/dmt.jpg)

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
