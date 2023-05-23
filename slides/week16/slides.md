---
marp: true
theme: uncover
paginate: true
---

<!-- _paginate: skip -->

# Progress - May 23rd

Complex Valued Autoencoders for Object Discovery

---

### Recent Progress

-   Optimizing transforms/hyperparameters
-   Generating 500k CLEVR images
-   Re-thinking contrastive approach
    -   Literature research

---

### Overlapping Objects

-   Previous results

| ![height:2.5in width:2.5in](assets/img0.jpg) | ![height:2.5in width:2.5in](assets/pha_p0.jpg) | ![height:2.5in width:2.5in](assets/img1.jpg) | ![height:2.5in width:2.5in](assets/pha_p1.jpg) |
| :------------------------------------------: | :--------------------------------------------: | :------------------------------------------: | :--------------------------------------------: |
|                                              |                                                |                                              |                                                |

---

### Overlapping Objects

-   Current results

| ![height:2.5in width:2.5in](assets/img0.jpg) | ![height:2.5in width:2.5in](assets/pha0.jpg) | ![height:2.5in width:2.5in](assets/img1.jpg) | ![height:2.5in width:2.5in](assets/pha1.jpg) |
| :------------------------------------------: | :------------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|                                              |                                              |                                              |                                              |

---

### PiPA - Chen et al.

![width:9in](assets/diagram.jpg)

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
