---
marp: true
theme: uncover
paginate: true
---

<!-- _paginate: skip -->

# Progress - June 7th

Complex Valued Autoencoders for Object Discovery

---

![width:12in](assets/ctcae.jpg)

---

### Model Improvements

-   Also Upsampling
    -   No Deconvolution
-   No Image Rescaling Layer
    -   No Sigmoid
-   Channel Reduction not Specified

---

### Contrastive Approach

![width:12in](assets/ctcae2.jpg)

---

### Results

![width:11in](assets/ctcae_res.jpg)

---

### Re-Creation

![width:11in](assets/res1.jpg)

---

### Patch-based Contrast

![width:11in](assets/con.jpg)

---

### Results

| ![height:2.5in width:2.5in](assets/cs1.jpg) | ![height:2.5in width:2.5in](assets/cp1.jpg) | ![height:2.5in width:2.5in](assets/cs2.jpg) | ![height:2.5in width:2.5in](assets/cp2.jpg) |
| :------------------------------------------: | :------------------------------------------: | :------------------------------------------: | :------------------------------------------: |
|                                              |                                              |                                              |                                              |

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
