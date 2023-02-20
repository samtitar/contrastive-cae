---
marp: true
theme: uncover
---

# Progress - Febuary 20th

Complex Valued Autoencoders for Object Discovery

---

### Mask Training

![width:8in](assets/CAE.jpg)

---

### Improved Pseudo Clustering

-   Connected components
-   Binary closing
-   Binary opening

---

### Mask Training

| ![height:3in width:3in](assets/res1_s.jpg) | ![height:3in width:3in](assets/res1_l.jpg) | ![height:3in width:3in](assets/res1_m.jpg) |
| :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
|                   Sample                   |                Pseudo-Masks                |               Decoded Masks                |

---

### Mask Training

| ![height:3in width:3in](assets/res2_s.jpg) | ![height:3in width:3in](assets/res2_l.jpg) | ![height:3in width:3in](assets/res2_m.jpg) |
| :----------------------------------------: | :----------------------------------------: | :----------------------------------------: |
|                   Sample                   |                Pseudo-Masks                |               Decoded Masks                |

---

### Next Stage: Latent Optimization

![width:5in](assets/CAE2.jpg)

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
