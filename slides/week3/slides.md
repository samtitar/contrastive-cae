---
marp: true
theme: uncover
---

# Progress - Febuary 6th

Complex Valued Autoencoders for Object Discovery

---

### Challenges

-   Checkerboard pattern
    -   MaxUnpool > Upsample
    -   Local Contrast Reg.

![bg right vertical height:3in](assets/contrast_c.jpg)
![bg right height:3in](assets/contrast_p.jpg)

---

### Challenges

-   Local Contrast Reg.
    -   Low variability

![bg right vertical height:3in](assets/contrast2_c.jpg)
![bg right height:3in](assets/contrast2_p.jpg)

---

### Contrastive Learning

-   Patches > Learned mask
    -   Masked features
    -   Immediate clustering
    -   Unsupervised pre-training

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
