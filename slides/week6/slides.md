---
marp: true
theme: uncover
---

# Progress - Febuary 27th

Complex Valued Autoencoders for Object Discovery

---

### Magnitude Norm Challenges

-   Inversely proportional regularization
-   Bottleneck activation
    -   Features only applied to magnitudes
    -   Normally distributed features

---

### Smaller Clustering Model

-   MLP
    -   Huge number of parameters
-   Small CNN

---

### RGB Training

-   Challenges:
    -   Seperation becomes trivial
    -   Phases become meaningless
    -   Everyting encoded in magnitude
-   Channel pooling
    -   Min./Max./Avg.
    -   Copy information to each channel
    -   At each convolutional block

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
