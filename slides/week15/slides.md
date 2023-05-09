---
marp: true
theme: uncover
paginate: true
---

<!-- _paginate: skip -->

# Progress - May 9th

Complex Valued Autoencoders for Object Discovery

---

### Recap

-   Writing
-   Literature review
-   Evaluation
-   Segmentation transfer

---

### Current Contributions

-   ComplexLayers
    -   `ComplexMaxPooling` - Polar-based max.
    -   `ComplexUpsampling` - Polar-based interpolation
    -   `ComplexChannelPooling` - Polar-based avg.
-   Magnitude Enhancement
-   Positional Phase Input
    -   Effectiveness explained as regularization

---

### Current Weaknesses - 1/2

-   Lacking evaluation
    -   Comparison with other methods
    -   Evaluation metrics
-   Semi-supervision unexplored
    -   Synthesized (CLEVR) to natural (Faces)
    -   Segmentation head
-   Dataset sizes unexplored
    -   How much synthesized data is required

---

### Current Weaknesses - 2/2

-   Magnitude enhancement parameter unexplored
    -   Increasing value seems to improve results
-   Different phase inputs unexplored

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
