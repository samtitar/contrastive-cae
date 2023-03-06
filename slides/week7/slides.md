---
marp: true
theme: uncover
---

# Progress - March 6th

Complex Valued Autoencoders for Object Discovery

---

### Current Architecture (VGG16-based)

![width:12in](assets/CAE.jpg)

---

### Current Challenges - Recap

-   Moving towards RGB
    -   All encoding ends up in magnitude
    -   Elaborately described in paper
-   Some magnitudes are small

---

### RGB Images - Channel Collapse

![width:10in](assets/Collapse.jpg)

-   Apply to all layers or only to the last layer

---

### RGB Images - Results

| ![height:3in width:3in](assets/sample.jpg) | ![height:3in width:3in](assets/all_layer_r.jpg) | ![height:3in width:3in](assets/last_layer_r.jpg) |
| :----------------------------------------: | :---------------------------------------------: | :----------------------------------------------: |
|                   Sample                   |                    All Layer                    |                    Last Layer                    |

---

### RGB Images - Results

| ![height:3in width:3in](assets/sample.jpg) | ![height:3in width:3in](assets/all_layer_p.jpg) | ![height:3in width:3in](assets/last_layer_p.jpg) |
| :----------------------------------------: | :---------------------------------------------: | :----------------------------------------------: |
|                   Sample                   |                    All Layer                    |                    Last Layer                    |

---

### RGB Images - Results

| ![height:3in width:3in](assets/no_collapse_m.jpg) | ![height:3in width:3in](assets/all_layer_m.jpg) | ![height:3in width:3in](assets/last_layer_m.jpg) |
| :-----------------------------------------------: | :---------------------------------------------: | :----------------------------------------------: |
|                    No Collapse                    |                    All Layer                    |                    Last Layer                    |

---

### RGB Images - Next Steps

-   Channel Collapse
    -   Magnitude-based channel-pooling
    -   Forcing larger magnitudes

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
