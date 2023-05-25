# Experiment design
## Setup

### Dataset
 - Stats, splits
### Metrics
 - 2 set of metrics (AR, AS)
 - Choose 1 set per task beforehand
### General approach
 - Fix augmentation, LR, optimizer, scheduler
 - (May be different for video / positions)
 - Order:
    - Data (Graph /Transformer)
    - Variants of Unimodal approaches (head)
        - (Graph model & Input format)
    - Uni- vs Multimodal
- "Optional":
    - Sequence length / sampling rate
    - augmentations
    - post-processing & action spotting
    - pooling (best model)

### Data

 - Completely unbalanced
 - Balanced, upsampled shots
 - Balanced, downsampled passes
 - Balanced, #background to #shots / #passes
 - Each with or without Overlap

### Transformations

 (- No Transformations
 - Instance-wise)
 - Batch-wise

### Model architecture

For video:
 - MViT with/out NETVLAD(++)
 - (ResNet / lightweight cnn)

For positions:
 - Transformer
 - GAT
 - GIN
 - GCN
 - GraphSAGE (?)
 - (Image-based)

Head:
    Normal
    Normal + Regression

### Position format

- Raw
- With Team Indicator
- Position absolute or relative
- Graph structure (edge criteria and eps nbhood)

### Hyperparameters

 - things like LR, Optimizer, scheduler etc.
 -> not that interesting or insightful
 - sequence length and sampling rate 
 (need plot to visualize dataset/classes anyways)

 ### Post Processing

 - NMS, convolution + threshold
 - Finding peaks and spacing anchors

 ### Pooling (late fusion only)

 - Simple concat
 - Average of both stream's predictions
 - "Cross-modal fusion"