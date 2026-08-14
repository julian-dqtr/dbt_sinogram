# U-Net 2D Model and Dataset Documentation

This document explains the architecture of the implemented U-Net 2D model and the generation pipeline of the dataset used for sinogram completion in Digital Breast Tomosynthesis (DBT).

## 1. 2D U-Net Model Architecture

The core model used for the sinogram completion task is a 2D U-Net implemented in `src_2D/models/Unet2D/unet_2d.py`. 

### Key Characteristics:
*   **Base Framework:** The model uses **MONAI's `DynUNet`** (Dynamic U-Net), which is highly optimized for medical imaging tasks.
*   **Structure:** It follows the classic encoder-decoder architecture with skip connections. The encoder gradually downsamples the input spatial dimensions while increasing the number of feature channels, and the decoder upsamples back to the original resolution.
*   **Layer Configuration:**
    *   **Input/Output Channels:** `in_channels=1`, `out_channels=1` (since it maps a 2D single-channel incomplete sinogram to a 2D single-channel complete sinogram).
    *   **Kernel Sizes:** `3x3` convolutions across all levels.
    *   **Filters:** Feature map sizes scale up by factors of 2: `[filters, filters*2, filters*4, filters*8]` (default `filters=16`, scaling to `16, 32, 64, 128`).
    *   **Downsampling (Strides):** Strides are set to `[[1, 1], [2, 2], [2, 2], [2, 2]]`. The first layer maintains spatial resolution, and subsequent layers downsample by a factor of 2, resulting in a total spatial downsampling factor of 8.
    *   **Upsampling:** `2x2` upsample kernels.
    *   **Normalization:** Instance Normalization (`norm_name="instance"`) is used instead of Batch Normalization, which is often preferred for varying batch sizes and contrast enhancement in imaging tasks.
*   **Dynamic Padding (`_pad_to_divisor`):** To ensure that skip connections align perfectly between the encoder and decoder, the input tensor is dynamically padded (using reflection or zero padding at the edges) so that its height and width are perfectly divisible by 8 (the total stride product). The output is subsequently cropped back to the original input shape.

## 2. Dataset Generation Pipeline

The dataset, implemented in `src_2D/data/dataset_2d.py` (`SinogramCompletionDataset`), dynamically generates training pairs on the fly. 

The goal of the dataset is to provide pairs of:
1.  **Input:** Incomplete sinogram (simulating DBT limited-angle acquisition).
2.  **Target:** Complete sinogram (simulating full-angle CT acquisition).
3.  **Ground Truth:** The original 2D phantom (image).

### Phantom Generation
The `PhantomGenerator` creates diverse 2D object masks to act as ground truth tissues or structures. The supported types (`phantom_type`) include:
*   **`shepp_logan`:** The classic Shepp-Logan medical imaging phantom (via `skimage`).
*   **`ellipses`:** Randomly generated, rotated, and positioned ellipses.
*   **`blobs`:** Gaussian blobs mimicking soft tissue masses.
*   **`rectangles`:** Random rectilinear structures.
*   **`mixed`:** Randomly chooses between any of the above for high dataset diversity.

**Augmentation:** Every phantom undergoes random rigid transformation (rotation, scaling, translation) to avoid overfitting to specific positions. The output phantom values are clamped between `0.0` and `1.0`.

### Projection (Forward Model)
To create sinograms from the generated 2D phantoms, a forward projection is applied:
*   **ASTRA Toolbox Projector:** Uses the high-performance GPU-accelerated ASTRA toolbox (`FP_CUDA`).
*   **Geometry:** Based on `DBTGeometryConfig`. It configures a parallel/fan-beam geometry mapping real-world dimensions (source/detector radius in mm, detector pixel size) into an ASTRA-compatible projection geometry.
*   **Normalization:** The full sinogram is divided by a `global_sino_norm` factor of `100.0` to roughly scale values into the `[-1, 1]` range, aiding neural network convergence.
*   **Noise (Optional):** Supports addition of Poisson noise to simulate realistic sensor variations.

### Incomplete Sinogram Masking
The input to the model (the "incomplete sinogram") is created by taking the full `[Views, Detectors]` sinogram and applying a mask.
*   The system checks which full-arc angles fall within the defined DBT limited acquisition window (e.g., `-25°` to `+25°`).
*   The `_crop_to_acquired_views` method zeros out all sinogram rows corresponding to angles *outside* this window, mimicking unacquired data.

### Final Output 
For every sample, `__getitem__` returns a tuple of 3 tensors of shape `[1, H, W]`:
`(incomplete_sinogram, full_sinogram, phantom)`
