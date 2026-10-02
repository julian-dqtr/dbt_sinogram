from __future__ import annotations

from src_2D.models.Unet2D.unet_2d import SinogramUNet


class Unet2dHLCC(SinogramUNet):
    """
    U-Net trained with the Helgason-Ludwig Consistency Conditions (HLCC).

    The NETWORK is strictly identical to ``SinogramUNet`` (same backbone, residual
    learning, soft data consistency). The only difference between the "UNet2D" and
    "UNet2dHLCC" experiments is the training loss (MSE vs MSE + annealed HLCC penalty, see
    ``SinoSheavesNN/physics_loss.py``), which makes the comparison a clean ablation of the
    physics term.
    """
