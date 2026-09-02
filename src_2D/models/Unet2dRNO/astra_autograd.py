from __future__ import annotations

import astra
import numpy as np
import torch


class AstraProjection(torch.autograd.Function):
    """
    Differentiable Forward Projection using ASTRA CUDA.
    Maps a 2D spatial image [B, 1, H, W] to a Sinogram [B, 1, Angles, Detectors].
    """

    @staticmethod
    def forward(ctx, image: torch.Tensor, vol_geom: dict, proj_geom: dict, projector_id: int) -> torch.Tensor:
        ctx.vol_geom = vol_geom
        ctx.proj_geom = proj_geom
        ctx.projector_id = projector_id

        batch_size = image.shape[0]
        device = image.device

        image_np = np.ascontiguousarray(image.detach().cpu().numpy())
        sinograms = []
        
        gpu_id = device.index if device.index is not None else 0
        astra.set_gpu_index(gpu_id)

        for i in range(batch_size):
            vol_id = astra.data2d.create("-vol", vol_geom, image_np[i, 0])
            sino_id = astra.data2d.create("-sino", proj_geom)

            cfg = astra.astra_dict("FP_CUDA")
            cfg["ProjectorId"] = projector_id
            cfg["VolumeDataId"] = vol_id
            cfg["ProjectionDataId"] = sino_id

            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)

            sino_out = astra.data2d.get(sino_id)
            sinograms.append(sino_out)

            astra.algorithm.delete(alg_id)
            astra.data2d.delete(vol_id)
            astra.data2d.delete(sino_id)

        sino_tensor = torch.from_numpy(np.stack(sinograms))[:, None, :, :]
        return sino_tensor.to(device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        vol_geom = ctx.vol_geom
        proj_geom = ctx.proj_geom
        projector_id = ctx.projector_id

        batch_size = grad_output.shape[0]
        device = grad_output.device

        grad_output_np = np.ascontiguousarray(grad_output.detach().cpu().numpy())
        grad_images = []
        
        gpu_id = device.index if device.index is not None else 0
        astra.set_gpu_index(gpu_id)

        for i in range(batch_size):
            sino_id = astra.data2d.create("-sino", proj_geom, grad_output_np[i, 0])
            vol_id = astra.data2d.create("-vol", vol_geom)

            cfg = astra.astra_dict("BP_CUDA")
            cfg["ProjectorId"] = projector_id
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = vol_id

            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)

            vol_out = astra.data2d.get(vol_id)
            grad_images.append(vol_out)

            astra.algorithm.delete(alg_id)
            astra.data2d.delete(vol_id)
            astra.data2d.delete(sino_id)

        grad_image_tensor = torch.from_numpy(np.stack(grad_images))[:, None, :, :]

        # Return gradients for the inputs of `forward` (except for non-tensor arguments)
        return grad_image_tensor.to(device, dtype=torch.float32), None, None, None


class AstraBackProjection(torch.autograd.Function):
    """
    Differentiable Back Projection using ASTRA CUDA.
    Maps a Sinogram [B, 1, Angles, Detectors] to a 2D spatial image [B, 1, H, W].
    """

    @staticmethod
    def forward(ctx, sinogram: torch.Tensor, vol_geom: dict, proj_geom: dict, projector_id: int) -> torch.Tensor:
        ctx.vol_geom = vol_geom
        ctx.proj_geom = proj_geom
        ctx.projector_id = projector_id

        batch_size = sinogram.shape[0]
        device = sinogram.device

        sinogram_np = np.ascontiguousarray(sinogram.detach().cpu().numpy())
        images = []
        
        gpu_id = device.index if device.index is not None else 0
        astra.set_gpu_index(gpu_id)

        for i in range(batch_size):
            sino_id = astra.data2d.create("-sino", proj_geom, sinogram_np[i, 0])
            vol_id = astra.data2d.create("-vol", vol_geom)

            cfg = astra.astra_dict("BP_CUDA")
            cfg["ProjectorId"] = projector_id
            cfg["ProjectionDataId"] = sino_id
            cfg["ReconstructionDataId"] = vol_id

            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)

            vol_out = astra.data2d.get(vol_id)
            images.append(vol_out)

            astra.algorithm.delete(alg_id)
            astra.data2d.delete(vol_id)
            astra.data2d.delete(sino_id)

        image_tensor = torch.from_numpy(np.stack(images))[:, None, :, :]
        return image_tensor.to(device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        vol_geom = ctx.vol_geom
        proj_geom = ctx.proj_geom
        projector_id = ctx.projector_id

        batch_size = grad_output.shape[0]
        device = grad_output.device

        grad_output_np = np.ascontiguousarray(grad_output.detach().cpu().numpy())
        grad_sinograms = []
        
        gpu_id = device.index if device.index is not None else 0
        astra.set_gpu_index(gpu_id)

        for i in range(batch_size):
            vol_id = astra.data2d.create("-vol", vol_geom, grad_output_np[i, 0])
            sino_id = astra.data2d.create("-sino", proj_geom)

            cfg = astra.astra_dict("FP_CUDA")
            cfg["ProjectorId"] = projector_id
            cfg["VolumeDataId"] = vol_id
            cfg["ProjectionDataId"] = sino_id

            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)

            sino_out = astra.data2d.get(sino_id)
            grad_sinograms.append(sino_out)

            astra.algorithm.delete(alg_id)
            astra.data2d.delete(vol_id)
            astra.data2d.delete(sino_id)

        grad_sino_tensor = torch.from_numpy(np.stack(grad_sinograms))[:, None, :, :]

        return grad_sino_tensor.to(device, dtype=torch.float32), None, None, None
