import random
from pathlib import Path
import torch
from src_2D.utils.evaluation import build_full_astra_geometries, reconstruct_volume_fbp, plot_qualitative_example
from src_2D.models.SinoSheavesNN.graph_data import create_sinogram_data
from torch_geometric.data import Batch

def generate_snn_example_figure(model, dataset, device, geometry, args, run) -> None:
    model.eval()
    idx = random.randrange(len(dataset))
    incomplete, full, phantom = dataset[idx]
    
    from src_2D.geometry.dbt_geometry_2d import DBTGeometry
    geom = DBTGeometry(
        angles=geometry.full_angles,
        src_radius=geometry.src_radius_mm,
        det_radius=geometry.det_radius_mm,
        det_col_count=geometry.det_col_count,
        det_pixel_size=geometry.det_pixel_size_mm,
    )
    with torch.no_grad():
        inc_sino = incomplete[0].to(device) # [180, 128]
        graph_data = create_sinogram_data(inc_sino, geom).to(device)
        graph_batch = Batch.from_data_list([graph_data])
        
        refined_out = model(graph_batch) # [180, 128]
        refined_out = refined_out.unsqueeze(0).unsqueeze(0) # [1, 1, 180, 128]

    reconstructed_image = None
    try:
        proj_geom, vol_geom = build_full_astra_geometries(geometry, tuple(phantom.shape[1:]))
        reconstructed_image = reconstruct_volume_fbp(
            refined_out.squeeze(0).squeeze(0).cpu().numpy() * 100.0, proj_geom, vol_geom
        )
    except Exception as exc:
        print(f"Skipping volume reconstruction panel ({exc}).")

    save_path = args.figures_dir / "example_after_training.png"
    fig = plot_qualitative_example(
        phantom.squeeze(0), full.squeeze(0), incomplete.squeeze(0), refined_out.squeeze(0).squeeze(0).cpu(), reconstructed_image, geometry, save_path
    )
    print(f"Saved qualitative example to {save_path}")

    if run is not None:
        import wandb
        wandb.log({"example": wandb.Image(fig)})
