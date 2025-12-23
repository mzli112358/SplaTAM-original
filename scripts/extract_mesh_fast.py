"""
Fast Mesh Extraction from IsoGS checkpoint using Tile-based/Block-based algorithm.

This script uses a push-based approach: instead of querying all Gaussians for each voxel,
it assigns Gaussians to spatial blocks and only computes density for relevant Gaussians.
"""

import argparse
import os
import re
import sys
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage import measure
import trimesh

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from utils.slam_external import build_rotation


def parse_args():
    parser = argparse.ArgumentParser(description="Fast mesh extraction from IsoGS checkpoint")
    parser.add_argument("config", type=str, help="Path to experiment config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Specific checkpoint file. If None, auto-selects latest.")
    parser.add_argument("--output", type=str, default=None,
                       help="Output mesh file path (default: mesh_fast.ply)")
    parser.add_argument("--voxel-size", type=float, default=0.02,
                       help="Voxel size in meters (default: 0.02)")
    parser.add_argument("--iso-level", type=float, default=1.0,
                       help="Iso-surface threshold level (default: 1.0)")
    parser.add_argument("--padding", type=float, default=0.5,
                       help="Padding around bounding box in meters (default: 0.5)")
    parser.add_argument("--block-size", type=int, default=16,
                       help="Block size for tiling (default: 16, meaning 16x16x16 voxels per block)")
    parser.add_argument("--truncate-sigma", type=float, default=3.0,
                       help="Truncation distance in sigma (default: 3.0)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--no-cleaning", action="store_true",
                       help="Disable mesh cleaning (keep full mesh instead of only largest component)")
    parser.add_argument("--no-show", action="store_true",
                       help="Do not open 3D viewer after mesh export")
    return parser.parse_args()


def load_checkpoint(config_path, checkpoint_path=None):
    """Load experiment config and checkpoint parameters."""
    # Load config
    experiment = SourceFileLoader(
        os.path.basename(config_path), config_path
    ).load_module()
    config = experiment.config
    
    # Determine checkpoint path
    result_dir = os.path.join(config['workdir'], config['run_name'])
    
    checkpoint_frame = None
    if checkpoint_path is None:
        # Smart checkpoint selection
        params_npz_path = os.path.join(result_dir, "params.npz")
        if os.path.exists(params_npz_path):
            checkpoint_path = params_npz_path
            print(f"✓ Found final params file: {checkpoint_path}")
        else:
            # Find all checkpoint files
            pattern = re.compile(r'^params(\d+)\.npz$')
            checkpoint_files = []
            if os.path.exists(result_dir):
                for filename in os.listdir(result_dir):
                    match = pattern.match(filename)
                    if match:
                        checkpoint_num = int(match.group(1))
                        checkpoint_files.append((checkpoint_num, filename))
            
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: x[0], reverse=True)
                latest_checkpoint = checkpoint_files[0]
                checkpoint_frame = latest_checkpoint[0]
                checkpoint_path = os.path.join(result_dir, latest_checkpoint[1])
                print(f"✓ Auto-selected latest checkpoint: {latest_checkpoint[1]} (frame {latest_checkpoint[0]})")
                if len(checkpoint_files) > 1:
                    print(f"  (Found {len(checkpoint_files)} checkpoints total)")
            else:
                raise FileNotFoundError(
                    f"No checkpoint found in {result_dir}\n"
                    f"Expected files: params.npz or params*.npz"
                )
    else:
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(result_dir, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"✓ Using specified checkpoint: {checkpoint_path}")
        # Try to parse frame number from filename (e.g., params800.npz)
        basename = os.path.basename(checkpoint_path)
        m = re.match(r'^params(\d+)\.npz$', basename)
        if m:
            checkpoint_frame = int(m.group(1))
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    params = dict(np.load(checkpoint_path, allow_pickle=True))
    
    return config, params, result_dir, checkpoint_path, checkpoint_frame


def build_inverse_covariances(params, device, min_scale_limit=0.0):
    """Build inverse covariance matrices for all Gaussians.
    
    Args:
        params: Dictionary containing Gaussian parameters
        device: Device to use for computation
        min_scale_limit: Minimum scale limit to prevent pancaking artifacts (default: 0.0, disabled)
    """
    # Get parameters
    means = params['means3D']
    log_scales = params['log_scales']
    unnorm_rots = params['unnorm_rotations']
    
    # Convert to tensors on device
    if not isinstance(means, torch.Tensor):
        means = torch.tensor(means, device=device, dtype=torch.float32)
    else:
        means = means.to(device).float()
    
    if not isinstance(log_scales, torch.Tensor):
        log_scales = torch.tensor(log_scales, device=device, dtype=torch.float32)
    else:
        log_scales = log_scales.to(device).float()
    
    if not isinstance(unnorm_rots, torch.Tensor):
        unnorm_rots = torch.tensor(unnorm_rots, device=device, dtype=torch.float32)
    else:
        unnorm_rots = unnorm_rots.to(device).float()
    
    # Handle isotropic case
    if log_scales.shape[1] == 1:
        log_scales = log_scales.repeat(1, 3)
    
    # Get scales and opacities
    scales = torch.exp(log_scales).clamp(min=1e-5)  # [N, 3]
    
    # Apply minimum scale limit to prevent pancaking artifacts
    # This ensures each Gaussian covers at least one sampling point
    if min_scale_limit > 0:
        print(f"Clamping scales to minimum: {min_scale_limit}")
        scales = torch.clamp(scales, min=min_scale_limit)
    
    # Handle opacities
    logit_opacities = params['logit_opacities']
    if not isinstance(logit_opacities, torch.Tensor):
        logit_opacities = torch.tensor(logit_opacities, device=device, dtype=torch.float32)
    else:
        logit_opacities = logit_opacities.to(device).float()
    opacities = torch.sigmoid(logit_opacities).squeeze(-1)  # [N]
    
    # Build rotation matrices
    quats = F.normalize(unnorm_rots, dim=1)  # [N, 4]
    R = build_rotation(quats)  # [N, 3, 3]
    
    # Build inverse covariance matrices: Σ^{-1} = R S^{-2} R^T
    S_inv_sq = 1.0 / (scales ** 2 + 1e-8)  # [N, 3]
    S_inv_sq_diag = torch.diag_embed(S_inv_sq)  # [N, 3, 3]
    R_S_inv_sq = torch.bmm(R, S_inv_sq_diag)  # [N, 3, 3]
    inverse_covariances = torch.bmm(R_S_inv_sq, R.transpose(1, 2))  # [N, 3, 3]
    
    print(f"Loaded {len(means)} Gaussians")
    print(f"  Means shape: {means.shape}")
    print(f"  Scales shape: {scales.shape}")
    print(f"  Opacities range: [{opacities.min().item():.4f}, {opacities.max().item():.4f}]")
    
    return means, inverse_covariances, opacities, scales


def compute_bounding_box(means, padding=0.5):
    """Compute bounding box from Gaussian means."""
    means_np = means.cpu().numpy() if isinstance(means, torch.Tensor) else means
    min_bounds = means_np.min(axis=0) - padding
    max_bounds = means_np.max(axis=0) + padding
    print(f"Bounding box: min={min_bounds}, max={max_bounds}")
    return min_bounds, max_bounds


def compute_density_tiled(voxel_coords, dims, means, inverse_covariances, opacities, scales,
                          min_bounds, max_bounds, voxel_size, block_size=16, 
                          truncate_sigma=3.0, device='cuda'):
    """
    Compute density using tile-based/block-based approach (Push strategy).
    
    Algorithm:
    1. Divide voxel grid into blocks (e.g., 16x16x16)
    2. For each Gaussian, compute its AABB (based on 3σ)
    3. Assign Gaussians to blocks they overlap with
    4. For each block, only compute density for relevant Gaussians
    """
    voxel_coords_tensor = torch.tensor(voxel_coords, device=device, dtype=torch.float32)
    num_voxels = len(voxel_coords)
    num_gaussians = len(means)
    
    # Reshape voxel coordinates to 3D grid structure
    voxel_grid_3d = voxel_coords_tensor.reshape(dims[0], dims[1], dims[2], 3)
    
    # Compute max scales for each Gaussian (for AABB)
    max_scales = scales.max(dim=1).values  # [N]
    max_truncate_dist = truncate_sigma * max_scales  # [N]
    
    # Initialize density array
    densities = torch.zeros(num_voxels, device=device, dtype=torch.float32)
    
    # Calculate block dimensions
    block_dims = np.ceil(np.array(dims) / block_size).astype(int)
    num_blocks = np.prod(block_dims)
    
    print(f"Grid dimensions: {dims}")
    print(f"Block dimensions: {block_dims} (block_size={block_size})")
    print(f"Total blocks: {num_blocks}")
    print(f"Processing using tile-based algorithm...")
    
    # Pre-compute block bounds
    block_bounds_list = []
    for bx in range(block_dims[0]):
        for by in range(block_dims[1]):
            for bz in range(block_dims[2]):
                # Block voxel ranges
                x_start = bx * block_size
                x_end = min((bx + 1) * block_size, dims[0])
                y_start = by * block_size
                y_end = min((by + 1) * block_size, dims[1])
                z_start = bz * block_size
                z_end = min((bz + 1) * block_size, dims[2])
                
                # Block world space bounds
                block_voxel_coords = voxel_grid_3d[x_start:x_end, y_start:y_end, z_start:z_end]
                block_min = block_voxel_coords.reshape(-1, 3).min(dim=0).values
                block_max = block_voxel_coords.reshape(-1, 3).max(dim=0).values
                
                # Expand by max truncate distance to include nearby Gaussians
                block_center = (block_min + block_max) / 2
                block_extent = (block_max - block_min) / 2
                # Add some padding for safety
                block_min_expanded = block_min - max_truncate_dist.max() * 1.5
                block_max_expanded = block_max + max_truncate_dist.max() * 1.5
                
                block_bounds_list.append({
                    'voxel_range': ((x_start, x_end), (y_start, y_end), (z_start, z_end)),
                    'world_bounds': (block_min, block_max),
                    'world_bounds_expanded': (block_min_expanded, block_max_expanded),
                })
    
    # Process each block
    block_idx = 0
    for bx in tqdm(range(block_dims[0]), desc="Processing blocks (X)"):
        for by in range(block_dims[1]):
            for bz in range(block_dims[2]):
                block_info = block_bounds_list[block_idx]
                block_idx += 1
                
                # Get block voxel coordinates
                (x_start, x_end), (y_start, y_end), (z_start, z_end) = block_info['voxel_range']
                block_voxel_coords = voxel_grid_3d[x_start:x_end, y_start:y_end, z_start:z_end]
                block_voxel_coords_flat = block_voxel_coords.reshape(-1, 3)  # [B, 3]
                num_block_voxels = len(block_voxel_coords_flat)
                
                if num_block_voxels == 0:
                    continue
                
                # Find Gaussians that overlap with this block
                # Use expanded bounds for culling
                block_min_exp, block_max_exp = block_info['world_bounds_expanded']
                
                # Check which Gaussians are within expanded bounds
                # means: [N, 3], block_min_exp: [3], block_max_exp: [3]
                in_bounds = (
                    (means >= block_min_exp.unsqueeze(0)).all(dim=1) &
                    (means <= block_max_exp.unsqueeze(0)).all(dim=1)
                )
                
                # Further culling: check distance from block center
                block_center = (block_min_exp + block_max_exp) / 2
                dists_to_block_center = torch.norm(means - block_center.unsqueeze(0), dim=1)  # [N]
                # Only consider Gaussians within max_truncate_dist + block_diagonal/2
                block_diagonal = torch.norm(block_max_exp - block_min_exp)
                cull_distance = max_truncate_dist + block_diagonal / 2
                in_range = dists_to_block_center <= cull_distance  # [N]
                
                # Combine both culling criteria
                relevant_mask = in_bounds & in_range
                relevant_indices = torch.where(relevant_mask)[0]
                
                if len(relevant_indices) == 0:
                    # No Gaussians in this block, density is zero
                    continue
                
                # Extract relevant Gaussians
                relevant_means = means[relevant_indices]  # [K, 3]
                relevant_inv_covs = inverse_covariances[relevant_indices]  # [K, 3, 3]
                relevant_opacities = opacities[relevant_indices]  # [K]
                relevant_max_truncate = max_truncate_dist[relevant_indices]  # [K]
                
                # Internal voxel batching to prevent OOM
                voxel_batch_size = 512
                num_relevant_gaussians = len(relevant_indices)
                
                # Initialize block densities
                block_densities = torch.zeros(num_block_voxels, device=device, dtype=torch.float32)
                
                # Process voxels in batches
                for i in range(0, num_block_voxels, voxel_batch_size):
                    batch_end = min(i + voxel_batch_size, num_block_voxels)
                    batch_voxels = block_voxel_coords_flat[i:batch_end]  # [batch_size, 3]
                    batch_size = len(batch_voxels)
                    
                    # Compute density for batch voxels
                    # batch_voxels: [batch_size, 3]
                    # relevant_means: [K, 3]
                    
                    # Compute distances: [batch_size, K, 3]
                    deltas = batch_voxels.unsqueeze(1) - relevant_means.unsqueeze(0)  # [batch_size, K, 3]
                    dists = torch.norm(deltas, dim=2)  # [batch_size, K]
                    
                    # Apply truncation mask: [batch_size, K]
                    truncate_mask = dists < relevant_max_truncate.unsqueeze(0)  # [batch_size, K]
                    
                    # Compute quadratic form: delta^T @ inv_cov @ delta
                    # deltas: [batch_size, K, 3] -> [batch_size*K, 3]
                    deltas_flat = deltas.reshape(-1, 3)  # [batch_size*K, 3]
                    
                    # relevant_inv_covs: [K, 3, 3] -> [batch_size, K, 3, 3]
                    inv_covs_expanded = relevant_inv_covs.unsqueeze(0).expand(batch_size, -1, -1, -1)
                    inv_covs_flat = inv_covs_expanded.reshape(-1, 3, 3)  # [batch_size*K, 3, 3]
                    
                    # Compute quadratic form
                    deltas_flat_expanded = deltas_flat.unsqueeze(1)  # [batch_size*K, 1, 3]
                    inv_cov_delta = torch.bmm(deltas_flat_expanded, inv_covs_flat)  # [batch_size*K, 1, 3]
                    quad_form_flat = torch.bmm(inv_cov_delta, deltas_flat.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [batch_size*K]
                    
                    # Reshape back: [batch_size*K] -> [batch_size, K]
                    quad_form = quad_form_flat.reshape(batch_size, num_relevant_gaussians)
                    
                    # Compute exponential
                    exp_term = torch.exp(-0.5 * quad_form)  # [batch_size, K]
                    
                    # Multiply by opacities
                    relevant_opacities_expanded = relevant_opacities.unsqueeze(0).expand(batch_size, -1)  # [batch_size, K]
                    density_contrib = relevant_opacities_expanded * exp_term  # [batch_size, K]
                    
                    # Apply truncation mask
                    density_contrib = density_contrib * truncate_mask.float()
                    
                    # Sum over Gaussians: [batch_size, K] -> [batch_size]
                    batch_densities = density_contrib.sum(dim=1)  # [batch_size]
                    
                    # Store batch densities in the corresponding positions
                    block_densities[i:batch_end] = batch_densities
                
                # Map block densities back to global voxel indices
                # Calculate global linear indices for this block
                block_linear_indices = []
                idx_in_block = 0
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        for z in range(z_start, z_end):
                            # Convert 3D index to linear index (C-order: z varies fastest)
                            linear_idx = x * dims[1] * dims[2] + y * dims[2] + z
                            block_linear_indices.append(linear_idx)
                            idx_in_block += 1
                
                # Ensure we have the right number of indices
                assert len(block_linear_indices) == num_block_voxels, \
                    f"Mismatch: {len(block_linear_indices)} indices vs {num_block_voxels} voxels"
                
                block_linear_indices = torch.tensor(block_linear_indices, device=device, dtype=torch.long)
                densities[block_linear_indices] = block_densities
                
                # Clear cache periodically
                if block_idx % 100 == 0:
                    torch.cuda.empty_cache()
    
    return densities.cpu().numpy()


def create_voxel_grid(min_bounds, max_bounds, voxel_size):
    """Create voxel grid coordinates."""
    # Calculate grid dimensions
    size = max_bounds - min_bounds
    dims = np.ceil(size / voxel_size).astype(int)
    print(f"Voxel grid dimensions: {dims} (voxel_size={voxel_size})")
    print(f"Total voxels: {np.prod(dims):,}")
    
    # Create coordinate grids
    x = np.linspace(min_bounds[0], max_bounds[0], dims[0])
    y = np.linspace(min_bounds[1], max_bounds[1], dims[1])
    z = np.linspace(min_bounds[2], max_bounds[2], dims[2])
    
    # Calculate actual voxel size
    actual_voxel_size = np.array([
        (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else voxel_size,
        (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else voxel_size,
        (z[-1] - z[0]) / (len(z) - 1) if len(z) > 1 else voxel_size
    ])
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    voxel_coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    return voxel_coords, dims, actual_voxel_size, min_bounds


def extract_mesh(density_grid, dims, iso_level=1.0, voxel_spacing=None, origin=None):
    """Extract mesh using Marching Cubes."""
    print(f"Extracting mesh at iso-level {iso_level}...")
    
    # Reshape density grid to 3D
    density_3d = density_grid.reshape(dims)
    
    # Marching Cubes
    if voxel_spacing is not None and origin is not None:
        vertices, faces, normals, values = measure.marching_cubes(
            density_3d,
            level=iso_level,
            spacing=voxel_spacing,
            gradient_direction='descent'
        )
        vertices = vertices + origin
    else:
        vertices, faces, normals, values = measure.marching_cubes(
            density_3d,
            level=iso_level,
            spacing=(1.0, 1.0, 1.0),
            gradient_direction='descent'
        )
    
    print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    return vertices, faces, normals


def clean_mesh(vertices, faces):
    """Clean mesh by keeping only largest connected component."""
    print("Cleaning mesh...")
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    components = mesh.split(only_watertight=False)
    print(f"Found {len(components)} connected components")
    
    if len(components) > 1:
        largest_idx = np.argmax([c.vertices.shape[0] for c in components])
        mesh = components[largest_idx]
        print(f"Keeping largest component with {len(mesh.vertices)} vertices")
    
    # Clean mesh: merge duplicate vertices and remove unreferenced ones
    mesh.merge_vertices()  # Merge duplicate vertices
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    
    print(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    return mesh


def main():
    args = parse_args()
    
    # Load checkpoint
    config, params, result_dir, checkpoint_path, checkpoint_frame = load_checkpoint(args.config, args.checkpoint)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build inverse covariances with minimum scale limit to prevent pancaking artifacts
    # Set minimum scale to half voxel size (Nyquist sampling theorem)
    min_scale_limit = args.voxel_size * 0.5
    means, inverse_covariances, opacities, scales = build_inverse_covariances(
        params, device, min_scale_limit=min_scale_limit
    )
    
    # Compute bounding box
    min_bounds, max_bounds = compute_bounding_box(means, padding=args.padding)
    
    # Create voxel grid
    voxel_coords, dims, voxel_spacing, origin = create_voxel_grid(
        min_bounds, max_bounds, args.voxel_size
    )
    
    # Compute density values using tile-based algorithm
    print("\nComputing density values using fast tile-based algorithm...")
    density_values = compute_density_tiled(
        voxel_coords, dims, means, inverse_covariances, opacities, scales,
        min_bounds, max_bounds, args.voxel_size,
        block_size=args.block_size,
        truncate_sigma=args.truncate_sigma,
        device=device
    )
    
    # Print density statistics
    print(f"\nDensity statistics:")
    print(f"  Min: {density_values.min():.4f}")
    print(f"  Max: {density_values.max():.4f}")
    print(f"  Mean: {density_values.mean():.4f}")
    print(f"  Std: {density_values.std():.4f}")
    
    # Extract mesh
    vertices, faces, normals = extract_mesh(
        density_values, dims,
        iso_level=args.iso_level,
        voxel_spacing=voxel_spacing,
        origin=origin
    )

    # Optionally clean mesh
    if not args.no_cleaning:
        mesh = clean_mesh(vertices, faces)
    else:
        print("Skipping mesh cleaning, keeping full mesh with all components.")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    
    # Save mesh
    # 如果用户没有显式给输出文件名，则根据 checkpoint 帧号自动命名：
    #   mesh_thickened_{frame}.ply
    # 若无法解析出帧号，则回退到原来的 mesh_fast.ply
    if args.output is None:
        if checkpoint_frame is not None:
            base_name = f"mesh_thickened_{checkpoint_frame}"
        else:
            base_name = "mesh_fast"
        output_path = os.path.join(result_dir, f"{base_name}.ply")
    else:
        output_path = args.output if os.path.isabs(args.output) else os.path.join(result_dir, args.output)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    mesh.export(output_path)
    print(f"\nMesh saved to: {output_path}")
    
    # Print statistics
    print("\nMesh Statistics:")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces: {len(mesh.faces):,}")
    print(f"  Bounds: {mesh.bounds}")
    print(f"  Volume: {mesh.volume:.6f}")
    
    # Auto-export OBJ next to PLY (same name & directory)
    file_dir = os.path.dirname(output_path)
    if file_dir:
        obj_path = os.path.join(file_dir, f"{base_name}.obj")
        stl_path = os.path.join(file_dir, f"{base_name}.stl")
    else:
        obj_path = f"{base_name}.obj"
        stl_path = f"{base_name}.stl"

    print(f"\nExporting mesh to OBJ: {obj_path}")
    mesh.export(obj_path)
    print(f"✓ Successfully exported OBJ to: {obj_path}")
    print("  You can open it with Blender, MeshLab, CloudCompare, etc.")

    # Auto-export STL next to PLY (same name & directory)
    print(f"\nExporting mesh to STL: {stl_path}")
    mesh.export(stl_path)
    print(f"✓ Successfully exported STL to: {stl_path}")
    print("  You can open it with Blender, MeshLab, CloudCompare, etc.")

    # Also export a TXT log with the same命名规范，记录本次导出关键信息和调用命令
    if file_dir:
        txt_path = os.path.join(file_dir, f"{base_name}.txt")
    else:
        txt_path = f"{base_name}.txt"

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            # 还原命令行（近似）：在前面加上 python 方便复制
            cmd_str = "python " + " ".join(sys.argv)
            f.write(f"{cmd_str}\n\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            if checkpoint_frame is not None:
                f.write(f"Checkpoint frame: {checkpoint_frame}\n")
            f.write(f"Voxel size: {args.voxel_size}\n")
            f.write(f"Iso level: {args.iso_level}\n")
            f.write(f"Block size: {args.block_size}\n")
            f.write(f"No cleaning: {args.no_cleaning}\n")
            f.write(f"Output PLY: {output_path}\n")
            f.write(f"Output OBJ: {obj_path}\n")
            f.write(f"Output STL: {stl_path}\n")
            f.write(f"Vertices: {len(mesh.vertices)}\n")
            f.write(f"Faces: {len(mesh.faces)}\n")
        print(f"✓ Exported log TXT to: {txt_path}")
    except Exception as e:
        print(f"[Warning] Failed to write TXT log file: {e}")

    # Optionally open interactive 3D viewer
    if args.no_show:
        print("\nSkipping 3D viewer (--no-show set).")
    else:
        print("\nOpening 3D viewer...")
        print("Controls:")
        print("  Left mouse drag: Rotate")
        print("  Mouse wheel: Zoom")
        print("  Middle/Right mouse drag: Pan")
        print("  'w' key: Toggle wireframe")
        print("  'a' key: Toggle axes")
        print("\nClose the window to exit.")
        mesh.show()


if __name__ == "__main__":
    main()

