import argparse
import os
import re
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation


def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    
    # Check if camera parameters exist in the file
    if 'org_width' in params and 'org_height' in params and 'w2c' in params and 'intrinsics' in params:
        # Use existing camera parameters
        org_width = params['org_width']
        org_height = params['org_height']
        w2c = params['w2c']
        intrinsics = params['intrinsics']
        k = intrinsics[:3, :3]
        
        # Scale intrinsics to match the visualization resolution
        k[0, :] *= cfg['viz_w'] / org_width
        k[1, :] *= cfg['viz_h'] / org_height
    else:
        # Camera parameters not in file, compute from cam_unnorm_rots and cam_trans
        print("Warning: Camera parameters not found in .npz file, using defaults and computing from camera poses...")
        
        # Default values for Replica dataset
        org_width = 1200
        org_height = 680
        # Default intrinsics for Replica (fx, fy typically ~600 for 1200x680 resolution)
        fx = fy = 600.0
        cx = org_width / 2.0
        cy = org_height / 2.0
        
        # Compute first frame w2c from cam_unnorm_rots and cam_trans
        if 'cam_unnorm_rots' in params and 'cam_trans' in params:
            cam_rots = torch.tensor(params['cam_unnorm_rots']).cuda().float()
            cam_trans = torch.tensor(params['cam_trans']).cuda().float()
            
            # Handle different shapes: (1, 4, T) or (4, T) or (4,)
            if len(cam_rots.shape) == 3:
                # Shape: (batch, 4, timesteps) - get first batch and first timestep
                cam_rot_1d = cam_rots[0, :, 0]  # Shape: (4,)
                cam_tran = cam_trans[0, :, 0]
            elif len(cam_rots.shape) == 2:
                # Shape: (4, timesteps) - get first timestep
                cam_rot_1d = cam_rots[:, 0]  # Shape: (4,)
                cam_tran = cam_trans[:, 0]
            else:
                # Shape: (4,) - single rotation
                cam_rot_1d = cam_rots  # Shape: (4,)
                cam_tran = cam_trans
            
            # Normalize and convert to 2D for build_rotation (expects (N, 4))
            cam_rot_1d = F.normalize(cam_rot_1d, dim=0)
            cam_rot_2d = cam_rot_1d.unsqueeze(0)  # Shape: (1, 4)
            
            # build_rotation expects (N, 4) and returns (N, 3, 3)
            rot_matrix = build_rotation(cam_rot_2d)  # Shape: (1, 3, 3)
            rot_matrix = rot_matrix[0]  # Shape: (3, 3)
            
            w2c = torch.eye(4).cuda().float()
            w2c[:3, :3] = rot_matrix
            w2c[:3, 3] = cam_tran
            w2c = w2c.cpu().numpy()
        else:
            # Use identity matrix as fallback
            print("Warning: cam_unnorm_rots and cam_trans not found, using identity camera pose")
            w2c = np.eye(4)
        
        # Create intrinsics matrix
        k = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        
        # Scale intrinsics to match the visualization resolution
        k[0, :] *= cfg['viz_w'] / org_width
        k[1, :] *= cfg['viz_h'] / org_height
    
    return w2c, k


def load_scene_data(scene_path):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}
    params = all_params

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot_raw = params['cam_unnorm_rots'][..., t_i]  # Could be (4,) or (1, 4) or (batch, 4)
        cam_tran_raw = params['cam_trans'][..., t_i]  # Could be (3,) or (1, 3) or (batch, 3)
        
        # Ensure cam_rot is 1D, then normalize
        if len(cam_rot_raw.shape) > 1:
            cam_rot_1d = cam_rot_raw.flatten()[:4]  # Take first 4 elements
        else:
            cam_rot_1d = cam_rot_raw
        
        # Ensure cam_tran is 1D
        if len(cam_tran_raw.shape) > 1:
            cam_tran = cam_tran_raw.flatten()[:3]  # Take first 3 elements
        else:
            cam_tran = cam_tran_raw
        
        cam_rot_1d = F.normalize(cam_rot_1d, dim=0)
        cam_rot_2d = cam_rot_1d.unsqueeze(0)  # Shape: (1, 4) for build_rotation
        
        # build_rotation expects (N, 4) and returns (N, 3, 3)
        rot_matrix = build_rotation(cam_rot_2d)  # Shape: (1, 3, 3)
        rot_matrix = rot_matrix[0]  # Shape: (3, 3)
        
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = rot_matrix
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())
    
    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    return params, all_w2cs


def get_rendervars(params, w2c, curr_timestep):
    # Check if timestep field exists for filtering Gaussians
    if 'timestep' in params:
        params_timesteps = params['timestep']
        selected_params_idx = params_timesteps <= curr_timestep
    else:
        # If no timestep field, show all Gaussians
        selected_params_idx = torch.ones(params['means3D'].shape[0], dtype=torch.bool, device='cuda')
    
    keys = [k for k in params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices', 'timestep']]
    selected_params = deepcopy(params)
    for k in keys:
        selected_params[k] = selected_params[k][selected_params_idx]
    if selected_params['log_scales'].shape[-1]  == 1:
        log_scales = torch.tile(selected_params['log_scales'], (1, 3))
    else:
        log_scales = selected_params['log_scales']
    w2c = torch.tensor(w2c).cuda().float()
    rendervar = {
        'means3D': selected_params['means3D'],
        'colors_precomp': selected_params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
    }
    depth_rendervar = {
        'means3D': selected_params['means3D'],
        'colors_precomp': get_depth_and_silhouette(selected_params['means3D'], w2c),
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
    }
    return rendervar, depth_rendervar


def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def render(w2c, k, timestep_data, timestep_depth_data, cfg):
    with torch.no_grad():
        cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'])
        white_bg_cam = Camera(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered
        )
        im, _, depth, = Renderer(raster_settings=white_bg_cam)(**timestep_data)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**timestep_depth_data)
        differentiable_depth = depth_sil[0, :, :].unsqueeze(0)
        sil = depth_sil[1, :, :].unsqueeze(0)
        return im, depth, sil


def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    
    # Colorize point cloud
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def visualize(scene_path, cfg):
    # Load Scene Data
    first_frame_w2c, k = load_camera(cfg, scene_path)

    params, all_w2cs = load_scene_data(scene_path)
    print(params['means3D'].shape)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']), 
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    scene_data, scene_depth_data = get_rendervars(params, first_frame_w2c, curr_timestep=0)
    im, depth, sil = render(first_frame_w2c, k, scene_data, scene_depth_data, cfg)
    init_pts, init_cols = rgbd2pcd(im, depth, first_frame_w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    # Initialize Estimated Camera Frustums
    frustum_size = 0.045
    num_t = len(all_w2cs)
    cam_centers = []
    cam_colormap = plt.get_cmap('cool')
    norm_factor = 0.5
    total_num_lines = num_t - 1
    line_colormap = plt.get_cmap('cool')
    
    # Initialize View Control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    first_view_w2c = first_frame_w2c
    first_view_w2c[:3, 3] = first_view_w2c[:3, 3] + np.array([0, 0, 0.5])
    cparams.extrinsic = first_view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    # Rendering of Online Reconstruction
    start_time = time.time()
    num_timesteps = num_t
    viz_start = True
    curr_timestep = 0
    while curr_timestep < (num_timesteps-1) or not cfg['enter_interactive_post_online']:
        passed_time = time.time() - start_time
        passed_frames = passed_time * cfg['viz_fps']
        curr_timestep = int(passed_frames % num_timesteps)
        if not viz_start:
            if curr_timestep == prev_timestep:
                continue

        # Update Camera Frustum
        if curr_timestep == 0:
            cam_centers = []
            if not viz_start:
                vis.remove_geometry(prev_lines)
        if not viz_start:
            vis.remove_geometry(prev_frustum)
        new_frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[curr_timestep], frustum_size)
        new_frustum.paint_uniform_color(np.array(cam_colormap(curr_timestep * norm_factor / num_t)[:3]))
        vis.add_geometry(new_frustum)
        prev_frustum = new_frustum
        cam_centers.append(np.linalg.inv(all_w2cs[curr_timestep])[:3, 3])
        
        # Update Camera Trajectory
        if len(cam_centers) > 1 and curr_timestep > 0:
            num_lines = [1]
            cols = []
            for line_t in range(curr_timestep):
                cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
            cols = np.array(cols)
            all_cols = [cols]
            out_pts = [np.array(cam_centers)]
            linesets = make_lineset(out_pts, all_cols, num_lines)
            lines = o3d.geometry.LineSet()
            lines.points = linesets[0].points
            lines.colors = linesets[0].colors
            lines.lines = linesets[0].lines
            vis.add_geometry(lines)
            prev_lines = lines
        elif not viz_start:
            vis.remove_geometry(prev_lines)

        # Get Current View Camera
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        view_w2c = cam_params.extrinsic
        view_w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])
        cam_params.extrinsic = view_w2c
        view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        scene_data, scene_depth_data = get_rendervars(params, view_w2c, curr_timestep=curr_timestep)
        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, sil = render(view_w2c, k, scene_data, scene_depth_data, cfg)
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, view_w2c, k, cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()
        prev_timestep = curr_timestep
        viz_start = False

    # Enter Interactive Mode once all frames have been visualized
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg)
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()
    
    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize online 3D Gaussian Splatting reconstruction")
    parser.add_argument("input", type=str, help="Path to experiment file (.py) or .npz file")

    args = parser.parse_args()

    # Check if input is a .npz file (direct file specification)
    if args.input.endswith('.npz') and os.path.isfile(args.input):
        scene_path = os.path.abspath(args.input)
        print(f"Using specified .npz file: {scene_path}")
        
        # Use default visualization configuration
        viz_cfg = dict(
            render_mode='color',  # ['color', 'depth' or 'centers']
            offset_first_viz_cam=True,  # Offsets the view camera back by 0.5 units along the view direction
            show_sil=False,  # Show Silhouette instead of RGB
            visualize_cams=True,  # Visualize Camera Frustums and Trajectory
            viz_w=600, viz_h=340,
            viz_near=0.01, viz_far=100.0,
            view_scale=2,
            viz_fps=5,
            enter_interactive_post_online=True,  # Enter Interactive Mode after Online Recon Viz
        )
        
        seed_everything(seed=42)  # Use default seed when loading file directly
    else:
        # Original logic: load experiment configuration file
        experiment = SourceFileLoader(
            os.path.basename(args.input), args.input
        ).load_module()

        seed_everything(seed=experiment.config["seed"])

        if "scene_path" not in experiment.config:
            results_dir = os.path.join(
                experiment.config["workdir"], experiment.config["run_name"]
            )
            
            # Check if params.npz exists (final result)
            params_npz_path = os.path.join(results_dir, "params.npz")
            
            if os.path.exists(params_npz_path):
                scene_path = params_npz_path
                print(f"Found final params file: {scene_path}")
            else:
                # Find the latest checkpoint file (params{数字}.npz)
                pattern = re.compile(r'^params(\d+)\.npz$')
                checkpoint_files = []
                
                if os.path.exists(results_dir):
                    for filename in os.listdir(results_dir):
                        match = pattern.match(filename)
                        if match:
                            checkpoint_num = int(match.group(1))
                            checkpoint_files.append((checkpoint_num, filename))
                
                if checkpoint_files:
                    # Sort by checkpoint number and get the latest
                    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
                    latest_checkpoint = checkpoint_files[0]
                    scene_path = os.path.join(results_dir, latest_checkpoint[1])
                    print(f"Found latest checkpoint file: {scene_path} (frame {latest_checkpoint[0]})")
                else:
                    raise FileNotFoundError(f"No params file found in {results_dir}. Please check if the experiment has been run.")
        else:
            scene_path = experiment.config["scene_path"]
        viz_cfg = experiment.config["viz"]

    # Visualize Online Reconstruction
    visualize(scene_path, viz_cfg)
