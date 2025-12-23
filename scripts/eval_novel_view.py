import argparse
import os
import random
import sys
import shutil
import re
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
from utils.common_utils import seed_everything
from utils.eval_helpers import eval, eval_nvs


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def find_latest_checkpoint(results_dir):
    """
    自动查找最新的检查点文件。
    优先使用 params.npz（最终文件），如果不存在则查找最新的 params{num}.npz
    
    返回: (checkpoint_path, checkpoint_frame_num)
    - checkpoint_path: 检查点文件路径
    - checkpoint_frame_num: 实际训练到的帧数（从文件名提取），如果是 params.npz 则返回 None
    """
    # 首先检查 params.npz（最终文件）
    params_npz_path = os.path.join(results_dir, "params.npz")
    if os.path.exists(params_npz_path):
        print(f"✓ 找到最终参数文件: {params_npz_path}")
        return params_npz_path, None
    
    # 查找所有检查点文件
    pattern = re.compile(r'^params(\d+)\.npz$')
    checkpoint_files = []
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            match = pattern.match(filename)
            if match:
                checkpoint_num = int(match.group(1))
                checkpoint_files.append((checkpoint_num, filename))
    
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        latest_checkpoint = checkpoint_files[0]
        checkpoint_path = os.path.join(results_dir, latest_checkpoint[1])
        checkpoint_frame_num = latest_checkpoint[0]
        print(f"✓ 自动选择最新检查点: {latest_checkpoint[1]} (帧 {checkpoint_frame_num})")
        if len(checkpoint_files) > 1:
            print(f"  (共找到 {len(checkpoint_files)} 个检查点)")
        return checkpoint_path, checkpoint_frame_num
    else:
        raise FileNotFoundError(
            f"在 {results_dir} 中未找到检查点文件\n"
            f"期望的文件: params.npz 或 params*.npz"
        )


def load_scene_data(scene_path):
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"场景参数文件不存在: {scene_path}")
    params = dict(np.load(scene_path, allow_pickle=True))
    params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
    return params


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="评估新视角合成或训练集评估")
    
    parser.add_argument("experiment", type=str, help="配置文件路径")
    parser.add_argument("--train-split", action="store_true", 
                       help="使用训练集评估模式 (Train Split Eval)，默认使用新视角合成评估 (Novel View Synthesis)")
    parser.add_argument("--nvs", action="store_true", 
                       help="使用新视角合成评估模式 (Novel View Synthesis Eval)，这是默认模式")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    config = experiment.config

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    device = torch.device(config["primary_device"])

    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    
    # 命令行参数优先于配置文件设置
    if args.train_split:
        dataset_config["use_train_split"] = True
        print("=" * 60)
        print("评估模式: 训练集评估 (Train Split Eval)")
        print("=" * 60)
    elif args.nvs:
        dataset_config["use_train_split"] = False
        print("=" * 60)
        print("评估模式: 新视角合成评估 (Novel View Synthesis Eval)")
        print("=" * 60)
    elif "use_train_split" not in dataset_config:
        # 如果命令行未指定且配置文件中也没有，默认使用新视角合成评估
        dataset_config["use_train_split"] = False
        print("=" * 60)
        print("评估模式: 新视角合成评估 (Novel View Synthesis Eval) [默认]")
        print("=" * 60)
    else:
        # 使用配置文件中的设置
        mode = "训练集评估" if dataset_config["use_train_split"] else "新视角合成评估"
        print("=" * 60)
        print(f"评估模式: {mode} (来自配置文件)")
        print("=" * 60)
    # Poses are relative to the first training frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # 确定场景参数文件路径和实际训练帧数
    checkpoint_frame_num = None
    if 'scene_path' in config:
        scene_path = config['scene_path']
        # 如果是相对路径，先尝试相对于当前工作目录，如果不存在则尝试相对于结果目录
        if not os.path.isabs(scene_path):
            # 先尝试原始路径（相对于项目根目录）
            if not os.path.exists(scene_path):
                # 如果不存在，尝试相对于结果目录
                scene_path_alt = os.path.join(results_dir, os.path.basename(scene_path))
                if os.path.exists(scene_path_alt):
                    scene_path = scene_path_alt
                    print(f"使用相对于结果目录的路径: {scene_path}")
        
        # 尝试从文件名提取帧数
        basename = os.path.basename(scene_path)
        match = re.match(r'^params(\d+)\.npz$', basename)
        if match:
            checkpoint_frame_num = int(match.group(1))
    else:
        # 如果配置中没有指定 scene_path，自动查找最新检查点
        print("配置中未指定 scene_path，自动查找最新检查点...")
        scene_path, checkpoint_frame_num = find_latest_checkpoint(results_dir)
    
    # 检查文件是否存在，如果不存在则尝试自动查找最新检查点
    if not os.path.exists(scene_path):
        print(f"警告: 指定的场景路径不存在: {scene_path}")
        print("尝试自动查找最新检查点...")
        scene_path, checkpoint_frame_num = find_latest_checkpoint(results_dir)
    
    print(f"使用场景参数文件: {scene_path}")
    params = load_scene_data(scene_path)
    
    # 获取检查点中保存的实际帧数（模型训练到的帧数）
    # 优先使用从文件名提取的帧数，如果没有则使用 cam_unnorm_rots 的维度
    if checkpoint_frame_num is not None:
        checkpoint_num_frames = checkpoint_frame_num
        print(f"从检查点文件名提取的训练帧数: {checkpoint_num_frames}")
    else:
        # 对于 params.npz（最终文件），使用 cam_unnorm_rots 的维度
        # 但要注意这可能包含所有帧，需要进一步判断
        cam_frames = params['cam_unnorm_rots'].shape[-1]
        checkpoint_num_frames = cam_frames
        print(f"检查点中 cam_unnorm_rots 的帧数: {checkpoint_num_frames} (可能是数据集总帧数)")
        print(f"注意: 如果这是最终文件，实际训练帧数可能小于此值")
    
    print(f"数据集总帧数: {num_frames}")
    
    # 根据评估模式限制评估范围
    if dataset_config['use_train_split']:
        # 训练集评估：只评估训练时见过的帧（检查点中的帧数）
        eval_num_frames = min(num_frames, checkpoint_num_frames)
        print(f"训练集评估模式：评估前 {eval_num_frames} 帧（训练时见过的帧）")
        if num_frames > checkpoint_num_frames:
            print(f"注意: 数据集有 {num_frames} 帧，但只评估前 {checkpoint_num_frames} 帧（训练时见过的帧）")
        eval_dir = os.path.join(results_dir, "eval_train")
        wandb_name = config['wandb']['name'] + "_Train_Split"
    else:
        # 新视角合成评估：评估所有帧，测试模型的泛化能力
        # 注意：NVS 模式跳过第一帧（训练帧），所以实际评估的是 1 到 num_frames
        eval_num_frames = num_frames
        print(f"新视角合成评估模式：评估所有 {eval_num_frames} 帧（泛化评估，跳过第0帧训练帧）")
        if num_frames > checkpoint_num_frames:
            print(f"注意: 数据集有 {num_frames} 帧，检查点训练到 {checkpoint_num_frames} 帧")
            print(f"将评估所有 {num_frames} 帧以测试模型泛化能力")
        eval_dir = os.path.join(results_dir, "eval_nvs")
        wandb_name = config['wandb']['name'] + "_NVS_Split"
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=wandb_name,
                               config=config)

    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            if dataset_config['use_train_split']:
                eval(dataset, params, eval_num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], save_frames=True)
            else:
                eval_nvs(dataset, params, eval_num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], save_frames=True)
        else:
            if dataset_config['use_train_split']:
                eval(dataset, params, eval_num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], save_frames=True)
            else:
                eval_nvs(dataset, params, eval_num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                    mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                    eval_every=config['eval_every'], save_frames=True)
    
    # Close WandB
    if config['use_wandb']:
        wandb_run.finish()
