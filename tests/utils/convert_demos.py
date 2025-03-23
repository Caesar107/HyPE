import torch
import numpy as np
import os
from pathlib import Path

def convert_walker2d_demos():
    """
    将Walker2d的demo文件转换为项目所需的格式
    
    转换逻辑:
    1. 使用torch.load加载原始的.npy文件
    2. 将数据重新组织为项目所需的字典结构
    3. 保存为新的npz文件
    """
    # 设置输入和输出文件路径
    input_file = "/data/home/yche767/Hype/experts/Walker2d-v3/transitions_Walker2d-v3.npy"
    output_dir = "/data/home/yche767/Hype/experts/Walker2d-v3/"
    output_file = f"{output_dir}Walker2d-v3_demos.npz"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在从 {input_file} 加载专家数据...")
    # 使用torch.load加载原始数据
    data = torch.load(input_file)
    
    print(f"数据加载成功, 包含 {data.obs.shape[0]} 条转换记录")
    print(f"观察空间维度: {data.obs.shape[1]}")
    
    # 提取dones数组以便重用
    dones_np = data.dones.numpy() if torch.is_tensor(data.dones) else data.dones
    
    # 全零填充rewards，使用float64类型，IRL算法不依赖于原始reward
    rewards = np.zeros_like(dones_np, dtype=np.float64)
    
    # 创建符合项目要求的字典结构
    demos_dict = {
        "observations": data.obs.numpy() if torch.is_tensor(data.obs) else data.obs,
        "actions": data.acts.numpy() if torch.is_tensor(data.acts) else data.acts,
        "next_observations": data.next_obs.numpy() if torch.is_tensor(data.next_obs) else data.next_obs,
        "rewards": rewards,  # 使用float64全零rewards，因为IRL不依赖reward信息
        "terminals": dones_np,
        
        # 添加timeouts字段，全零填充以避免超时报错
        "timeouts": np.zeros_like(dones_np),
        
        # 从观察中提取qpos和qvel (假设前8维是qpos，接下来9维是qvel)
        "qpos": (data.obs.numpy() if torch.is_tensor(data.obs) else data.obs)[:, :8],
        "qvel": (data.obs.numpy() if torch.is_tensor(data.obs) else data.obs)[:, 8:17],
    }
    
    # 保存为npz格式
    print(f"正在将转换后的数据保存到 {output_file}...")
    np.savez(output_file, **demos_dict)
    print("转换完成!")
    
    # 输出数据形状信息，便于验证
    print("\n转换后的数据形状:")
    for key, value in demos_dict.items():
        print(f"  {key}: {value.shape}")

if __name__ == "__main__":
    convert_walker2d_demos()
