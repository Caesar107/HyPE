import numpy as np
import os

def inspect_demo_file(demo_file):
    if not os.path.exists(demo_file):
        print(f"❌ 文件不存在: {demo_file}")
        return

    print(f"✅ 正在加载: {demo_file}")
    demos = np.load(demo_file, allow_pickle=True)

    print(f"包含字段: {list(demos.keys())}\n")

    for key in demos.files:
        data = demos[key]

        # 输出字段基本信息
        print(f"字段: {key}")
        print(f"  类型: {type(data)}")
        print(f"  dtype: {data.dtype}")
        print(f"  shape: {data.shape}")

        # 打印第一个样本的数据内容（可选）
        print(f"  示例: {data[0]}\n")

if __name__ == "__main__":
    # 官方Humanoid-v3 demo路径
    demo_file = "/home/yche767/Hype/experts/Humanoid-v3/Humanoid-v3_demos.npz"

    inspect_demo_file(demo_file)
