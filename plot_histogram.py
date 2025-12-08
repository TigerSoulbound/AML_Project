import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# ================= ✅ 你的路径配置 =================
LOG_DIR = r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-07_19-13-12"
Z_DATA_PATH = os.path.join(LOG_DIR, "z_data.torch")
INLIERS_DIR = os.path.join(LOG_DIR, "preds_superpoint-lg")
# ==================================================

def plot_final_histogram():
    print("🚀 正在解析 Ground Truth 数据...")
    
    # 1. 加载并解析字典，得到 Top-1 正确性 Mask
    if not os.path.exists(Z_DATA_PATH):
        print("❌ 找不到 z_data.torch")
        return
        
    try:
        z_data = torch.load(Z_DATA_PATH, weights_only=False)
        predictions = z_data['predictions']          
        positives = z_data['positives_per_query']    
        
        is_correct_mask = []
        for i in range(len(predictions)):
            top_pred = predictions[i][0]
            true_matches = positives[i]
            
            if isinstance(top_pred, torch.Tensor):
                top_pred = top_pred.item()
            
            # 判断 Top-1 是否命中
            if len(true_matches) > 0:
                if isinstance(true_matches, torch.Tensor):
                    hit = (top_pred == true_matches).any().item()
                else:
                    hit = top_pred in true_matches
            else:
                hit = False 
            
            is_correct_mask.append(hit)
            
        print(f"📊 解析完成: Top-1 正确查询 {sum(is_correct_mask)} 个，错误查询 {len(is_correct_mask) - sum(is_correct_mask)} 个。")
        
    except Exception as e:
        print(f"❌ 解析 z_data 失败: {e}")
        return

    # 2. 读取内点数据
    print("正在读取内点数据 (核心步骤)...")
    files = sorted([f for f in os.listdir(INLIERS_DIR) if f.endswith(".torch")])
    
    # 对齐
    min_len = min(len(files), len(is_correct_mask))
    files = files[:min_len]
    is_correct_mask = is_correct_mask[:min_len]

    correct_inliers = []
    wrong_inliers = []

    for idx, filename in enumerate(files):
        try:
            filepath = os.path.join(INLIERS_DIR, filename)
            data = torch.load(filepath, weights_only=False)
            
            # === 🔥 最终修复逻辑 🔥 ===
            # data 是一个包含 20 个字典的列表。
            # 我们要遍历这 20 个字典，找到 'num_inliers' 最大的那个值。
            
            max_val = 0
            if isinstance(data, list):
                counts = [x['num_inliers'] for x in data if isinstance(x, dict) and 'num_inliers' in x]
                max_val = max(counts) if counts else 0
            
            # 分类
            if is_correct_mask[idx]:
                correct_inliers.append(max_val)
            else:
                wrong_inliers.append(max_val)
        except Exception as e:
            # print(f"Skipping {filename} due to error: {e}")
            pass

    # 3. 画红绿对比图
    print("正在绘图...")
    plt.figure(figsize=(10, 6))
    
    # 堆叠直方图：这次应该能看到分布了
    plt.hist([correct_inliers, wrong_inliers], bins=50, range=(0, 200), stacked=True,
             color=['#4CAF50', '#F44336'], label=['Correct Queries', 'Wrong Queries'],
             edgecolor='black', alpha=0.8)
    
    plt.title('Inliers Distribution: Correct vs Wrong (SuperPoint+LightGlue)')
    plt.xlabel('Number of Inliers (Confidence)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(LOG_DIR, "inliers_split_histogram_final_v3.png")
    plt.savefig(save_path)
    print(f"✅ 恭喜！最终修复版图片已保存到: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_final_histogram()