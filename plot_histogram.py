import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# ================= ✅ YOUR PATH CONFIGURATION =================
LOG_DIR = r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-07_19-13-12"
Z_DATA_PATH = os.path.join(LOG_DIR, "z_data.torch")
INLIERS_DIR = os.path.join(LOG_DIR, "preds_superpoint-lg")
# ==============================================================

def plot_final_histogram():
    print("🚀 Parsing Ground Truth Data...")
    
    # 1. Load and parse the dictionary to get the Top-1 correctness mask
    if not os.path.exists(Z_DATA_PATH):
        print("❌ Error: z_data.torch not found.")
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
            
            # Check if the Top-1 prediction is correct
            if len(true_matches) > 0:
                if isinstance(true_matches, torch.Tensor):
                    hit = (top_pred == true_matches).any().item()
                else:
                    hit = top_pred in true_matches
            else:
                hit = False 
            
            is_correct_mask.append(hit)
            
        print(f"📊 Parsing Complete: Top-1 Correct Queries {sum(is_correct_mask)}, Wrong Queries {len(is_correct_mask) - sum(is_correct_mask)}.")
        
    except Exception as e:
        print(f"❌ Failed to parse z_data: {e}")
        return

    # 2. Read inliers data
    print("Reading Inliers Data (Core Step)...")
    files = sorted([f for f in os.listdir(INLIERS_DIR) if f.endswith(".torch")])
    
    # Align data lengths
    min_len = min(len(files), len(is_correct_mask))
    files = files[:min_len]
    is_correct_mask = is_correct_mask[:min_len]

    correct_inliers = []
    wrong_inliers = []

    for idx, filename in enumerate(files):
        try:
            filepath = os.path.join(INLIERS_DIR, filename)
            data = torch.load(filepath, weights_only=False)
            
            # === 🔥 Final Fix Logic 🔥 ===
            # 'data' is a list containing 20 dictionaries.
            # We iterate through these 20 dictionaries to find the maximum 'num_inliers' value.
            
            max_val = 0
            if isinstance(data, list):
                counts = [x['num_inliers'] for x in data if isinstance(x, dict) and 'num_inliers' in x]
                max_val = max(counts) if counts else 0
            
            # Classification
            if is_correct_mask[idx]:
                correct_inliers.append(max_val)
            else:
                wrong_inliers.append(max_val)
        except Exception as e:
            # print(f"Skipping {filename} due to error: {e}")
            pass

    # 3. Plot the red/green contrast histogram
    print("Generating Plot...")
    plt.figure(figsize=(10, 6))
    
    # Stacked histogram: This should now show the distribution
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
    print(f"✅ Success! Final fixed image saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_final_histogram()