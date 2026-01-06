# Visual Place Recognition & Uncertainty Estimation Project

This project evaluates various Visual Place Recognition (VPR) methods and implements a post-hoc **Uncertainty Estimation** module (Section 6.2). It uses geometric verification (Inliers) to predict the confidence of VPR retrievals, enabling robust localization even in challenging cross-domain scenarios (e.g., Day vs. Night).

---

## 🛠 1. Environment Setup (Installation)

### 1.1 Python Environment
Ensure you have **Python 3.10** installed.

```bash
# Create virtual environment
py -3.10 -m venv venv

# Activate it
.\venv\Scripts\activate  # Windows
# source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

### 1.2 Git Submodules & SuperGlue Fix
This project relies on `image-matching-models` and `imatch-toolbox`. Follow these steps to fix submodule paths for proper installation:

```bash
# 1. Update the main submodule
git submodule update --init

# 2. Update nested submodules
cd image-matching-models
git submodule update --init matching/third_party/LightGlue
git submodule update --init matching/third_party/imatch-toolbox

# 3. Apply SuperGlue Fix
cd matching\third_party\imatch-toolbox

## Action: Open .gitmodules in the imatch-toolbox directory and replace its content with the following to ensure valid URLs:

Ini, TOML

[submodule "third_party/SuperGluePretrainedNetwork"]
	path = third_party/superglue
	url = [https://github.com/magicleap/SuperGluePretrainedNetwork.git](https://github.com/magicleap/SuperGluePretrainedNetwork.git)
[submodule "third_party/r2d2"]
	path = third_party/r2d2
	url = [https://github.com/naver/r2d2.git](https://github.com/naver/r2d2.git)
	ignore = untracked
[submodule "third_party/d2net"]
	path = third_party/d2net
	url = [https://github.com/mihaidusmanu/d2-net.git](https://github.com/mihaidusmanu/d2-net.git)
[submodule "third_party/patch2pix"]
	path = third_party/patch2pix
	url = [https://github.com/GrumpyZhou/patch2pix.git](https://github.com/GrumpyZhou/patch2pix.git)

Finalize Setup:
# Run inside imatch-toolbox folder
git submodule update --init
