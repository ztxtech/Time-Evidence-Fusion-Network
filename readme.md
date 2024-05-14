<div align="center">
  <h2><b> Time Evidence Fusion Network (TEFN): 
    <br/> Multi-source View in Long-Term Time Series Forecasting </b></h2>
</div>

**Repo Status:**

[![GitHub last commit](https://img.shields.io/github/last-commit/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network/activity?ref=master&activity_type=direct_push)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/t/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network/graphs/commit-activity)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network)
[![GitHub Repo stars](https://img.shields.io/github/stars/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network)
[![GitHub forks](https://img.shields.io/github/forks/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network)
[![GitHub watchers](https://img.shields.io/github/watchers/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network)

**Implementation and License:**

[![arxiv](https://img.shields.io/badge/cs.LG-2405.06419-b31b1b?style=flat&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2405.06419)
![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![nVIDIA](https://img.shields.io/badge/nVIDIA-cuda-%2376B900.svg?logo=nVIDIA&logoColor=white)
![Apple](https://img.shields.io/badge/Mac-MPS-%23000000.svg?logo=apple&logoColor=white)

## Updates

🚩 **News** (2024.05.14) Compatible with MPS backend, TEFN can be trained by MacBook Air M1 RAM 8G.

## Overview

This is the official code implementation project for paper **"Time Evidence Fusion Network: Multi-source View in
Long-Term Time Series Forecasting"**. The code implementation refers
to [![GitHub](https://img.shields.io/badge/thuml-Time_Series_Library-%23121011?logo=github&logoColor=white)](https://github.com/thuml/Time-Series-Library).
Thanks very much
for [![GitHub](https://img.shields.io/badge/thuml-Time_Series_Library-%23121011?logo=github&logoColor=white)](https://github.com/thuml/Time-Series-Library)'
s contribution to this project.

![TEFN](/fig/TEFN.png)
The **Time-Evidence Fusion Network (TEFN)** is a groundbreaking deep learning model designed for long-term time series
forecasting. It integrates the principles of information fusion and evidence theory to achieve superior performance in
real-world applications where timely predictions are crucial. TEFN introduces the Basic Probability Assignment (BPA)
Module, leveraging fuzzy theory, and the Time Evidence Fusion Network to enhance prediction accuracy, stability, and
interpretability.

## Key Features

- **Information Fusion Perspective**: TEFN addresses time series forecasting from a unique angle, focusing on the fusion
  of multi-source information to boost prediction accuracy.
  ![Information Fusion Perspective](/fig/ms.png)
- **BPA Module**: At its core, TEFN incorporates a BPA Module that maps diverse information sources to probability
  distributions related to the target outcome. This module exploits the interpretability of evidence theory, using fuzzy
  membership functions to represent uncertainty in predictions.
  ![BPA](/fig/bpa.png)
  ![BPA Diagram](./fig/inver_conv.png)
- **Interpretability**: Due to its roots in fuzzy logic, TEFN provides clear insights into the decision-making process,
  enhancing model explainability.
  ![Channel dimension interpretability](/fig/CBV.png)
  ![Time dimension interpretability](/fig/TBV.png)
- **State-of-the-Art Performance**: TEFN demonstrates competitive results, with prediction errors comparable to leading
  models like PatchTST, while maintaining high efficiency and requiring fewer parameters than complex models such as
  Dlinear.
  ![SOTA](/fig/sota.png)
- **Robustness and Stability**: The model showcases resilience to hyperparameter tuning, exhibiting minimal fluctuations
  even under random selections, ensuring consistent performance across various settings.
  ![Visualization of Robustness](/fig/vr.png)
  ![Variance](/fig/var.png)
- **Efficiency**: With optimized training times and a compact model footprint, TEFN is particularly suitable for
  resource-constrained environments.
  ![Efficiency](/fig/size.png)

## Getting Started

### Requirements

- ![Python](https://img.shields.io/badge/python->3.6-3670A0?logo=python&logoColor=ffdd54) Python >= 3.6
- ![PyTorch](https://img.shields.io/badge/PyTorch->1.7.0-%23EE4C2C.svg?logo=PyTorch&logoColor=white) PyTorch >= 1.7.0
- ![Python](https://img.shields.io/badge/PyPI-3670A0?logo=PyPI&logoColor=ffdd54) Other dependencies listed
  in `requirements.txt`

### Installation

Clone the repository:

```bash
git clone https://github.com/ztxtech/Time-Evidence-Fusion-Network.git
cd Time-Evidence-Fusion-Network
pip install -r requirements.txt
```

### Usage

#### Download Dataset

You can obtain datasets
from [![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing)
or [![Baidu Drive](https://img.shields.io/badge/Baidu-Pan-2932E1?logo=Baidu&logoColor=white)](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy),
Then place the downloaded data in the folder`./dataset`.

#### Load Config

1. Modify the specific configuration file in `./run_config.py`.

```python
config_path = '{your chosen config file path}'
```

2. Run `./run_config.py` directly.

```bash
python run_config.py
```

#### Other Operations

Other related operations refer
to [![GitHub](https://img.shields.io/badge/thuml-Time_Series_Library-%23121011?logo=github&logoColor=white)](https://github.com/thuml/Time-Series-Library).

#### Citation

If you find TEFN useful in your research, please cite our work as per the citation.

```bibtex
@misc{TEFN,
      title={Time Evidence Fusion Network: Multi-source View in Long-Term Time Series Forecasting}, 
      author={Tianxiang Zhan and Yuanpeng He and Zhen Li and Yong Deng},
      year={2024},
      journal={arXiv}
}

```

## Contact

If you have any questions or suggestions, feel free to contact:

- (**Primary**)Tianxiang Zhan [(zhantianxianguestc@hotmail.com)](mailto:zhantianxianguestc@hotmail.com)
  [![Outlook](https://img.shields.io/badge/Tianxiang_Zhan-0078D4?logo=microsoft-outlook&logoColor=white)](mailto:zhantianxianguestc@hotmail.com)
  [![Google Scholar](https://img.shields.io/badge/Tianxiang_Zhan-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com.hk/citations?user=bRYz250AAAAJ)
  [![ResearchGate](https://img.shields.io/badge/Tianxiang_Zhan-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Tianxiang-Zhan)
- Yuanpeng He [(heyuanpeng@stu.pku.edu.cn)](mailto:heyuanpeng@stu.pku.edu.cn)
  [![Outlook](https://img.shields.io/badge/Yuanpeng_He-0078D4?logo=microsoft-outlook&logoColor=white)](mailto:heyuanpeng@stu.pku.edu.cn)
  [![Google Scholar](https://img.shields.io/badge/Yuanpeng_He-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=HaefBCQAAAAJ)
  [![ResearchGate](https://img.shields.io/badge/Yuanpeng_He-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Yuanpeng-He)

Or describe it in Issues.
