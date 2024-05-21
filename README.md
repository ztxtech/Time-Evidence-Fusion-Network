<div align="center">
  <h2><b> Time Evidence Fusion Network (TEFN): 
    <br/> Multi-source View in Long-Term Time Series Forecasting </b></h2>
</div>

**Repo Status:**

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
[![Visits Badge](https://badges.pufler.dev/visits/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network)
[![GitHub last commit](https://img.shields.io/github/last-commit/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network/activity?ref=master&activity_type=direct_push)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/t/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network/graphs/commit-activity)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network)
[![GitHub Repo stars](https://img.shields.io/github/stars/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network)
[![GitHub forks](https://img.shields.io/github/forks/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network)
[![GitHub watchers](https://img.shields.io/github/watchers/ztxtech/Time-Evidence-Fusion-Network)](https://github.com/ztxtech/Time-Evidence-Fusion-Network)

**Implementation:**

[![arxiv](https://img.shields.io/badge/cs.LG-2405.06419-b31b1b?style=flat&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2405.06419)
[![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![nVIDIA](https://img.shields.io/badge/nVIDIA-cuda-%2376B900.svg?logo=nVIDIA&logoColor=white)](https://pytorch.org/docs/2.1/cuda.html)
[![Apple](https://img.shields.io/badge/Mac-MPS-%23000000.svg?logo=apple&logoColor=white)](https://pytorch.org/docs/2.1/mps.html)

## Updates

🚩 **News** (2024.05.14) Compatible with MPS backend, TEFN can be trained by [![Apple](https://img.shields.io/badge/MacBook_Air_2020-M1_8G-%23000000.svg?logo=apple&logoColor=white)](https://support.apple.com/zh-cn/111883).


## Overview

This is the official code implementation project for paper **"Time Evidence Fusion Network: Multi-source View in
Long-Term Time Series Forecasting"**. The code implementation refers
to [![GitHub](https://img.shields.io/badge/thuml-Time_Series_Library-%23121011?logo=github&logoColor=white)](https://github.com/thuml/Time-Series-Library).
Thanks very much
for [![GitHub](https://img.shields.io/badge/thuml-Time_Series_Library-%23121011?logo=github&logoColor=white)](https://github.com/thuml/Time-Series-Library)'s contribution to this project.

![TEFN](https://cdn.jsdelivr.net/gh/ztxtech/Time-Evidence-Fusion-Network/fig/TEFN.png)
The **Time Evidence Fusion Network (TEFN)** is a groundbreaking deep learning model designed for long-term time series
forecasting. It integrates the principles of information fusion and evidence theory to achieve superior performance in
real-world applications where timely predictions are crucial. TEFN introduces the Basic Probability Assignment (BPA)
Module, leveraging fuzzy theory, and the Time Evidence Fusion Network to enhance prediction accuracy, stability, and
interpretability.

## Key Features

- **Information Fusion Perspective**: TEFN addresses time series forecasting from a unique angle, focusing on the fusion
  of multi-source information to boost prediction accuracy.
  ![Information Fusion Perspective](https://cdn.jsdelivr.net/gh/ztxtech/Time-Evidence-Fusion-Network/fig/ms.png)
- **BPA Module**: At its core, TEFN incorporates a BPA Module that maps diverse information sources to probability
  distributions related to the target outcome. This module exploits the interpretability of evidence theory, using fuzzy
  membership functions to represent uncertainty in predictions.
  ![BPA](https://cdn.jsdelivr.net/gh/ztxtech/Time-Evidence-Fusion-Network/fig/bpa.png)
  ![BPA Diagram](https://cdn.jsdelivr.net/gh/ztxtech/Time-Evidence-Fusion-Network/fig/inver_conv.png)
- **Interpretability**: Due to its roots in fuzzy logic, TEFN provides clear insights into the decision-making process,
  enhancing model explainability.
  ![Channel dimension interpretability](https://cdn.jsdelivr.net/gh/ztxtech/Time-Evidence-Fusion-Network/fig/CBV.png)
  ![Time dimension interpretability](https://cdn.jsdelivr.net/gh/ztxtech/Time-Evidence-Fusion-Network/fig/TBV.png)
- **State-of-the-Art Performance**: TEFN demonstrates competitive results, with prediction errors comparable to leading
  models like PatchTST, while maintaining high efficiency and requiring fewer parameters than complex models such as
  Dlinear.
  ![SOTA](https://cdn.jsdelivr.net/gh/ztxtech/Time-Evidence-Fusion-Network/fig/sota.png)
- **Robustness and Stability**: The model showcases resilience to hyperparameter tuning, exhibiting minimal fluctuations
  even under random selections, ensuring consistent performance across various settings.
  ![Visualization of Robustness](https://cdn.jsdelivr.net/gh/ztxtech/Time-Evidence-Fusion-Network/fig/vr.png)
  ![Variance](https://cdn.jsdelivr.net/gh/ztxtech/Time-Evidence-Fusion-Network/fig/var.png)
- **Efficiency**: With optimized training times and a compact model footprint, TEFN is particularly suitable for
  resource-constrained environments.
  ![Efficiency](https://cdn.jsdelivr.net/gh/ztxtech/Time-Evidence-Fusion-Network/fig/size.png)

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

#### Switching Running Devices

1. Find required configuration file `*.json` in `./configs`.
2. Modify `*.json` file.

``` 
{
  # ...
  # Nvidia CUDA Device {0}
  # 'gpu': 0
  # Apple MPS Device
  # 'gpu': 'mps'
  # ...
}
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

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- [Time Series Library ![GitHub](https://img.shields.io/badge/thuml-Time_Series_Library-%23121011?logo=github&logoColor=white)](https://github.com/thuml/Time-Series-Library)'
- [TSFpaper ![GitHub](https://img.shields.io/badge/ddz16-TSFpaper-%23121011?logo=github&logoColor=white)](https://github.com/ddz16/TSFpaper)
- [Time-Series-Forecasting-and-Deep-Learning ![GitHub](https://img.shields.io/badge/DaoSword-Time--Series--Forecasting--and--Deep--Learning-%23121011?logo=github&logoColor=white)](https://github.com/DaoSword/Time-Series-Forecasting-and-Deep-Learning)
- [awesome-opensource ![GitHub](https://img.shields.io/badge/gitroomhq-awesome--opensource-%23121011?logo=github&logoColor=white)](https://github.com/gitroomhq/awesome-opensource)



## Contact

If you have any questions or suggestions, feel free to contact:

- (**Primary**) Tianxiang Zhan [(ztxtech@std.uestc.edu.cn)](mailto:ztxtech@std.uestc.edu.cn)
  [![Outlook](https://img.shields.io/badge/Tianxiang_Zhan-0078D4?logo=microsoft-outlook&logoColor=white)](mailto:ztxtech@std.uestc.edu.cn)
  [![Google Scholar](https://img.shields.io/badge/Tianxiang_Zhan-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com.hk/citations?user=bRYz250AAAAJ)
  [![ResearchGate](https://img.shields.io/badge/Tianxiang_Zhan-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Tianxiang-Zhan)
- Yuanpeng He [(heyuanpeng@stu.pku.edu.cn)](mailto:heyuanpeng@stu.pku.edu.cn)
  [![Outlook](https://img.shields.io/badge/Yuanpeng_He-0078D4?logo=microsoft-outlook&logoColor=white)](mailto:heyuanpeng@stu.pku.edu.cn)
  [![Google Scholar](https://img.shields.io/badge/Yuanpeng_He-4285F4?logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?user=HaefBCQAAAAJ)
  [![ResearchGate](https://img.shields.io/badge/Yuanpeng_He-00CCBB?logo=ResearchGate&logoColor=white)](https://www.researchgate.net/profile/Yuanpeng-He)

Or describe it in Issues.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ztxtech/Time-Evidence-Fusion-Network&type=Date)](https://star-history.com/#ztxtech/Time-Evidence-Fusion-Network&Date)
