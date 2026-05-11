# AMSW: Adaptive Multi-view Semantic Weighting for Chinese Harmful Meme Detection

This repository contains the official implementation for the paper: **"AMSW: Adaptive Multi-view Semantic Weighting for Chinese Harmful Meme Detection"**. **AMSW** is a framework for Chinese harmful meme detection that addresses a key limitation of prior work: the tendency to apply a single, uniform fusion strategy to all memes regardless of how their harmful cues are actually distributed. In Chinese harmful memes, some samples convey harm primarily through imagery, others through implicit linguistic patterns, and still others only through the interplay of image and text. A fixed pipeline cannot optimally handle this heterogeneity. Our proposed AMSW framework overcomes this limitation by integrating three distinct modules to achieve a comprehensive understanding and adaptive detection of harmful memes: 

- **CIR** (Cross-modal Interaction Representation) — captures fine-grained bidirectional interactions between image patches and text tokens via symmetric cross-attention, producing interaction-enriched representations.
- **MSI** (Multi-view Semantic Interpretation) — employs a Multimodal Large Language Model  to generate complementary semantic descriptions of each meme from three distinct views.
- **SAA** (Signal-guided Adaptive Aggregation) — constructs an instance-specific control signal from the original multimodal representations and uses it to estimate weights over the three semantic branches, so the final decision representation reflects whichever view is most informative for the current meme.

## Overview

![AMSW](.\AMSW.png)

## File

```
AMSW/
├── models/
│   ├── __init__.py
│   ├── amsw.py          # Full AMSW model (entry point)
│   ├── cir.py           # Cross-modal Interaction Representation
│   ├── msi.py           # Multi-view Semantic Interpretation + prompt 
│   └── saa.py           # Signal-guided Adaptive Aggregation
├── data/
│   ├── __init__.py
│   └── dataset.py       # ToxiCNMMDataset + DataLoader factory
├── utils/
│   ├── __init__.py
│   ├── metrics.py       # Precision / Recall / macro-F1 per task
│   └── logger.py        # Console + file logging
├── configs/
│   └── default.yaml     # Hyperparameters matching the paper
├── scripts/
│   └── generate_interpretations.py  # Offline MLLM inference
├── train.py             # Training entry point
├── evaluate.py          # Evaluation + prediction export
├── requirements.txt
└── README.md
```

---

## Dataset

This study utilized the public ToxiCN MM dataset, the first large-scale benchmark for Chinese harmful meme detection. The dataset is released with the paper "**Towards Comprehensive Detection of Chinese Harmful Memes**" at NeurIPS 2024.

ToxiCN MM dataset contains 12,000 Chinese memes annotated for two progressive tasks: (1)Harmful Meme Detection; (2)Harmful Type Identification

**Obtaining the data.** Please refer to the ToxiCN MM repository for access instructions and license terms:  https://github.com/DUT-lujunyu/ToxiCN_MM

---

## Citation

If you find our work helpful to your research, or use this code in your research, please cite our paper: 

```bibtex
@inproceedings{Que2026amsw,
  title     = {AMSW: Adaptive Multi-view Semantic Weighting for Chinese Harmful Meme Detection},
  author    = {Que, Henghua and He, Yuxiang and Wang, Haizhou}
  booktitle = {Advanced Intelligent Computing Technology and Applications: 22nd International Conference, ICIC 2026, Toronto, Canada, July 22-26, 2026, Proceedings},
  year      = {2026},
}
```

Since our experiments are conducted on the ToxiCN MM dataset, we strongly recommend that you also cite the original paper.

```bibtex
@inproceedings{lu2024toxicnmm,
  title     = {Towards Comprehensive Detection of Chinese Harmful Memes},
  author    = {Lu, Junyu and Xu, Bo and Zhang, Xiaokun and Wang, Hongbo and Zhu, Haohao and Zhang, Dongyu and Yang, Liang and Lin, Hongfei},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  pages     = {13302--13320},
  year      = {2024},
}
```

---

## Contact

For questions or collaboration, please reach out to:

- **Haizhou Wang** ([whzh.nc@scu.edu.cn](mailto:whzh.nc@scu.edu.cn))
- **Henghua Que** ([quehenghua@stu.scu.edu.cn](mailto:quehenghua@stu.scu.edu.cn))

------

This project is released under the **MIT License**.
