# A Method Based on Multi-source Adaptive for Single Cell Classification


Single-cell ribonucleic acid (RNA) sequencing technology has been successfully used to generate high-resolution cellular maps of human tissues and organs, which has deepened our understanding of cellular heterogeneity in human disease tissues. Cellular annotation is a very critical step in the analysis of single-cell RNA sequencing data. Many typical methods utilize a labeled single-cell reference dataset to annotate the target dataset, but some cell types in the target dataset may not be in the reference dataset. Integrating multiple reference data can better cover the cell types in the target dataset, however, there are batch effects between multiple reference datasets and the target dataset due to differences in sequencing technology and other reasons. To this end, in this paper, we propose a single-cell classification model based on multi-source domain adaptation, which achieves batch elimination by using multiple reference datasets with labeled cell types trained against unlabeled target datasets, respectively. In addition, we use virtual adversarial training to further enhance the robustness of model prediction results to small local perturbations or noise around data points and prevent overfitting. By comparing on multiple single-cell datasets, our method achieves higher cell identification accuracy than the state-of-the-art method. This provides new options and lessons for single-cell identity identification for new sequencing.
       


![model structure](Workflow.png)



# Usage
```bash
python main_pseudo_dann.py --lr 0.5 --epoch 50 --batch_size 64 --name pbmc_ding --cuda 0
```
Note: the detailed parameters instructions please see [MSASC_Train](https://github.com/biomed-AI/MSASC/blob/main/main_pseudo_dann.py)


## System environment
Required package:
- PyTorch >= 1.10
- scanpy >= 1.8
- python >=3.7
- tensorboard
- anndata == 0.7.6

# Datasets

 -  PBMC
 -  pancreas
