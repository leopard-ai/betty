# DSRNA-Betty
This an unofficial code of [CVPR'2021](https://openaccess.thecvf.com/CVPR2021): ***"DSRNA: Differentiable Search of Robust Neural Architectures"*** [paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Hosseini_DSRNA_Differentiable_Search_of_Robust_Neural_Architectures_CVPR_2021_paper.pdf)   with Betty implementation to search for robust neural architectures.

## Architecture Search
You can change the robustness metric using ```--loss_type``` and the model search ```--darts_type```:

```python dsrna_search.py```
## Results
We present the CIFAR-10 image classification results under PGD-100 in the below table.
|                      | Test Acc. | 
|----------------------|-----------|
| DSRNA (original)     | 59.47%    | 
| DSRNA (ours, step=1) | 59.71%    |
