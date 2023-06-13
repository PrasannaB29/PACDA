# Official implementation for PaCDA
[[CLVision Workshop - CVPR 2023] Continual Domain Adaptation Through Pruning-Aided Domain-Specific Weight Modulation](https://openaccess.thecvf.com/content/CVPR2023W/CLVision/html/B_Continual_Domain_Adaptation_Through_Pruning-Aided_Domain-Specific_Weight_Modulation_CVPRW_2023_paper.html)

## Commands for running
### General instructions
Within each training .py file, modify lines enclosed within #TODO and the list variable "names" appropriately before each run
### Baseline
#### Source train
```
python image_source.py --trte val --da uda --output ./debug_bl_1/ACPR --dset office-home --max_epoch 100
```
#### Target adaptation
```
python image_target_train.py --dset office-home --output ./debug_bl_1/ACPR --output_src ./debug_bl_1/ACPR
```

### Ours
#### Source train
```
python image_source.py --trte val --da uda --output ./debug_1/ACPR --dset office-home --max_epoch 100
```
#### Source prune and finetune
Set --pf_c to be the desired fraction of params of each layer in source model to be pruned
```
python image_prune_source_finetune.py --trte val --da uda --output ./debug_1/ACPR --dset office-home --pf_c 0.4 --output_src ./debug_1/ACPR
```
#### Target train
```
python image_prune_target_train.py --dset domain-net --output ./debug_1/ACPR --output_src ./debug_1/ACPR
```
#### Target prune and finetune
Set --pf_c to be the desired fraction of params of each layer in train model to be pruned
```
python image_prune_target_finetune.py --pf_c 0.3 --output ./debug_1/ACPR --output_src ./debug_1/ACPR --dset office-home
```
## Acknowledgement
The code is based on [SHOT (ICML, 2020)](https://github.com/tim-learn/SHOT)
## Citation
If you find this work useful, please feel free to cite this work.
```
@InProceedings{B_2023_CVPR,
    author    = {B, Prasanna and Sanyal, Sunandini and Babu, R. Venkatesh},
    title     = {Continual Domain Adaptation Through Pruning-Aided Domain-Specific Weight Modulation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {2456-2462}
}
```
