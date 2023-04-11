# PACDA
Official Code for PACDA (CVPR Workshop - 4th CLVision)

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
