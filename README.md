# PACDA
Official Code for PACDA (CVPR Workshop - 4th CLVision)

## Commands for running
### General instructions
Within each training .py file, modify lines enclosed within #TODO and the list variable "names" appropriately before each run
### Baseline
#### Source train
python image_source.py --trte val --da uda --output ./debug_bl_1/ACPR --dset office-home --max_epoch 100
#### Target adaptation
python image_target_train.py --dset office-home --output ./debug_bl_1/ACPR --output_src ./debug_bl_1/ACPR

## Acknowledgement
The code is based on [SHOT (ICML, 2020)](https://github.com/tim-learn/SHOT)
