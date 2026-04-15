# Safenet-like branch

## How to run

From here, run :

`python -m safenet_pipeline_ceed`

or

`python -m safenet_pipeline_china`

depending on the dataset you want to train on

### Parameters 

Add `--help` to get the list of all parameters that influence the run

Example : 

`python -m safenet_pipeline_ceed --wandb-run-name high_lr_more_epochs_2 --lr 1.75e-4 --epochs 30`
