## Model Training

### Prepare dataset
See [prepare dataset](https://github.com/KLASS-gait-recognitionn/gait_training/blob/main/OpenGait-for-local/docs/0.prepare_dataset.md).

### Get trained model
- Option 1:
    ```
    python misc/download_pretrained_model.py
    ```
- Option 2: Go to the [release page](https://github.com/ShiqiYu/OpenGait/releases/), then download the model file and uncompress it to [output](output).

### Train
Train a model by
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./config/baseline/baseline.yaml --phase train
```
-  `python -m torch.distributed.launch` [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) launch instruction.
-  `--nproc_per_node` The number of gpus to use, and it must equal the length of `CUDA_VISIBLE_DEVICES`.
-  `--cfgs` The path to config file.
-  `--phase` Specified as `train`.
<!-- - `--iter` You can specify a number of iterations or use `restore_hint` in the config file and resume training from there. -->
- `--log_to_file` If specified, the terminal log will be written on disk simultaneously. 

You can run commands in [train.sh](https://github.com/KLASS-gait-recognitionn/gait_training/blob/main/OpenGait-for-local/train.sh) for training different models. Training is done on a Linux environment. Commands and code would have to be changed manually according to your specific OS environment.

#### Models We Created
To run the models we created, a Gait Ensemble of GaitSet + GaitPart and a Hybrid model, use these commands:

**Gait Ensemble**
```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./config/gaitens.yaml --phase train
```
**Hybrid**
```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./config/hybrid.yaml --phase train
```

### Test
Evaluate the trained model by
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./config/baseline/baseline.yaml --phase test
```
`--phase` : Specified as `test` ,`--iter` : Specify a iteration checkpoint

**Tip**: Other arguments are the same as train phase.

You can run commands in [test.sh](https://github.com/KLASS-gait-recognitionn/gait_training/blob/main/OpenGait-for-local/test.sh) for testing different models.

### Customize
1. Read the [detailed config](https://github.com/KLASS-gait-recognitionn/gait_training/blob/main/OpenGait-for-local/docs/1.detailed_config.md) to figure out the usage of needed setting items;
2. See [how to create your model](https://github.com/KLASS-gait-recognitionn/gait_training/blob/main/OpenGait-for-local/docs/2.how_to_create_your_model.md);
3. There are some advanced usages, refer to [advanced usages](https://github.com/KLASS-gait-recognitionn/gait_training/blob/main/OpenGait-for-local/docs/3.advanced_usages.md), please.

### Warning
- In `DDP` mode, zombie processes may be generated when the program terminates abnormally. You can use this command [sh misc/clean_process.sh](https://github.com/KLASS-gait-recognitionn/gait_training/blob/main/OpenGait-for-local/misc/clean_process.sh) to clear them.