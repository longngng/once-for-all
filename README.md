# Once for All on ECG Dataset
Also refer to README from [here](https://github.com/mit-han-lab/once-for-all/blob/master/README.md) for more informations in the repo

## Prepare the dataset
Using the ecg-tinyml repo, generate the three .csv files, or use the prepared mitbih.zip and extract it under `$HOME/dataset/mitbih` as below
```
├── dataset
    ├──mitbih
        ├── mitbih_test.csv
        ├── mitbih_train.csv
        ├── mitbih_val.csv
```

If you want to use the dataset placed elsewhere, please change the directory specified at `ofa/imagenet_classification/data_providers/mitbih.py`

## Prepare the environment
Use the same environment as from ecg-tinyml repo

## To train OFA Networks
```bash
conda activate tinyml-ecg

# The following commands can be run separately as it takes significant time for each stage 
python train_ofa_net.py --task kernel
python train_ofa_net.py --task depth --phase 1
python train_ofa_net.py --task depth --phase 2
python train_ofa_net.py --task expand --phase 1
python train_ofa_net.py --task expand --phase 2
```

## Sample the subnetworks
After the **last stage** training, there will be a .tar file under `exp/kernel_depth2kernel_depth_width/phase2/checkpoint/checkpoint.pth.tar`. Use this file to perform sampling the subnetworks.

To get random networks and benchmark its performance on validation set, run `python sample_randomly.py` script

After picking the best networks from the accuracy-complexity tradeoff graph, update the config file at `configs/sample_subnetwork.yaml` and run `python sample_with_config.py` to sample a particular subnetworks. The script will export .pth and .onnx models. After that, use the scripts in ecg-tinyml repo to further convert it to TensorFlow format