# CANDA

## Setup 
Required packages can be installed as follows:

```
conda create -n canda python=3.10
conda activate canda
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Run Experiments
### Prepare dataset
We use three publicly available datasets: SEED, SEED-IV, SEED-V. All of them can be requested at https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/. 

Feel free to write your own code to split each dataset to training and test sets. The common way to separate each dataset is specified in the table:

|    Dataset   | num_classes |  Test Scheme  |
|    :----:    |    :----:   |     :----:    |
| SEED      | 3 emotions       | train:test = 3:2  |
| SEED-IV    | 4 emotions        | train:test = 2:1   |
| SEED-V    | 5 emotions        | 3-fold cross-validation    |

We provide function splitTrainTest in data_utils.py for splitting training and test datasets. However, you need to organize the original data from matlab format to .npy and put each session of each subject in separate folders like this:

```
Dataset
├── Subject1
│   ├── 1
│   ├── 2
│   └── 3
├── Subject2
│   ├── 1
│   ├── 2
│   └── 3
├── ...
```

### Train model
Default values of parameters have been set in parsing.py. Default setting trains and test the model on SEED dataset. You can run with default with following simple comment:

```
python train_seed.py
```

Feel free to adjust the parameters as you needed:

```
python train_seed.py --dataset seediv \
    --lr 1e-4 \
    --wd 1e-3 \
    --alpha 0.8 \
    --epochs 100 
```

