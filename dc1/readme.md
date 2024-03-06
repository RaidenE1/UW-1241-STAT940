# Report for Data Challenge 1

> Student Name: Jinhe Zhang
> Student Number: 21068423
> Kaggle Name: j423zhan
> Leaderboard Score: 0.6532

## 1. Structure

```
├── info.txt						# a txt file contains student number, kaggle name and leader board score
├── src
│   ├── data						# data folder
│   		├── test  			# test data
│   		│   └── test
│   		└── train 			# train data
│       		└── train
├── model	
│		├── resnet.py				# cnn model
│   └── mydataset.py 		# customized dataset
├── utils
│   ├── get_mean_std.py # compute mean and std for whole dataset
│   ├── load_labels 		# load ground truth lables from `train_labels.csv`
│   ├── sep_dataset.py 	# divide dataset into train set and validation set
│   └── logger.py 			# customized logger
├── requirements.txt		# required packages
└── readme.md       		# current file     

```

## 2. Setup

The code is based on:

- Python: 3.10
- CUDA: 12.1
- GPU: Nvidia RTX-4090
- PyTorch: 2.2.0
- Torch vision: 0.17.0
- other packages can be found in requirements.txt
- Torch seed is mannually set to 3407

## 3. How to run

```bash
python main.py --model resnet18 --epochs 200 --batch-size 128 --lr 0.1 --opt SGD
```



## 4. Brief introduction of the code

The source code can be divided to such few steps:

1. Preparation：
  1. Compute mean&std of the dataset
  2. Load model
    1. The model used in this challenge is [Resnet](https://arxiv.org/abs/1512.03385)
2. Load dataset
3. Data augmentation
   1. random crop
   2. horizontal flip
   3. Transform
4. Train
   1. Load inputs data and labels to gpu
   2. Compute the outputs
   3. Compute loss
   4. Backwords
5. Tricks:
   1. SGD optimizer
   2. Cosine learning rate scheduler
   3. A little warmup, no more than 1% epochs

