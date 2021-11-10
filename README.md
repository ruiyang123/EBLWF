# EBLWF

## Requirements
'''bash
$ pip3 install -r requirements.txt
'''

## Data Preparation

PACS: you can get the PACS dataset from [https://github.com/MachineLearning2020/Homework3-PACS](https://github.com/MachineLearning2020/Homework3-PACS)

Cifar100: In our experiments we use the .png version of cifar100.
You can download the dataset by using the following commands :

```bash
$ pip install cifar2png
$ cifar2png cifar100superclass path/to/cifar100png
```
Before you start the experiments, it is neccesaire to place the two datasets under the directory "./datasets/"

Generally the directories for PACS looks like: 

```bash
./datasets/PACS
├── art_painting
│   ├── dog
...
...
```

```bash
For cifar100 : 

./datasets/cifar100
├── train
│   ├── fish
│   │   ├── flatfish
...
├── test
...
```

## Run experiments 

For PACS you can run  
'''bash
$ sh Benchmark_PACS.sh
'''

For cifar100 you can run  
'''bash
$ sh Benchmark_cifar.sh
'''




