import pickle
from data import get_multitask_experiment
import pandas as pd





factors={"cifar_DIL":5,"cifar_smoothDIL1":8,"cifar_smoothDIL2":8,"cifar_smoothDIL3":8,
         "cifar_CIL":10,"cifar_smoothCIL1":8,"cifar_smoothCIL2":8,"cifar_smoothCIL3":8}

for factor, task in factors.items():
    file_path = "./{}.csv".format(factor)
    df = pd.read_csv(file_path)
    pkl=get_multitask_experiment(name='cifar_dataset', scenario='domain', tasks=task, verbose=True, exception=True,factor=factor,df=df)
    with open(factor+'.pk','wb') as f:
        pickle.dump(pkl,f)
