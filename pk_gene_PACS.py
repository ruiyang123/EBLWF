import pickle
from data import get_multitask_experiment
import pandas as pd





factors={"PACS_DIL":4,"PACS_smoothDIL1":6,"PACS_smoothDIL2":6,"PACS_smoothDIL3":6,
         "PACS_CIL":7,"PACS_smoothCIL1":6,"PACS_smoothCIL2":6,"PACS_smoothCIL3":6}

for factor, task in factors.items():
    file_path = "./{}.csv".format(factor)
    df = pd.read_csv(file_path)
    pkl=get_multitask_experiment(name='Pacs_dataset', scenario='domain', tasks=task, verbose=True, exception=True,factor=factor,df=df)
    with open(factor+'.pk','wb') as f:
        pickle.dump(pkl,f)
