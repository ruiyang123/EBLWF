import os
import pandas as pd
import pickle
from data import get_multitask_experiment


factors={"PACS_DIL":4 ,
         "PACS_smoothDIL1":6,"PACS_smoothDIL2":6,"PACS_smoothDIL3":6,
         "PACS_CIL":7,
         "PACS_smoothCIL1":6,"PACS_smoothCIL2":6,"PACS_smoothCIL3":6}

lr = 0.05
batch_size = 8
lrd = 0.9
drop_p = 0.25
budget = 400
iters = 800

N = 3 # repetation times

for i in range(N):
    for factor,tasks in factors.items():
        methods =  ["naive", "erlwf", "cul", "oewc", "si", "er", "lwf", "mlwf"]
        for method in methods:
            if method == "cul":
                os.system("python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --cumulative 1 --batch {}".format(factor,iters,method,tasks,batch_size))


            elif method == "naive":
                os.system("python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1".format(factor,iters,
                                                                                                                         method,
                                                                                                                         tasks,batch_size))

            elif method == "er":
                os.system(
                    "python main.py --factor {} --iters {} --savepath={} --optimizer=adam --tasks {} --batch {} --reInitOptimizer 1 --replay=exemplars --budget={} --select-memory random --meta 0".format(
                        factor,iters, method, tasks,batch_size,budget))


            elif method == "lwf":
                os.system(
                    "python main.py --factor {} --iters {} --replay=current --distill --savepath={} --reInitOptimizer 1 --batch {} --optimizer=adam --tasks {} ".format(factor,iters, method,batch_size,
                                                                                                               tasks))
            elif method == "erlwf":
                os.system(
                    "python main.py --factor {} --iters {} --replay=current --distill --savepath={} --optimizer=adam --tasks {} --autolwf --reInitOptimizer 1 --batch {}".format(factor, iters, method,
                                                                                                               tasks,batch_size))

            elif method == "mlwf":
                os.system(
                    "python main.py --factor {} --iters {} --replay=exemplars --distill --savepath={} --batch {} --optimizer=adam --tasks {} --budget={} --select-memory random".format(
                        factor,iters, method,batch_size,
                        tasks,budget))

            elif method == "oewc":
                os.system(
                    "python main.py --factor {} --iters {} --ewc --online --batch {} --savepath={} --optimizer=adam --tasks {} --lambda 5000".format(
                        factor, iters, batch_size,method,
                        tasks))

            elif method =="si":
                os.system(
                    "python main.py --factor {} --iters {} --batch {} --si --savepath={} --optimizer=adam --tasks {} --c 0.95".format(
                        factor, iters, batch_size, method,
                        tasks))