import os
import pandas as pd
import pickle
from data import get_multitask_experiment



factors={"cifar_DIL":5,"cifar_smoothDIL1":8,"cifar_smoothDIL2":8,"cifar_smoothDIL3":8,
         "cifar_CIL":10,"cifar_smoothCIL1":8,"cifar_smoothCIL2":8,"cifar_smoothCIL3":8}

batch_size = 32
budget = 1500
iters = 2500
N = 3
for i in range(N):
    for factor,tasks in factors.items():
        methods =  ["naive", "eblwf", "cul", "oewc", "si", "er", "lwf", "mlwf"]
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
            elif method == "eblwf":
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

        os.system("python post_results.py --result-path ./{}_result --save-name {}".format(factor,factor))