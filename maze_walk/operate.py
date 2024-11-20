import subprocess
from copy import copy
import sys

import random as rand
def run_path(numeric_args,is_lin):
    args = ["./path","maze1.png",is_lin] + [str(a) for a in numeric_args] + ["4"]
    try:
        output = subprocess.check_output(args,timeout=30).decode("utf-8")
        sys.stdout.write(output)
        return float(output.split()[-1])
    except subprocess.TimeoutExpired:
        return 10e50
    except subprocess.CalledProcessError:
        print("error called process",file=sys.stderr)
        return 10e50

def change_arg_to(args,val,idx):
    newargs = copy(args)
    newargs[idx] = val
    return newargs

def run_args():
    curargs = [10.0 ,0.3 ,-3.0]

    for i in range(100):
        argdif_idx = rand.randint(0,len(curargs)-1)

        newval = curargs[argdif_idx]*(1+rand.uniform(-0.15,0.15))
        argsdif = change_arg_to(curargs,newval,argdif_idx)
        if run_path(argsdif,"true") < run_path(curargs,"true"):
            curargs = argsdif

def run_rands():
    for i in range(20):
        run_path([0,0,0],"false")

run_rands()
