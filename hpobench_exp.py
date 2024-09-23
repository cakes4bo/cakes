import torch

from cakes import CAKES
from gp import fit_gp_model
from benchmark import get_task
from utils import *
from baseline import get_next_query_fixed, get_next_query_adaptive

import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on", device)

dataset = "qsar" # openml dataset to use (see 'benchmark.py')
ml_model = "lr" # ml model to use

n = 5 # number of initial observations
BUDGET = 25 # number of queries
REPEAT = 5 # number of repetitions
method = "cakes"

if method == "cakes":
    strategy = "gpt-4o-mini"

incumbents = torch.zeros(REPEAT, BUDGET, device=device)
for r in range(REPEAT):
    print("trial", r + 1)

    benchmark, objective, bounds, config_space = get_task(ml_model, dataset, seed=r)
    print(f"dataset: {dataset}, ml_model: {ml_model}")

    task_info = {
        "benchmark": benchmark,
        "objective": objective,
        "bounds": bounds,
        "config_space": config_space,
        "dim": len(bounds[0]),
        "device": device
    }

    # initialize data
    train_x, train_y = generate_config(n, task_info, seed=r, device=device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)

    # initialize CAKES
    if method == "cakes":
        cakes = CAKES(model_name=strategy, device=device)
        print(f"LLM: {strategy}")

    for i in tqdm(range(BUDGET)):
        # compute incumbents and GAP
        incumbents[r, i] = train_y.max().item()
        print(f"incumbent: {incumbents[r, i]:.4f}")

        # use CAKES to select the kernel
        if method == "cakes":
            cakes.run(train_x, train_y)
            next_x = cakes.get_next_query(bounds)

        cand_x = preprocess_query(next_x, ml_model)
        next_y = torch.tensor([objective(cand_x)])

        next_x = next_x.to(device)
        next_y = next_y.to(device)

        # update the data
        train_x = torch.cat([train_x, next_x])
        train_y = torch.cat([train_y, next_y])

    # save the results
    torch.save(incumbents, f"./hpobench/{method}/incumbents_{strategy}_{ml_model}_{dataset}_repeat{REPEAT}.pth")