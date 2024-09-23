import torch

from cakes import CAKES
from gp import fit_gp_model
from benchmark import get_objective
from utils import generate_train_data

import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on", device)

# define the task
task = "ackley2" # name of test function (see 'benchmark.py')
print(f"simulation on {task}")

objective, bounds, ground_truth_y = get_objective(task)
print(f"ground truth: {ground_truth_y:.4f}")
# plot_objective(objective, bounds)

# create a problem info dictionary
configs = {
    "objective": objective,
    "bounds": bounds,
    "dim": len(bounds[0]), # dimension of the input
    "ground_truth_y": ground_truth_y,
    "device": device
}

method = "cakes" # "cakes" or "fixed or "adaptive"
print(f"method: {method}")

n = 5 # number of initial observations
BUDGET = 10 * configs["dim"] # number of queries
REPEAT = 10 # number of repetitions

incumbents = torch.zeros(REPEAT, BUDGET, device=device)
gap = torch.zeros((REPEAT, BUDGET), device=device)
for r in range(REPEAT):
    print("trial", r + 1)

    # initialize data
    train_x, train_y = generate_train_data(n, objective, bounds, seed=r, device=device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    init_y = train_y.max().item() # get the initial best value

    # initialize CAKES
    if method == "cakes":
        cakes = CAKES(model_name = "gpt-4o-mini", device=device)
        strategy = cakes.model_name
        print(f"LLM: {strategy}")

    for i in tqdm(range(BUDGET)):
        # compute incumbents and GAP
        incumbents[r, i] = train_y.max().item()
        gap[r, i] = (train_y.max().item() - init_y) / (ground_truth_y - init_y)
        print(f"incumbent: {incumbents[r, i]:.4f}")
        print(f"GAP: {gap[r, i]:.4f}")

        # use CAKES to select the kernel
        if method == "cakes":
            cakes.run(train_x, train_y)
            next_x = cakes.get_next_query(bounds)

        next_y = objective(next_x)

        next_x = next_x.to(device)
        next_y = next_y.to(device)

        # update the data
        train_x = torch.cat([train_x, next_x])
        train_y = torch.cat([train_y, next_y])

# save the results
torch.save(incumbents, f"./results/{method}/incumbents_{strategy}_{task}_repeat{REPEAT}.pth")
torch.save(gap, f"./results/{method}/gap_{strategy}_{task}_repeat{REPEAT}.pth")