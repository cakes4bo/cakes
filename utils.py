import numpy as np
import torch
from torch.quasirandom import SobolEngine

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")


def generate_train_data(n, objective, bounds, seed=None, device="cpu"):
    """
    Helper function to randomly generate training data.
    Args:
        n (int): number of data points to generate
        objective (callable): objective function
        bounds (tuple): bounds of the input space
        seed (int): random seed
        device (str): device to store the data
    Returns:
        train_x (torch.Tensor): training input data
        train_y (torch.Tensor): training output data
    """
    if seed is None:
        seed = np.random.randint(100)
    torch.manual_seed(seed) # set seed for reproducibility

    d = len(bounds[0]) # dimension of the input space
    train_x = torch.zeros(n, d)
    for i in range(d):
        train_x[:, i] = (bounds[1][i] - bounds[0][i]) * torch.rand(n, device=device) + bounds[0][i]
    train_y = objective(train_x)

    return train_x, train_y

def plot_objective(objective, bounds, ns=1000):
    """
    Helper function to plot the objective function.
    Args:
        objective (callable): objective function
        bounds (tuple): bounds of the input space
        ns (int): number of samples to plot
    """
    plt.figure(figsize=(8, 6))

    d = len(bounds[0]) # dimension of the input
    if d == 1:
        ns = 1000
        xs = torch.linspace(bounds[0].item(), bounds[1].item(), ns + 1).unsqueeze(1)
        ys = objective(xs)

        sns.lineplot(x=xs.squeeze().numpy(), y=ys.numpy(), color="r")
        plt.xlabel("$x$", fontsize=16)
        plt.ylabel("$f(x)$", fontsize=16)

    elif d == 2:
        xs1 = torch.linspace(bounds[0][0], bounds[1][0], ns + 1)
        xs2 = torch.linspace(bounds[0][1], bounds[1][1], ns + 1)
        x1, x2 = torch.meshgrid(xs1, xs2, indexing="ij")
        xs = torch.vstack((x1.flatten(), x2.flatten())).transpose(-1, -2)
        ys = objective(xs)

        plt.imshow(
            ys.reshape(ns + 1, ns + 1).T,
            extent=(bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]),
            origin="lower",
            cmap="viridis"
        )
        plt.colorbar()
        plt.xlabel("$x_1$", fontsize=16)
        plt.ylabel("$x_2$", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_metric(path, method, plot_ci=False, linestyle="-", linewidth=3, color="r", ax=plt):
    """
    Helper function to plot the metric.
    Args:
        path (str): path to the metric values
        method (str): method name
        metric (str): metric name
        linestyle (str): line style
        color (str): line color
    """
    def ci(y):
        return 2 * y.std(axis=0) / np.sqrt(REPEAT)

    values = torch.load(path, map_location=torch.device("cpu"))
    REPEAT, BUDGET = values.shape

    avg_values = values.mean(axis=0)
    ci_values = ci(values)
    queries = np.arange(1, BUDGET+1)

    ax.plot(queries, avg_values, linestyle=linestyle, label=method, color=color, linewidth=linewidth)

    if plot_ci:
        ax.fill_between(
            queries,
            avg_values + ci_values,
            avg_values - ci_values,
            alpha=0.1,
            color=color
        )

def generate_config(n, task_info, seed=1, device="cpu"):
    """
    Helper function to generate sample configurations.
    Args:
        n (int): number of configurations to generate
        task_info (dict): task information
        seed (int): random seed
        device (str): device to store the data
       Returns:
        train_x (torch.Tensor): training input data
        train_y (torch.Tensor): training output data
    """
    benchmark = task_info["benchmark"]
    objective = task_info["objective"]
    config_space = task_info["config_space"]

    initial_configs = [dict(config_space.sample_configuration()) for _ in range(n)]
    train_x = torch.tensor([list(config.values()) for config in initial_configs], dtype=torch.float64)
    train_y = torch.tensor([objective(config) for config in initial_configs], dtype=torch.float64).unsqueeze(-1)

    return train_x, train_y.squeeze_()

def preprocess_query(next_x, ml_model):
    """
    Helper function to preprocess the next query point.
    Args:
        next_x (torch.Tensor): next query point.
        ml_model (str): type of ML model.
    Returns:
        cand_x (torch.Tensor): preprocessed query point.
    """
    if ml_model == "lr":
        cand_x = next_x[0].tolist()
        cand_x = {
           "alpha": cand_x[0],
           "eta0": cand_x[1]
        }
    elif ml_model == "svm":
        cand_x = next_x[0].tolist()
        cand_x = {
           "C": cand_x[0],
           "gamma": cand_x[1]
        }
    elif ml_model == "rf":
        cand_x = next_x[0].tolist()
        cand_x = {
            "max_depth": int(round(cand_x[0])),
            "max_features": cand_x[1],
            "min_samples_leaf": int(round(cand_x[2])),
            "min_samples_split": int(round(cand_x[3]))
        }
    elif ml_model == "xgb":
        cand_x = next_x[0].tolist()
        cand_x = {
            "colsample_bytree": cand_x[0],
            "eta": cand_x[1],
            "max_depth": int(round(cand_x[2])),
            "reg_lambda": cand_x[3]
        }
    elif ml_model == "mlp":
        cand_x = next_x[0].tolist()
        cand_x = {
            "alpha": cand_x[0],
            "batch_size": int(round(cand_x[1])),
            "depth": int(round(cand_x[2])),
            "learning_rate_init": cand_x[3],
            "width": int(round(cand_x[4]))
        }

    return cand_x