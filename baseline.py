import numpy as np
import torch
import gpytorch
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms import Normalize, Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, MaternKernel, LinearKernel, PeriodicKernel, RQKernel


from gp import fit_gp_model
from utils import generate_train_data, generate_config

class EGP:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs["device"]
        self.bounds = configs["bounds"]
        self.dim = configs["dim"]
        self.kernels = [RBFKernel(), LinearKernel(), PeriodicKernel(), RQKernel()]
        self.kernel_names = ["SE", "LIN", "PER", "RQ"]
        self.M = len(self.kernels)
        self.weights = torch.ones(self.M, device=self.device) / self.M

    def fit_models(self, train_x, train_y):
        self.models = []
        for kernel in self.kernels:
            model = SingleTaskGP(
                train_x, train_y.unsqueeze(-1),
                covar_module=kernel,
                outcome_transform=Standardize(m=1),
                input_transform=Normalize(d=self.dim)
            )
            likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-5))
            likelihood.noise = 1e-4
            mll = ExactMarginalLogLikelihood(likelihood, model)
            fit_gpytorch_model(mll)
            self.models.append(model)

    def update_weights(self, train_x, train_y):
        log_likelihoods = torch.zeros(self.M, device=self.device)
        for i, model in enumerate(self.models):
            log_likelihoods[i] = model.likelihood(model(train_x)).log_prob(train_y).sum()
        self.weights = torch.softmax(log_likelihoods, dim=0)

    def sample_model(self):
        return np.random.choice(self.M, p=self.weights.detach().numpy())

    def get_next_query(self, train_x, train_y, configs):
        self.fit_models(train_x, train_y)
        self.update_weights(train_x, train_y)

        sampled_model_index = self.sample_model()
        model = self.models[sampled_model_index]

        print(f"kernel: {self.kernel_names[sampled_model_index]}")

        policy = ExpectedImprovement(model=model, best_f=train_y.max())
        try:
            next_x, _ = optimize_acqf(
                acq_function=policy,
                bounds=self.bounds,
                q=1,
                num_restarts=20 * self.dim,
                raw_samples=50 * self.dim,
                retry_on_optimization_warning=True
            )
        except:
            # next_x, _ = generate_train_data(1, configs["objective"], self.bounds, device=self.device)
            next_x, _ = generate_config(1, self.configs, device=self.device)

        return next_x

def get_next_query_egp(train_x, train_y, configs):
    egp = EGP(configs)
    return egp.get_next_query(train_x, train_y, configs)

def get_next_query_fixed(train_x, train_y, configs, kernel="SE"):
    device = configs["device"]
    bounds = configs["bounds"]
    dim = configs["dim"]

    print(f"kernel: {kernel}")
    model, likelihood = fit_gp_model(train_x, train_y, kernel=kernel, compute_bic=False, device=device)
    policy = ExpectedImprovement(model=model, best_f=train_y.max())
    try:
        next_x, _ = optimize_acqf(
            acq_function=policy,
            bounds=bounds,
            q=1,
            num_restarts=20 * dim,
            raw_samples=50 * dim,
            retry_on_optimization_warning=True
        )
    except:
        # next_x, _ = generate_train_data(1, configs["objective"], bounds, device=device)
        next_x, _ = generate_config(1, configs, device=device)
    return next_x

def get_next_query_adaptive(train_x, train_y, configs, strategy="random"):
    device = configs["device"]
    bounds = configs["bounds"]
    dim = configs["dim"]

    kernels = ["SE", "PER", "LIN", "RQ", "M3", "M5"]
    if strategy == "random":
        kernel = np.random.choice(kernels)
        print(f"kernel: {kernel}")
        model, likelihood = fit_gp_model(train_x, train_y, kernel=kernel, compute_bic=False, device=device)
        policy = ExpectedImprovement(model=model, best_f=train_y.max())
        try:
            next_x, _ = optimize_acqf(
                acq_function=policy,
                bounds=bounds,
                q=1,
                num_restarts=20 * dim,
                raw_samples=50 * dim,
                retry_on_optimization_warning=True
            )
        except:
            # next_x, _ = generate_train_data(1, configs["objective"], bounds, device=device)
            next_x, _ = generate_config(1, configs, device=device)
    elif strategy in ["best", "bic"]:
        models = {}
        for kernel in kernels:
            model, likelihood, bic = fit_gp_model(train_x, train_y, kernel=kernel, compute_bic=True, device=device)
            models[kernel] = {"model": model, "bic": bic}

        if strategy == "best":
            best_acq_val = -np.inf
            for kernel in kernels:
                model = models[kernel]["model"]
                policy = ExpectedImprovement(model=model, best_f=train_y.max())
                try:
                    cand_next_x, acq_val = optimize_acqf(
                        acq_function=policy,
                        bounds=bounds,
                        q=1,
                        num_restarts=20 * dim,
                        raw_samples=50 * dim
                    )
                except:
                    acq_val = -np.inf
                    cand_next_x = torch.tensor([float("nan") for _ in range(dim)], device=device).unsqueeze(0)
                if acq_val > best_acq_val:
                    best_acq_val = acq_val
                    next_x = cand_next_x
                    best_kernel = kernel
            print(f"kernel: {best_kernel}")
        elif strategy == "bic":
            best_bic = np.inf
            for kernel in kernels:
                bic = models[kernel]["bic"]
                if bic < best_bic:
                    best_bic = bic
                    best_model = models[kernel]["model"]
                    best_kernel = kernel
            print(f"kernel: {best_kernel}")
            policy = ExpectedImprovement(model=best_model, best_f=train_y.max())
            try:
                next_x, _ = optimize_acqf(
                    acq_function=policy,
                    bounds=bounds,
                    q=1,
                    num_restarts=20 * dim,
                    raw_samples=50 * dim,
                    retry_on_optimization_warning=True
                )
            except:
                # next_x, _ = generate_train_data(1, configs["objective"], bounds, device=device)
                next_x, _ = generate_config(1, configs, device=device)
    return next_x