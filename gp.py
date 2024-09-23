import torch
from torch.optim import Adam
from gpytorch.kernels import RBFKernel, PeriodicKernel, LinearKernel, RQKernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll

import math
import re

def parse_kernel(expression, d):
    """
    Helper function to parse the kernel expression.
    Args:
        expression (str): The kernel expression to parse.
    Returns:
        ScaleKernel: The parsed kernel.
    """
    # set of base kernels
    base_kernels = {
        'SE': RBFKernel(ard_num_dims=d),
        'PER': PeriodicKernel(ard_num_dims=d),
        'LIN': LinearKernel(ard_num_dims=d),
        'RQ': RQKernel(ard_num_dims=d),
        'M1': MaternKernel(nu=0.5, ard_num_dims=d),
        'M3': MaternKernel(nu=1.5, ard_num_dims=d),
        'M5': MaternKernel(nu=2.5, ard_num_dims=d)
    }
    def apply_operation(left, op, right):
        if op == '+':
            return left + right
        elif op == '*':
            return left * right

    def parse_subexpression(subexpr):
        base_kernel = re.findall(r'[\w]+', subexpr)
        operators = re.findall(r'[\+\*]', subexpr)

        result = base_kernels[base_kernel[0]] # base kernel
        # apply the operators to the base kernels
        for i, op in enumerate(operators):
            result = apply_operation(result, op, base_kernels[base_kernel[i + 1]])
        return ScaleKernel(result)

    pattern = r'\(([^()]+)\)'
    cache = {} # cache the parsed subexpressions
    while '(' in expression:
        for subexpr in re.findall(pattern, expression):
            if subexpr not in cache:
                sub_kernel = parse_subexpression(subexpr)
                cache[subexpr] = sub_kernel
                base_kernels[f'SubKernel{len(base_kernels)}'] = sub_kernel
            expression = expression.replace(f'({subexpr})', f'SubKernel{len(base_kernels) - 1}', 1)
    return parse_subexpression(expression)

def fit_gp_model(train_x, train_y, kernel='SE', num_train_iters=500, compute_bic=True, device="cpu"):
    """
    Function to fit a GP model to the training data.
    Args:
        train_x (torch.Tensor): training input data
        train_y (torch.Tensor): training output data
        kernel (str): kernel expression
        num_train_iters (int): number of training iterations
        compute_bic (bool): whether to compute the Bayesian Information Criterion (BIC)
        device (str): device to store the data
    Returns:
        model: the trained GP model
        likelihood: the likelihood of the GP model
        bic (float): the BIC value
    """
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    d = train_x.shape[-1] # dimension of the input space

    # parse the kernel expression
    covar_module = parse_kernel(kernel, d)

    # define the GP
    model = SingleTaskGP(
        train_x, train_y.unsqueeze(-1),
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=d)
    )

    likelihood = model.likelihood
    likelihood.noise = 1e-4
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # fit the GP model
    fit_gpytorch_mll(mll)

    # calculate BIC
    if compute_bic:
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            output = model(train_x)
            log_likelihood = mll(output, train_y).item()

        num_params = sum(param.numel() for param in model.parameters())
        num_data = train_x.size(0)
        bic = -2 * log_likelihood + num_params * math.log(num_data)

        return model, likelihood, bic

    return model, likelihood