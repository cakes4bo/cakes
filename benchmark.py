import torch
import botorch

from hpobench.benchmarks.ml import LRBenchmark, SVMBenchmark, RandomForestBenchmark, NNBenchmark
from hpobench.benchmarks.ml.xgboost_benchmark import XGBoostBenchmark
import ConfigSpace as CS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BoTorch test functions: https://botorch.org/api/test_functions.html.
TEST_FUNCTIONS = {
    "ackley2": botorch.test_functions.Ackley(dim=2, negate=True),
    "ackley5": botorch.test_functions.Ackley(dim=5, negate=True),
    "beale": botorch.test_functions.Beale(negate=True),
    "branin": botorch.test_functions.Branin(negate=True),
    "dropwave": botorch.test_functions.DropWave(negate=True),
    "eggholder": botorch.test_functions.EggHolder(negate=True),
    "griewank2": botorch.test_functions.Griewank(dim=2, negate=True),
    "griewank5": botorch.test_functions.Griewank(dim=5, negate=True),
    "hartmann": botorch.test_functions.Hartmann(dim=3, negate=True),
    "levy2": botorch.test_functions.Levy(dim=2, negate=True),
    "levy3": botorch.test_functions.Levy(dim=3, negate=True),
    "rastringin2": botorch.test_functions.Rastrigin(dim=2, negate=True),
    "rastringin4": botorch.test_functions.Rastrigin(dim=4, negate=True),
    "rosenbrock": botorch.test_functions.Rosenbrock(dim=2, negate=True),
    "sixhumpcamel": botorch.test_functions.SixHumpCamel(negate=True)
}

def get_objective(task):
    """
    Helper function to get the objective function, bounds, and ground truth value.
    """
    if task in TEST_FUNCTIONS:
        f = TEST_FUNCTIONS[task]
        bounds = f.bounds.to(device)
        if task in ["ackley2", "ackley5"]:
            f.bounds[0, :].fill_(-5)
            f.bounds[1, :].fill_(5)
        objective = lambda x: -f.evaluate_true(x)
        ground_truth_y = f.optimal_value

        return objective, bounds, ground_truth_y


TASK_IDS = {
    "credit_g": 31,
    "vehicle": 53,
    "kc1": 3917,
    "phoneme": 9952,
    "blood_transfusion": 10101,
    "australian": 146818,
    "car": 146821,
    "segment": 146822,
    "heart_h": 50,
    "tic_tac_toe": 145804,
    "kr_vs_kp": 3,
    "qsar": 9957
}

def get_bounds(config_space, custom_bounds):
    bounds = []
    for hp_name, (lower, upper) in custom_bounds.items():
        hp = config_space.get_hyperparameter(hp_name)
        if isinstance(hp, CS.UniformFloatHyperparameter) or isinstance(hp, CS.UniformIntegerHyperparameter):
            hp.lower = lower
            hp.upper = upper
            bounds.append((hp.lower, hp.upper))
        elif isinstance(hp, CS.CategoricalHyperparameter):
            hp.choices = hp.choices[lower:upper+1]
            bounds.append((0, len(hp.choices) - 1))
    return torch.tensor(bounds).T

def get_task(ml_model, dataset, seed=0):
    if ml_model == "lr":
        benchmark = LRBenchmark(task_id=TASK_IDS[dataset], rng=seed)
        custom_bounds = {
            "alpha": (0.001, 1.0),
            "eta0": (0.001, 1.0)
        }

    elif ml_model == "svm":
        benchmark = SVMBenchmark(task_id=TASK_IDS[dataset], rng=seed)
        custom_bounds = {
            "C": (0.01, 10),
            "gamma": (0.001, 1)
        }
    elif ml_model == "rf":
        benchmark = RandomForestBenchmark(task_id=TASK_IDS[dataset], rng=seed)
        custom_bounds = {
            "max_depth": (1, 50),
            "max_features": (0.0, 1.0),
            "min_samples_leaf": (1, 2),
            "min_samples_split": (2, 128)
        }
    elif ml_model == "xgb":
        benchmark = XGBoostBenchmark(task_id=TASK_IDS[dataset], rng=seed)
        custom_bounds = {
            "colsample_bytree": (0.1, 1.0),
            "eta": (0.001, 1.0),
            "max_depth": (1, 50),
            "reg_lambda": (0.01, 10.0)
        }
    elif ml_model == "mlp":
        benchmark = NNBenchmark(task_id=TASK_IDS[dataset], rng=seed)
        custom_bounds = {
            "alpha": (0.001, 1.0),
            "batch_size": (16, 128),
            "depth": (1, 3),
            "learning_rate_init": (0.001, 1.0),
            "width": (16, 128)
        }

    config_space = benchmark.get_configuration_space(seed=seed)
    objective = lambda x: benchmark.objective_function(configuration=x)["info"]["test_scores"]["acc"]
    bounds = get_bounds(config_space, custom_bounds)

    return benchmark, objective, bounds, config_space



