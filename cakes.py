import numpy as np
import openai

import torch
from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from botorch.optim import optimize_acqf

from gp import fit_gp_model

# prompts for the LLM
SYSTEM_PROMPT_TEMPLATE = """
You are an expert in machine learning, specializing in Gaussian processes.
Here are the observations we have collected so far:
{observations}

Please analyze these observations to identify patterns in the data that can be captured by a kernel function.
You can use squared exponential (SE) kernel to capture smoothness, periodic (PER) kernel to capture periodicity, linear (LIN) kernel to capture linear trends, or rational quadratic kernel (RQ) to capture varying data patterns.
You can also combine these kernels using the + and * operators to capture more complex patterns.
For example, LIN + PER can capture a linear trend with periodic fluctuations and LIN * PER can capture a periodic pattern with linearly increasing amplitude.
"""

CROSSOVER_PROMPT_TEMPLATE = """
You are given two parent kernels and their BIC values:
{parent_kernel1} (BIC: {bic1}) and {parent_kernel2} (BIC: {bic2}).
Please recommend a kernel that has a lower BIC value.
You can either combine the parent kernels with + or * operator.

Your output should follow the following format:
Kernel: <your proposed kernel here, only use the abbreviations (SE, PER, LIN, RQ) and operators (+, *)>
Analysis: <your analysis here, explaining your reason behind the chosen kernel>
"""

MUTATION_PROMPT_TEMPLATE = """
You are given a kernel and its BIC value:
{kernel} (BIC: {bic}).
Please recommend a kernel that has a lower BIC value.
You can only replace one of the base kernels in the kernel with another base kernel.
Do not add or change any operators.

Your output should follow the following format:
Kernel: <your proposed kernel here, only use the abbreviations (SE, PER, LIN, RQ) and operators (+, *)>
Analysis: <your analysis here, explaining your reason behind the chosen kernel>
"""

class CAKES:
    def __init__(
            self,
            num_crossover=1, # number of crossovers operation
            mutation_prob=0.7,
            num_population=6, # number of kernels to keep in the population
            model_name="gpt-4o-mini", # LLM to use
            temperature=0.7, # higher temperature indicates more randomness
            top_p=0.95, # higher top_p indicates more diversity
            keys_path="./secrets.txt", # path to API key
            device="cpu"
        ):
        self.num_crossover = num_crossover
        self.mutation_prob = mutation_prob
        self.num_population = num_population

        # initial population
        self.population = {
            "SE": {},
            "PER": {},
            "LIN": {},
            "RQ": {},
            "M1": {},
            "M3": {},
            "M5": {},
        }

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.device = device

        # openai API setup
        openai.api_base = "https://api.ai-gaochao.cn/v1"
        with open(keys_path, encoding="utf-8", mode="r") as fr:
            self.keys = [line.strip() for line in fr if len(line.strip()) >= 4]
            openai.api_key = self.keys[0]

    def __call__(self, message, system_prompt):
        if not message:
            return False, "Your input is empty."

        # create LLM
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            temperature=self.temperature,
            top_p=self.top_p
        )
        return response["choices"][0]["message"]["content"]

    @staticmethod
    def parse_response(response):
        """
        Function to parse the response from the LLM.
        Args:
            response (str): response from the LLM.
        """
        kernel_start = response.find("Kernel: ") + len("Kernel: ")
        kernel_end = response.find("\n", kernel_start)
        kernel = response[kernel_start:kernel_end]

        analysis_start = response.find("Analysis: ") + len("Analysis: ")
        analysis = response[analysis_start:]
        return kernel, analysis

    def update_data(self, train_x, train_y):
        """
        Function to update the training data and system prompt.
        Args:
            train_x (torch.Tensor): training input data.
            train_y (torch.Tensor): training output data.
        """
        self.train_x = train_x.to(self.device)
        self.train_y = train_y.to(self.device)

        # update the data and system prompt
        observations = list(zip(self.train_x.tolist(), self.train_y.tolist()))
        observations = "\n".join([f"x = {x}, y = {y}" for x, y in observations])
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(observations=observations)

    def compute_fitness(self):
        """
        Function to compute the BIC values for the kernels in the population.
        """
        # compute the BIC values for each kernel
        for kernel in self.population:
            model, likelihood, bic = fit_gp_model(self.train_x, self.train_y, kernel=kernel, device=self.device)
            self.population[kernel] = {
                "model": model,
                "fitness": bic
            }

        # get the fitness values
        fitness_values = torch.tensor([self.population[kernel]["fitness"] for kernel in self.population], dtype=torch.float64)
        fitness_values = (fitness_values - fitness_values.mean()) / fitness_values.std() # standardize the fitness values for numerical stability

        # normalize the fitness values as probabilities
        self.population_prob = torch.softmax(-fitness_values, dim=0)

    def generate_kernels(self):
        """
        Function to generate new kernels using crossover and mutation.
        """
        # crossover step
        mating_pool = list(self.population.keys())
        for _ in range(self.num_crossover):
            parent_kernel1, parent_kernel2 = np.random.choice(mating_pool, size=2, p=self.population_prob, replace=False)
            try:
                response = self(CROSSOVER_PROMPT_TEMPLATE.format_map({
                    "parent_kernel1": parent_kernel1,
                    "parent_kernel2": parent_kernel2,
                    "bic1": self.population[parent_kernel1]["fitness"],
                    "bic2": self.population[parent_kernel2]["fitness"]
                }), self.system_prompt)
                kernel, analysis = self.parse_response(response)
                # if the kernel is too long, discard it
            except:
                kernel = f"{parent_kernel1} {np.random.choice(['+', '*'])} {parent_kernel2}"
            try:
                if len(kernel) < 10:
                    model, likelihood, bic = fit_gp_model(self.train_x, self.train_y, kernel=kernel, device=self.device)
                    self.population[kernel] = {
                        "fitness": bic,
                        "model": model
                    }
            except:
                continue

        # mutation step
        if np.random.rand() < self.mutation_prob:
            # select the fittest kernel to mutate
            kernel_to_mutate = max(self.population, key=lambda x: self.population[x]["fitness"])
            try:
                response = self(MUTATION_PROMPT_TEMPLATE.format_map({
                    "kernel": kernel_to_mutate,
                    "bic": self.population[kernel_to_mutate]["fitness"]
                }), self.system_prompt)
                kernel, analysis = self.parse_response(response)
                model, likelihood, bic = fit_gp_model(self.train_x, self.train_y, kernel=kernel, device=self.device)
                self.population[kernel] = {
                    "fitness": bic,
                    "model": model
                }
            except:
                pass

    def update_population(self):
        """
        Function to update the population by selecting the fittest kernels.
        """
        # sort population by fitness
        self.population = dict(sorted(self.population.items(), key=lambda x: x[1]["fitness"])) # sort by fitness
        self.population = dict(list(self.population.items())[:self.num_population]) # keep the top kernels
        self.population_prob = torch.softmax(torch.tensor([self.population[kernel]["fitness"] for kernel in self.population], dtype=torch.float64), dim=0)

    def get_best_kernel(self):
        """
        Function to return the best kernel in the population.
        Returns:
            str: the best kernel in the population.
        """
        return max(self.population, key=lambda x: self.population[x]["fitness"])

    def get_next_query(self, bounds):
        """
        Function to select the next query point.
        Args:
            bounds (list): the bounds of the input space.
        Returns:
            torch.Tensor: the next query point.
        """
        dim = len(bounds[0])
        kernels = self.population.keys()

        # compute the weights as the normalized fitness values
        weights = {kernel: self.population_prob[i].item() for i, kernel in enumerate(kernels)}

        acq_vals = []
        cand_next_xs = []
        for kernel in kernels:
            # use EI as the acquisition function
            policy = ExpectedImprovement(
                model=self.population[kernel]["model"], best_f=self.train_y.max()
            )
            try:
                # optimize the acquisition function and obtain the next query point
                cand_next_x, acq_val = optimize_acqf(
                    acq_function=policy,
                    bounds=bounds,
                    q=1,
                    num_restarts=20 * dim,
                    raw_samples=50 * dim,
                    retry_on_optimization_warning=True
                )
                acq_vals.append(acq_val.item())
                cand_next_xs.append(cand_next_x)
            except:
                acq_vals.append(-np.inf)
                cand_next_xs.append(torch.tensor([np.nan]))

        # calculate the weighted acquisition values and select the best one
        weighted_acq_vals = [weights[kernel] * acq_vals[j] for j, kernel in enumerate(kernels)]
        best_acq_idxs = np.where(weighted_acq_vals == np.max(weighted_acq_vals))[0]
        best_acq_idx = np.random.choice(best_acq_idxs) # handle ties
        next_x = cand_next_xs[best_acq_idx]

        # print kernel
        print(f"kernel: {list(kernels)[best_acq_idx]}")

        return next_x

    def run(self, train_x, train_y):
        """
        Function to run CAKES for kernel selection.
        Args:
            train_x (torch.Tensor): training input data.
            train_y (torch.Tensor): training output data.
        Returns:
            str: the best kernel selected by CAKES.
        """
        self.update_data(train_x, train_y)
        self.compute_fitness()
        self.generate_kernels()
        self.update_population()