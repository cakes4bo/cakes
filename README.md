# Context-Aware Kernel Search for Bayesian Optimization with Large Language Models

Official code repository for "Context-Aware Kernel Search for Bayesian Optimization with Large Language Models"

## Setup

1. Create a `secrets.txt` file and provide the OpenAPI key
2. Set up a Conda environment:
```
git clone https://github.com/cakes4bo/cake.git
conda create -n cakes python=3.9
conda activate cakes
```

3. Install the requirements:
```
pip install -r requirements.txt
```

## Reproducing Results

To reproduce results, execute the following Python scripts:
- To run optimization benchmark: ```python exp.py```
- To run HPOBench benchmark: ```python hpobench_exp.py```
