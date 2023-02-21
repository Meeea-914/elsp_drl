# elsp_drl

## Get Started

### Installation on Linux
```
git clone https://github.com/*/elsp_drl.git & cd elsp_drl
python3 -m venv venv
source venv/bin/activate
```
Install gpu or cpu version [pytorch](https://pytorch.org/get-started/locally/)==1.7  
```
pip install -r requirments.txt
```
### Introduction
The usage of files or directories:
```
 elsp_env_manager - Python module for ELSP simulation
 experiment - Main functional module
 config.yaml - Experiment config file
```
The help context:
```
Usage: run.py train [OPTIONS]

Options:
  -t, --net_type [ssa|mlp]  Select the type of network to use
  -e, --env_no [3|4|5|6]    Set the number [i] of simulation environment,
                            where i means there are i products
  -h, --help                Show this message and exit.
```
```
Usage: run.py evaluate [OPTIONS]

Options:
  -t, --net_type [ssa|mlp]     Select the type of network to use
  -e, --env_no [3|4|5|6]       Set the number [i] of simulation environment,
                               where i means there are i products
  -s, --demand_scale FLOAT     Set the demand scale of all products
  -m, --model_path TEXT        Give the path of a trainded model file to
                               evaluate
  -r, --result_xlsx_path TEXT  Give the path of the result xlsx file
  -h, --help                   Show this message and exit.
```
## Train

```
python run.py train -t ssa -e 3
python run.py train -t ssa -e 4
python run.py train -t ssa -e 5
python run.py train -t ssa -e 6
python run.py train -t mlp -e 3
```
## Evaluate

```
python run.py evaluate -t ssa -e 3 -s 1 -m /path/to/a/optimized/model
python run.py evaluate -t mlp -e 3 -s 1 -m /path/to/a/optimized/model
```