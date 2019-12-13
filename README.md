# CS547 Final Project: LSTM, bi-LSTM and Attention model
Source code, project reports, execution instructions for our implementations of LSTM, bi-LSTM and Attention models for Yelp Review Polarity.

## Team members
* Zhepei Wang (<zhepeiw2@illinois.edu>)
* Peilun Zhang (<peilunz2@illinois.edu>)
* Hao Wu (<haow11@illinois.edu>)
* Efthymios Tzinis (<etzinis2@illinois.edu>)
* Sahand Mozaffari (<sahandm2@illinois.edu>)

## Folder structures

```
├── LICENSE
├── README.md
├── code
│   ├── data_loader
│   │   ├── __init__.py
│   │   ├── datatool.py
│   │   └── utils.py
│   ├── main.py
│   ├── models.py
│   ├── modules.py
│   ├── parallel_experiment_runner.py
│   ├── thymios_driver.sh
│   ├── tools
│   │   ├── __init__.py
│   │   ├── argtools.py
│   │   └── misc.py
│   ├── vis_results.py
│   └── zhepei_driver.sh
└── requirements.txt
```

## Execution Instructions

**Prerequisite**
* Python >= 3.6.9 

**Execution**
1. Install the required libaries.
```
pip install -r requirements.txt
```
2. Execute using the existing script
```
cd code/ && ./run.sh 
```
