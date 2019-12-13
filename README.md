# CS547 Final Project: LSTM, bi-LSTM and Attention model
Source code, project reports, execution instructions for our implementations of LSTM, bi-LSTM and Attention models for Yelp Review Polarity.

## Team Members
* Zhepei Wang (<zhepeiw2@illinois.edu>)
* Peilun Zhang (<peilunz2@illinois.edu>)
* Hao Wu (<haow11@illinois.edu>)
* Efthymios Tzinis (<etzinis2@illinois.edu>)
* Sahand Mozaffari (<sahandm2@illinois.edu>)

## Folder Structures and File Descriptions

```
├── LICENSE
├── README.md
├── code
│   ├── data_loader
│   │   ├── __init__.py
│   │   ├── datatool.py   # Wrapper class for the data set
│   │   └── utils.py      # Utility functions to convert data set to data loader
│   ├── main.py           # Main rountine for a single run of experiment
│   ├── models.py         # Wrapper class for all three different model structures
│   ├── modules.py        
│   ├── parallel_experiment_runner.py # Helper file to paralleling experiments 
│   ├── thymios_driver.sh
│   ├── tools
│   │   ├── __init__.py
│   │   ├── argtools.py  # Util functions to get and parse command line arguments
│   │   └── misc.py         
│   ├── vis_results.py   # File to generate visualizations 
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
