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
├── code
│   ├── data
│   ├── data_loader
│   │   ├── __init__.py
│   │   ├── datatool.py
│   │   └── utils.py
│   ├── main.py
│   ├── modules.py
│   ├── tools
│   │   ├── __init__.py
│   │   ├── argtools.py
│   │   └── misc.py
│   └── zhepei_driver.sh
├── LICENSE
├── README.md
└── requirements.txt
```

## Execution Instructions

**Prerequisite**
* Python V_T_B_A
* PyTorch V_T_B_A

**Execution**
1. Install the required libaries.
```
pip install -r requirements.txt
```
2. Execute the magic scripts
```
./run.sh
```
