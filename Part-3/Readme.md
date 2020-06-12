# Part 3: Profiling and Fingerprinting

## Group 15

- Sharwin Bobde (5011639)
- Nikhil Saldanha (4998707)

## Tasks Done

- Sharwin Bobde:
  1. Sketching Task
  2. Random hyperplane locality sensitive hashing task

- Nikhil Saldanha:
  1. Frequent Task
  2. Min-wise locality sensitive hashing task

## Setting up

- Create a folder for the data called `data/`.
- Download the data into `data/` from [here](https://www.stratosphereips.org/datasets-ctu13/).
- Setup virtualenv: `virtualenv -p python3 env`
- Enter virtualenv: `source env/bin/activate`
- Install requirements: `pip install -r requirements.txt`
- Start Jupyter Server: `jupyter notebook .`



Ensure that your working directory has the following structure

```
.
├── data
│   ├── CTU-13-Dataset
│   │   ├── 10
│   │   │   ├── botnet-capture-20110818-bot.pcap
│   │   │   ├── capture20110818.binetflow
│   │   │   ├── rbot.exe
│   │   │   └── README
│   │   ├── 11
│   │   │   ├── botnet-capture-20110818-bot-2.pcap
│   │   │   ├── capture20110818-2.binetflow
│   │   │   ├── rbot.exe
│   │   │   └── README
│   │   ├── 12
│   │   │   ├── 3d3d%3F%3F%3F%3F.xls.exe
│   │   │   ├── botnet-capture-20110819-bot.pcap
│   │   │   ├── capture20110819.binetflow
│   │   │   └── README
│   │   └── 9
│   │       ├── botnet-capture-20110817-bot.pcap
│   │       ├── capture20110817.binetflow
│   │       ├── Neris.exe
│   │       └── README
│   ├── _TotBytes_Dur_discretized.csv
│   ├── TotBytes_Dur_discretized.csv
│   └── utils
│       ├── discretized_scenario_09.csv
│       ├── discretized_scenario_10.csv
│       ├── discretized_scenario_11.csv
│       ├── discretized_scenario_12.csv
│       ├── discretized_scenario_9.csv
│       ├── dist_per_scene.pkl
│       ├── DstAddr_LabelEncoder_10.pkl
│       ├── DstAddr_LabelEncoder_11.pkl
│       ├── DstAddr_LabelEncoder_12.pkl
│       ├── DstAddr_LabelEncoder_9.pkl
│       ├── DstAddr_LabelEncoder.pkl
│       ├── labels_per_scene.pkl
│       ├── SrcAddr_LabelEncoder_10.pkl
│       ├── SrcAddr_LabelEncoder_11.pkl
│       ├── SrcAddr_LabelEncoder_12.pkl
│       ├── SrcAddr_LabelEncoder_9.pkl
│       └── SrcAddr_LabelEncoder.pkl
├── count_min.py
├── hyperplane_LSH.py
├── Lab3.ipynb
├── preprocessing.py
├── profiling.py
├── Readme.md
├── requirements.txt
├── task1-discretization.ipynb
├── task2-frequent-task.ipynb
├── task3-sketching.ipynb
├── task4-minwise-lsh-task.ipynb
├── task5-hyperplane.ipynb
├── task6-botnet-profiling.ipynb
├── task7-fingerprinting.ipynb
└── task8-bonus-task.ipynb
```

