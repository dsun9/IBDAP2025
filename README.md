# RIVGAE
This repository includes the source code used in our paper: *Modeling Online Ideological Community Dynamics with Recurrent Variational Graph Auto-Encoders*. The repository is modified from [CTGCN](https://github.com/jhljx/CTGCN). The code structure and requirements are the same.

## Functions

This modified repository has several functions, including: **preprocessing**, **graph embedding**. The corresponding Python commands are:

1. **Preprocessing**: generate k-core subgraphs and perform random walk.

       python3 main.py --config=config/sample.json --base_path=data/<folder> --task=preprocessing --method=EvolveGCN

2. **Graph Embedding**: perform graph embedding approaches on dynamic graph data sets and save the performance in JSONL file.

       python3 main.py --config=config/sample.json --base_path=data/<folder> --task=embedding --method=RIVGAE
