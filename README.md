# Masters-Thesis
Formulating the problem of data partitioning in a reinforcement learning framework. A working environment suitable for applying DRL agents, with options tailored towards improving its efficiency, such as file and runtime caching.

---

This environment can be executed directly in Colab (in future hopefully I will provide how we can pip install this environment and can adapt it as any other gym environment).

---

Dataset
---
As a benchmark dataset, I have used Transaction Processing Performance Council
Benchmark (TPC-H) for performing the experiments.

The benchmark consists of 8 tables and 22 queries

According to this repository (how I have set up the environment class for executing in Colab), for accessing, the dataset can be uploaded on Google Drive. 

Dataset Source: http://www.tpc.org/tpch/


There are two files:
```
environment_class.py -> The environment class
queries.py -> All the TPC-H queries
```

## TODO
* [ ] Provide documentation of how this environment work 
* [ ] How it can be used with DRL agents from frameworks for example ACME (pip install and in colab)

---
