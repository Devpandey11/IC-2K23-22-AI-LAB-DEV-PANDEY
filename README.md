# AI Lab Solutions (Python)

This repository contains runnable examples for the 7 labs in the supplied lab-wise mapping.

## Requirements
- Python 3.10+
- Install common packages: `pip install -r requirements.txt`
- Lab 6 and Lab 7 use PyTorch. For a CPU setup, the requirements file installs the standard PyTorch packages; if installation fails, use the official PyTorch install selector for your device.

## Run
Open a terminal in this folder and run one script at a time:
```bash
python lab1_environment_water_jug.py
python lab2_search_algorithms.py
python lab3_data_preprocessing_eda.py
python lab4_knowledge_representation.py
python lab5_decision_tree.py
python lab6_pytorch_basics.py
python lab7_xai_adversarial_pca.py
```

Some scripts create small sample datasets automatically. Lab 7 downloads MNIST on its first run and may take time. No dataset is included in this repository.

## Lab mapping
1. Environment setup, production systems, Water Jug Problem
2. BFS, DFS, Hill Climbing, Best First Search, A*
3. Data preprocessing, EDA, NumPy, Pandas, visualization
4. Knowledge representation, rule-based systems, semantic networks
5. Decision Tree classifier, model evaluation
6. PyTorch basics, linear regression, neural network fundamentals
7. Grad-CAM, FGSM, PCA/SVD, mini project (MNIST image classification)

Educational examples: adjust inputs and explain the output during your practical/viva.
