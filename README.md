# PCAT CODE
Official Code for paper "Paper-Level Computerized Adaptive Testing for High-Stakes
Examination via Multi-Objective Optimization"

## requirements:
See requirements.txt for details

## how to run codes
we have provided the preprocessed dataset Assistment17、MOOCRadar、ComputationalThinking、ProbabilityAndStatistic. So we can run the main.py program on NCD with these datasets directly in the scripts/run_code folder.

## how to run codes from scratch
1.put raw data in data/ .
2.run data/generate_densedata.py and Put the parameters of the preprocessed dataset in data_params_dict.py.
3.run scripts/run_code/main.py