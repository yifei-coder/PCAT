import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Assistment20_21', type=str, help='benchmark')
parser = parser.parse_args()


response_logs = pd.read_csv(f'./{parser.dataset}/TotalData.csv', header=None)
q = pd.read_csv(f'./{parser.dataset}/q.csv', header=None)
#print(response_logs[1])
print('学生数量:',len(response_logs[0].unique()))
print('题目数量:',len(response_logs[1].unique()))
print('知识点数量:',q.shape[1])