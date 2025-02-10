import pandas as pd
import os
import numpy as np
import json

#——————————————————————————————————————————————————————————————

#该文件解释说明：此代码用于将原本的稀疏数据集变成密集数据集

#——————————————————————————————————————————————————————————————


# 读取CSV文件并创建学生-题目矩阵
dataset= 'DatabaseTechnologyAndApplication'
type = 'new'    #new 获 old
print('数据集:',dataset)
print('item:',type)
if type == 'new':
    data = pd.read_csv('{}/TotalData.csv'.format(dataset))
    data.rename(columns={'user_id': 'student_id', 'item_id': 'question_id','score':'correct'}, inplace=True)
else:
    data = pd.read_csv('{}/TotalData.csv'.format(dataset), header=None, names=['student_id', 'question_id', 'correct'])

if not os.path.exists('{}/q.csv'.format(dataset)):
    # 如果目录不存在，则创建它
    item = pd.read_csv('{}/item.csv'.format(dataset))
    item['knowledge_code'] = item['knowledge_code'].apply(eval)
    num_knowledge = item['knowledge_code'].apply(max).max() + 1
    num_question = item['item_id'].max()+1
    #print('知识点数量:',num_knowledge)
    q = np.zeros((num_question, num_knowledge))
    for index, row in item.iterrows():
        #print('i:',i)
        #print('当前题目:',que)
        for know in row[1]:
            #print('当前知识点:',know)
            q[row[0], know] = 1
    q = pd.DataFrame(q)

else:
    q = pd.read_csv('{}/q.csv'.format(dataset), header=None)

data_sq = data.drop(columns=['correct'])
#预先筛除
# student_counts = data.groupby("student_id")["question_id"].count()
# question_counts = data.groupby("question_id")["student_id"].count()
# valid_students = student_counts[student_counts >= 50].index
# valid_questions = question_counts[question_counts >= 100].index
# filtered_df = data[(data["student_id"].isin(valid_students)) & (data["question_id"].isin(valid_questions))]

student_question_matrix = data_sq.pivot_table(index='student_id', columns='question_id', aggfunc=len, fill_value=0)
student_question_matrix = student_question_matrix.clip(upper=1)
print(student_question_matrix)
def find_largest_subset(matrix):
    # 获取初始的学生和题目集合
    students = list(matrix.index)
    questions = list(matrix.columns)
    while True:
        print('现在学生大小:',len(students),' 现在题目大小:',len(questions))
        # 计算每个学生答题的数量和每个题目的答题学生数
        student_answer_count = matrix.loc[students, questions].sum(axis=1)
        question_answer_count = matrix.loc[students, questions].sum(axis=0)
        
        # 找出答题最少的学生和被最少学生回答的题目
        min_answer_student = student_answer_count.idxmin()
        min_answer_question = question_answer_count.idxmin()

        # if len(students) < len(questions):
        if matrix.loc[min_answer_student, questions].sum()/len(questions) > matrix.loc[students, min_answer_question].sum()/len(students):
            questions.remove(min_answer_question)
        else:
            students.remove(min_answer_student)

        # 检查是否所有剩余的学生回答了所有剩余的题目
        if (matrix.loc[students, questions] == 0).sum().sum() == 0:
            break

    return students, questions

# 寻找面积最大的学生子集和题目子集
largest_students, largest_questions = find_largest_subset(student_question_matrix)

#提取题目及其知识点矩阵
q = q.iloc[largest_questions]

# 输出子集的大小
print(f"学生子集大小: {len(largest_students)}")
print(f"题目子集大小: {len(largest_questions)}")

# 过滤原始数据，保留子集的数据
filtered_data = data[(data['student_id'].isin(largest_students)) & (data['question_id'].isin(largest_questions))]

#创建题目、学生映射
question_map = {}
student_map = {}
n = 0
for que in largest_questions:
    question_map[int(que)] = n
    n = n + 1

n = 0
for stu in largest_students:
    student_map[stu] = n
    n = n + 1

filtered_data['student_id'] = filtered_data['student_id'].map(student_map)
filtered_data['question_id'] = filtered_data['question_id'].map(question_map)

print('filtered')

if not os.path.exists('Dense_{}'.format(dataset)):
    # 如果目录不存在，则创建它
    os.makedirs('Dense_{}'.format(dataset))
    print('没有该目录，创建目录')
else:
    print('已存在该目录')

# 将子数据集输出为新的CSV文件
filtered_data.to_csv('Dense_{}/TotalData.csv'.format(dataset), index=False, header=False)
q.to_csv('Dense_{}/q.csv'.format(dataset), index=False, header=False)
with open('Dense_{}/question_map.json'.format(dataset), 'w') as f:
    json.dump(question_map, f, indent=4)  # indent=4用于格式化输出，使JSON文件更加易读

print('数据集:',dataset)
# 输出子集的大小
print(f"学生子集大小: {len(largest_students)}")
print(f"题目子集大小: {len(largest_questions)}")
print(f"知识点数量: {q.shape[1]}")
print('largest_questions:',largest_questions)

