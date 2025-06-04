import os

import random  

from collections import defaultdict  

dir_path = os.path.dirname(os.path.abspath(__file__))
root_folder_path = './ratings.dat'
path_name = os.path.join(dir_path, root_folder_path)

# 读取rating.dat文件  

with open(path_name, 'r') as file:  

    lines = file.readlines()  

  

# 使用字典按用户分组  

user_groups = defaultdict(list)  

for line in lines:  

    user_id, *rest = line.strip().split('::')  # 提取用户ID和剩余部分  

    # 将剩余部分用空格连接成一个字符串  

    formatted_line = ' '.join(rest)  

    # 将用户ID和格式化后的字符串重新组合成一行  

    user_groups[user_id].append(f"{user_id} {formatted_line}")  

  

# 初始化两个列表来存储训练集和测试集的数据  

train_data = []  

test_data = []  

  

# 对每个用户的数据进行随机的8:2分割  

for user_id, user_lines in user_groups.items():  

    # 打乱用户的数据  

    random.shuffle(user_lines)  

      

    # 计算分割点（确保是整数）  

    split_index = int(len(user_lines) * 0.8)  

      

    # 分配数据到训练集和测试集  

    train_data.extend(user_lines[:split_index])  

    test_data.extend(user_lines[split_index:])  

  

# 写入train.txt  

with open('train.txt', 'w') as train_file:  

    for line in train_data:  

        train_file.write(line + '\n')  # 添加换行符  

  

# 写入test.txt  

with open('test.txt', 'w') as test_file:  

    for line in test_data:  

        test_file.write(line + '\n')  # 添加换行符  

  

print(f"训练集数据行数: {len(train_data)}")  

print(f"测试集数据行数: {len(test_data)}")