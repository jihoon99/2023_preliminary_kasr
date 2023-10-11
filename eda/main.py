import os
import logging
import pandas as pd
import nova
from nova import DATASET_PATH

dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data')
print('dataset_path : ', dataset_path)
label_path = os.path.join(DATASET_PATH, 'train', 'train_label') 
print(f'label_path : {label_path}')

print('-'*100)

dataset_path_list_dir = os.listdir(dataset_path)
print(f'dataset_path_list_dir top 10 :  {dataset_path_list_dir[:10]}')
print(f'len_dataset_path_list_dir : {len(dataset_path_list_dir)}')
print('-'*100)


read_label_csv = pd.read_csv(label_path)
print('label_csv')
print(read_label_csv.head(30))
print('*'*30)
print(len(read_label_csv))
print("*"*30)
print(read_label_csv.columns)

