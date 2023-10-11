import os
import logging
import pandas as pd
import nova
from nova import DATASET_PATH

######################################################################
#                       1. 1st eda                                  #
######################################################################
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



######################################################################
#                       2. 2nd eda                                  #
######################################################################
csv_len = len(read_label_csv)
read_label_csv['text_len'] = read_label_csv['text'].apply(lambda x: len(x))
read_label_csv = read_label_csv.sort_values("text_len")
print('.95 : ', '-'*80)
print(read_label_csv.iloc[int(csv_len*0.95)])
print(read_label_csv.iloc[int(csv_len*0.95)]['text'])
print('.99 : ', '-'*80)
print(read_label_csv.iloc[int(csv_len*0.99)])
print(read_label_csv.iloc[int(csv_len*0.99)]['text'])
print('1. : ', '-'*80)
print(read_label_csv.iloc[-1])
print(read_label_csv.iloc[-1]['text'])



######################################################################
#                       2. 3rd eda                                  #
######################################################################
from core import load_audio
dataset_path

read_label_csv['filename'] = dataset_path + read_label_csv['filename']
read_label_csv['len_wav'] = read_label_csv['filename'].apply(lambda x: load_audio(x, False, x.split(".")[-1]))
print(read_label_csv.head())
