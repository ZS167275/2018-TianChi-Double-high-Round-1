#coding:utf-8
import pandas as pd
print('wash label')
train = pd.read_csv('../data/meinian_round1_train_20180408.csv',encoding='gbk')
# test = pd.read_csv('../data/[new] meinian_round1_test_a_20180409.csv',encoding='gbk')


train['收缩压'] = train['收缩压'].apply(lambda x:str(x).replace('未查','0'))
train['收缩压'] = train['收缩压'].apply(lambda x:str(x).replace('弃查','0'))
train['收缩压'] = train['收缩压'].astype(float)

train['舒张压'] = train['舒张压'].apply(lambda x:str(x).replace('弃查','0'))
train['舒张压'] = train['舒张压'].apply(lambda x:str(x).replace('未查','0'))
train['舒张压'] = train['舒张压'].astype(float)

train['血清甘油三酯'] = train['血清甘油三酯'].apply(lambda x:str(x).replace('>',''))
train['血清甘油三酯'] = train['血清甘油三酯'].apply(lambda x:str(x).replace('+',''))
train['血清甘油三酯'] = train['血清甘油三酯'].apply(lambda x:str(x).replace('2.2.8','2.28'))
train['血清甘油三酯'] = train['血清甘油三酯'].apply(lambda x:str(x).replace('轻度乳糜',''))
train['血清甘油三酯'] = train['血清甘油三酯'].astype(float)

train['血清低密度脂蛋白'] = train['血清低密度脂蛋白'].apply(lambda x:str(x).replace('-',''))
train['血清低密度脂蛋白'] = train['血清低密度脂蛋白'].astype(float)

print(train.describe())
# exit()
train.to_csv('../data/train_csv.csv',index=False)