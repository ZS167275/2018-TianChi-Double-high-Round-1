#coding:utf-8
import pandas as pd
import numpy as np
from skopt import BayesSearchCV
import lightgbm as lgb
from sklearn.model_selection import KFold
import jieba

# from textblob import TextBlob

# def get_english(data):
#     text = TextBlob(data)
#     text = text.translate(to="en")
#     return (text)

print('custome')
# jieba.load_userdict("../搜狗双高相关词库/amuse.txt")
print('custome finish')
bayes_cv_tuner = BayesSearchCV(
    estimator = lgb.LGBMRegressor(
        objective='regression',
        metric='l2',
        # n_jobs=1,
        verbose=0
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (1, 1000),
        'max_depth': (3, 50),
        'min_child_samples': (0, 500),
        'max_bin': (100, 1000),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 100),
        'subsample_for_bin': (1000, 50000),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1000, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (50, 250),
    },
    scoring = 'neg_mean_squared_error',
    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    ),
    # n_jobs = 3,
    n_iter = 10000,
    verbose = 0,
    refit = True,
    random_state = 42
)


def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest mse : {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name + "_cv_results.csv")


train = pd.read_csv('../data/train_csv.csv',encoding='gbk',low_memory=False)
test = pd.read_csv('../data/test_b_csv.csv',encoding='gbk',low_memory=False)

# train = data[:train.shape[0]]
#
#
# test = data[train.shape[0]:]

train = train[train['收缩压']>0].reset_index(drop=True)
train = train[train['收缩压']<800].reset_index(drop=True)
train = train[train['舒张压']>0].reset_index(drop=True)
train = train[train['舒张压']<750].reset_index(drop=True)
train = train[train['血清甘油三酯']>0].reset_index(drop=True)
# train = train[train['血清甘油三酯']<20].reset_index(drop=True)
train = train[train['血清高密度脂蛋白']>0].reset_index(drop=True)

# train = train[train['血清低密度脂蛋白']<10].reset_index(drop=True)
train = train[train['血清低密度脂蛋白']>0].reset_index(drop=True)

# print(train[train['收缩压']<10])
#
# exit()
train_1 = train.pop('收缩压')
train_2 = train.pop('舒张压')
train_3 = train.pop('血清甘油三酯')
train_4 = train.pop('血清高密度脂蛋白')
train_5 = train.pop('血清低密度脂蛋白')

train_index = train.pop('vid')
test_index = test.pop('vid')

test = test[train.columns]

data = pd.concat([train,test])
# data = data.fillna(-1)
index = []
for i in data.columns:
    # print(i)
    if data[i].dtypes != 'object':
        index.append(i)

# index.extend(['vid'])
index.extend(['4001','0114','1308','1402','sp','A201','sp_1','2501','sp_2','sp_3','0436','0113','sp_4','sp_5','sp_6','1103','1319'])
data = data[index]
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

data['4001'] = data['4001'].fillna('未查')
data['4001'] = data['4001'].apply(lambda x:' '.join(jieba.cut(x)))
# data['4001'] = data['4001'].apply(get_english)
data['sp'] = data['sp'].fillna('未查')
data['sp'] = data['sp'].apply(lambda x:' '.join(jieba.cut(x)))
data['sp_1'] = data['sp_1'].fillna('未查')
data['sp_1'] = data['sp_1'].apply(lambda x:' '.join(jieba.cut(x)))
data['sp_2'] = data['sp_2'].fillna('未查')
data['sp_2'] = data['sp_2'].apply(lambda x:' '.join(jieba.cut(x)))
data['A201'] = data['A201'].fillna('未查')
data['A201'] = data['A201'].apply(lambda x:' '.join(jieba.cut(x)))
data['0113'] = data['0113'].fillna('未查')
data['0113'] = data['0113'].apply(lambda x:' '.join(jieba.cut(x)))
data['sp_3'] = data['sp_3'].fillna('未查')
data['sp_3'] = data['sp_3'].apply(lambda x:' '.join(jieba.cut(x)))
data['0436'] = data['0436'].fillna('未查')
data['0436'] = data['0436'].apply(lambda x:' '.join(jieba.cut(x)))
data['sp_4'] = data['sp_4'].fillna('未查')
data['sp_4'] = data['sp_4'].apply(lambda x:' '.join(jieba.cut(x)))
data['sp_5'] = data['sp_5'].fillna('未查')
data['sp_5'] = data['sp_5'].apply(lambda x:' '.join(jieba.cut(x)))
data['2501'] = data['2501'].fillna('未查')
data['2501'] = data['2501'].apply(lambda x:' '.join(jieba.cut(x)))
data['sp_6'] = data['sp_6'].fillna('未查')
data['sp_6'] = data['sp_6'].apply(lambda x:' '.join(jieba.cut(x)))
data['1103'] = data['1103'].fillna('未查')
data['1103'] = data['1103'].apply(lambda x:' '.join(jieba.cut(x)))
data['1402'] = data['1402'].fillna('未查')
data['1402'] = data['1402'].apply(lambda x:' '.join(jieba.cut(x)))
data['1319'] = data['1319'].astype(str)
data['1319'] = data['1319'].fillna('未查')
data['1319'] = data['1319'].apply(lambda x:' '.join(jieba.cut(x)))
data['0114'] = data['0114'].astype(str)
data['0114'] = data['0114'].fillna('未查')
data['0114'] = data['0114'].apply(lambda x:' '.join(jieba.cut(x)))
data['1308'] = data['1308'].astype(str)
data['1308'] = data['1308'].fillna('未查')
data['1308'] = data['1308'].apply(lambda x:' '.join(jieba.cut(x)))
tf1 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_1 = tf1.fit_transform(data['4001'])
tf2 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_2 = tf2.fit_transform(data['sp'])
tf3 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_3 = tf3.fit_transform(data['A201'])
tf4 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_4 = tf4.fit_transform(data['sp_1'])
tf5 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_5 = tf5.fit_transform(data['sp_2'])
tf6 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
#全面检查
discuss_tf_6 = tf6.fit_transform(data['sp_3'])
tf7 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
# 肝脏
discuss_tf_7 = tf7.fit_transform(data['0113'])
tf8 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_8 = tf8.fit_transform(data['0436'])
tf9 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_9 = tf9.fit_transform(data['sp_4'])
tf10 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_10 = tf10.fit_transform(data['sp_5'])
tf11 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_11 = tf11.fit_transform(data['2501'])
tf12 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_12 = tf12.fit_transform(data['sp_6'])
tf13 = TfidfVectorizer(ngram_range=(1,2),analyzer='char')
discuss_tf_13 = tf13.fit_transform(data['1103'])
discuss_tf_14 = tf13.fit_transform(data['1402'])
discuss_tf_15 = tf13.fit_transform(data['1319'])
# discuss_tf_16 = tf13.fit_transform(data['0114'])
discuss_tf_17 = tf13.fit_transform(data['1308'])
print('train')
del data['4001']
del data['sp']
del data['sp_1']
del data['sp_2']
del data['A201']
del data['sp_3']
del data['sp_4']
del data['sp_5']
del data['sp_6']
del data['2501']
del data['0113']
del data['0436']
del data['1402']
del data['1103']
del data['1319']
del data['1308']
del data['0114']
cst_v = csr_matrix((data))
all_feat = hstack((cst_v,discuss_tf_1,discuss_tf_2,discuss_tf_3,discuss_tf_4,discuss_tf_5,
                   discuss_tf_6,discuss_tf_7,discuss_tf_8,discuss_tf_9,discuss_tf_10,
                   discuss_tf_11,discuss_tf_12,discuss_tf_13,discuss_tf_14,discuss_tf_15,
                   discuss_tf_17)).tocsr()
# all_feat = cst_v
train = all_feat[:train.shape[0]]
test = all_feat[train.shape[0]:]

from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_log_error,mean_squared_error
def xx(y_true,y_pred):
    # print(len(y_pred))
    return np.sum(np.square(np.log(y_true + 1) - np.log(y_pred + 1))) / len(y_pred)

xxx_1 = []
xxx_2 = []

imp = pd.DataFrame()
# imp['xx'] = list(train.columns)

result = pd.DataFrame()
result['vid'] = list(test_index)
j = 0
train =train



params_set = [
 {'objective': 'regression','metric': {'l2'},'subsample': 0.8835665823899177, 'learning_rate': 0.075172392533426557, 'reg_lambda': 0.081868106223829978, 'min_child_samples': 452, 'min_child_weight': 72, 'subsample_for_bin': 13522, 'max_bin': 574, 'num_leaves': 651, 'scale_pos_weight': 0.029004593634154585, 'subsample_freq': 3, 'reg_alpha': 1.725862317016256e-05, 'colsample_bytree': 0.80155790719110143, 'max_depth': 37}
,
{'objective': 'regression','metric': {'l2'},'reg_lambda': 0.081868106223829978, 'subsample_for_bin': 13522, 'scale_pos_weight': 0.029004593634154585, 'min_child_samples': 452, 'max_bin': 574, 'num_leaves': 651, 'min_child_weight': 72, 'learning_rate': 0.075172392533426557, 'subsample_freq': 3, 'reg_alpha': 1.725862317016256e-05, 'colsample_bytree': 0.80155790719110143, 'subsample': 0.8835665823899177, 'n_estimators': 135, 'max_depth': 37}
,
 { 'verbose': 0,'objective': 'regression','metric': {'l2'},'learning_rate': 0.075172392533426557, 'subsample_for_bin': 13522, 'reg_lambda': 0.081868106223829978, 'max_depth': 37, 'colsample_bytree': 0.80155790719110143, 'scale_pos_weight': 0.029004593634154585, 'min_child_samples': 452, 'min_child_weight': 72, 'num_leaves': 651, 'subsample_freq': 3, 'reg_alpha': 1.725862317016256e-05, 'max_bin': 574,  'subsample': 0.8835665823899177}
,
  {'objective': 'dart','metric': {'l2'},'reg_lambda': 0.081868106223829978, 'max_bin': 574,'subsample_for_bin': 127778,
                  'num_leaves': 651, 'min_child_weight': 72, 'subsample': 0.8835665823899177, 'max_depth': 358,
                  'reg_alpha': 1.725862317016256e-05, 'colsample_bytree': 0.80155790719110143, 'subsample_freq': 3,
                  'min_child_samples': 452, 'scale_pos_weight': 0.029004593634154585,
                  'learning_rate': 0.075172392533426557}
,
{'objective': 'regression','metric': {'l2'},'num_leaves': 65,  'subsample_for_bin': 202222, 'colsample_bytree': 0.80155790719110143, 'max_bin': 574, 'min_child_weight': 7, 'learning_rate': 0.075172392533426557, 'scale_pos_weight': 0.029004593634154585, 'max_depth': 36, 'reg_alpha': 1.5057560255472018e-06, 'min_child_samples': 45, 'subsample_freq': 3, 'subsample': 0.8835665823899177, 'reg_lambda': 0.081868106223829978}

  ,  {'objective': 'regression','metric': {'l2'},'subsample_freq': 3, 'reg_lambda': 0.081868106223829978,
  'max_depth': 36, 'min_child_samples': 45, 'colsample_bytree': 0.80155790719110143,
  'subsample': 0.8835665823899177, 'scale_pos_weight': 0.029004593634154585,
  'max_bin': 574, 'learning_rate': 0.075172392533426557, 'num_leaves': 65,
  'reg_alpha': 1.5057560255472018e-06, 'subsample_for_bin': 202222,
  'min_child_weight': 7}

, {'objective': 'regression','metric': {'l2'},'min_child_samples': 45,  'learning_rate': 0.075172392533426557, 'max_bin': 574, 'min_child_weight': 7, 'reg_alpha': 1.5057560255472018e-06, 'num_leaves': 65, 'subsample_freq': 3, 'subsample_for_bin': 202222, 'subsample': 0.8835665823899177, 'reg_lambda': 0.081868106223829978, 'scale_pos_weight': 0.029004593634154585, 'colsample_bytree': 0.80155790719110143, 'max_depth': 36}
, {'objective': 'regression','metric': {'l2'},'subsample': 0.8835665823899177,  'subsample_freq': 3, 'learning_rate': 0.075172392533426557, 'min_child_samples': 45, 'subsample_for_bin': 202222, 'reg_lambda': 0.081868106223829978, 'max_depth': 36, 'colsample_bytree': 0.80155790719110143, 'num_leaves': 65, 'scale_pos_weight': 0.029004593634154585, 'max_bin': 574, 'min_child_weight': 7, 'reg_alpha': 1.5057560255472018e-06}
, {'objective': 'regression','metric': {'l2'},'reg_alpha': 1.5057560255472018e-06, 'num_leaves': 65, 'max_bin': 574, 'subsample_freq': 3, 'reg_lambda': 0.081868106223829978, 'colsample_bytree': 0.80155790719110143,  'subsample': 0.8835665823899177, 'subsample_for_bin': 202222, 'min_child_samples': 45, 'scale_pos_weight': 0.029004593634154585, 'learning_rate': 0.075172392533426557, 'min_child_weight': 7, 'max_depth': 36}
, {'objective': 'regression','metric': {'l2'},'min_child_weight': 7, 'reg_lambda': 0.081868106223829978, 'subsample_freq': 3, 'colsample_bytree': 0.80155790719110143, 'learning_rate': 0.075172392533426557, 'scale_pos_weight': 0.029004593634154585, 'max_bin': 574, 'max_depth': 36, 'subsample': 0.8835665823899177,  'num_leaves': 65, 'reg_alpha': 1.5057560255472018e-06, 'subsample_for_bin': 202222, 'min_child_samples': 45}


              ]

N = 10
for i in [train_1,train_2,train_3,train_4,train_5]:
# for i in [train_2,train_4,train_5]:

    xxxx_c = []
    xxxx_r = []

    # result = bayes_cv_tuner.fit(train, np.log1p(np.abs(i)), callback=status_print)
    # print(result)
    # exit()
    #
    # if j == 1:
    #     train_copy = hstack((train,csr_matrix(train_1.reshape(-1,1)))).tocsr()
    #     test_copy = hstack(((test,csr_matrix(result[0].values.reshape(-1,1))))).tocsr()
    # else:
    train_copy = train
    test_copy = test

    i = np.log1p(np.abs(i))
    kf = KFold(n_splits=N,random_state=42,shuffle=True)
    print(params_set[j])
    # params_set['objective'] = 'regression'
    # params_set['metric'] = 'l2'
    # if j == 1:
    #     train
    for train_in,test_in in kf.split(train_copy):



        # X_train, X_test, y_train, y_test = train_test_split(train, i, test_size=0.233, random_state=42)
        X_train, X_test, y_train, y_test = train_copy[train_in],train_copy[test_in],i[train_in],i[test_in]

        print((X_train.shape))
        print((test_copy.shape))

    #     # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    #
    #     # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2'},
            'num_leaves': 32,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }

        print('Start training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=20000,
                        valid_sets=lgb_eval,
                        verbose_eval=50,
                        early_stopping_rounds=150)

        # imp[j] = list(gbm.feature_importance())
        print('Save model...')
        # save model to file
        # gbm.save_model('model.t/xt')

        print('Start predicting...')
        # predict
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        y_pred_to = gbm.predict(test_copy, num_iteration=gbm.best_iteration)

        # y_test = np.log1p(np.exp2(y_test)-1)
        # y_pred = np.log1p(np.exp2(y_pred)-1)

        xxxx_c.append(mean_squared_error(y_test, y_pred))
        xxxx_r.append(np.expm1(y_pred_to))
        # xxxx_r.append((y_pred_to))

    s_s = 0
    for s in xxxx_r:
        s_s = s_s + s

    result[j] = list(s_s/N)
    j = j + 1


    # eval
    # xxx_1.append(xx(y_test, y_pred))
    xxx_2.append(xxxx_c)

print(xxx_2)
# print(np.mean(xxx_1))
print(np.mean(xxx_2))

result.to_csv('../submit/baseline_20180505_b_0.0290641925449.csv',index=False,header=False)