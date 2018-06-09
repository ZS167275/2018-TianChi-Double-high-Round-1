#coding:utf-8
import pandas as pd
import re
import numpy as np

print('wash data')
def get_number(data):
    data = re.findall(r"\-{0,1}\d+\.\d+|\d+",str(data))
    if len(data)>0:
        return data[0]
    else:
        return -1

def get_zh(df):
    df = str(df)
    line = df.strip()  # 处理前进行相关的处理，包括转换成Unicode等
    p2 = re.compile(r'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
    zh = " ".join(p2.split(line)).strip()
    zh = ",".join(zh.split())
    outStr = zh  # 经过相关处理后得到中文的文本
    return outStr

def get_sex(data):
    if str(data) == 'nan':
        return -1
    elif str(data).__contains__('子宫'):
        return 1
    else:
        return 0



data = pd.read_csv('../data/tmp.csv',low_memory=False,encoding='gbk')
print('clean data more nan data')
is_null_data = data.isnull()
clean_radio = is_null_data.sum()
clean_radio = pd.DataFrame(clean_radio) / data.shape[0]
clean_radio = clean_radio.sort_values(0)
clean_radio_col = clean_radio[clean_radio[0]<1]
# data = data[sorted(clean_radio_col.index)]

# exit()

data['0421'] = data['0421'].apply(get_number)
data['0421'] = data['0421'].replace('666666666666666666666666666666666666666666666',6)

data['0425'] = data['0425'].apply(get_number)
data['sp_5'] = data['A705'].copy()
data['A705'] = data['A705'].apply(lambda x:1 if str(x).__contains__('240') else 0)

data['sex'] = data['0120'].fillna(data['0121'])
data['sex'] = data['sex'].apply(get_sex)
data['sp_4'] = data['3601'].copy()
data['3601'] = data['3601'].apply(get_number)
data['3301'] = data['3301'].apply(get_number)
data['0104'] = data['0104'].apply(get_number)
data['0105'] = data['0105'].apply(get_number)
data['0106'] = data['0106'].apply(get_number)
data['0107'] = data['0107'].apply(get_number)
data['0108'] = data['0108'].apply(get_number)
data['0109'] = data['0109'].apply(get_number)
data['3730'] = data['3730'].apply(get_number)
data['310'] = data['310'].apply(get_number)
data['31'] = data['31'].apply(get_number)
data['0111'] = data['0111'].apply(get_number)
data['0112'] = data['0112'].apply(get_number)
data['Y79001'] = data['Y79001'].apply(get_number)

data['0424'] = data['0424'].apply(get_number)
data['300005'] = data['300005'].apply(get_number)
data['3429'] = data['3429'].apply(get_number)
data['0440'] = data['0440'].apply(get_number)
data['0440'] = data['0440'].apply(lambda x:str(x).replace('2004','-1'))
data['sp_6'] = data['1001'].copy()
data['1001_1'] = data['1001'].apply(lambda x:1 if str(x).__contains__('窦性') else 0)
data['1001_2'] = data['1001'].apply(lambda x:1 if str(x).__contains__('T') else 0)
data['1001_3'] = data['1001'].apply(get_number)
del data['1001']
data = pd.concat([data,pd.get_dummies(data['0428'])],axis=1)
data['3430']  = data['3430'].apply(get_number)
del data['3430']
del data['0428']
# exit()
# sp = data[['0439','A705','1319','1320']]
# data = pd.concat([data,pd.get_dummies(data['0439'])],axis=1)
# del data['0439']
# data = pd.concat([data,pd.get_dummies(data['A705'])],axis=1)
# del data['A705']
# data = pd.concat([data,pd.get_dummies(data['1319'])],axis=1)
# del data['1319']
# data = pd.concat([data,pd.get_dummies(data['1320'])],axis=1)
# del data['1320']
def get_numberliver(data):
    data = re.findall(r"脂肪肝（(.+?)）",str(data))
    if len(data)>0:
        return data[0]
    else:
        return '无'


number_list = [
    '004997','1814','190','191','2403','2404','1840','3193','2405','1815','1850','10004','192','1117','193','1115','314','183','2174','10002',
    '183','2174','10002','10003','1321','1322','38','31','319','316','37','315','312','32','100006','313','317','2333',
    '33','2372','1845','320','100005','2406','34','39','100007','269011','269005','269016','269014','269003',
    '269017','269012','269010','269009','269008','269006','269019','1127','269013','269025',
    '269015','269024','269021','269022','269023','269020','300036','269004','269018','155','1345','2420',
    '1325','1326','269007','979012','979016','979021','979003','979004','979017','979007','979015','979008','979009','979011',
    '979023','979022','979020','979001','979013','979014','979018','979019','979002','979006','979005','300021','300019',
    '809013','300014','809010','669021','2409','669004','300001','2376','300013','300012',
    '809021','300008','10009','809023','809025','809026','300092','669009','809017','100013','2177','2386',
    '100012','1112','669005','809004','100014','809009','809008','300017','1474','669006','669001','669002','143',
    '669007','669008','20002','669003','300009','809013','300014','809010','669021','2409','669004','300001',
    '2376','300013','300012','300011','809021','300008','1124','459159','459158','459156','459154','459155','300006',
    '2390','809020','809032','809033','809031','809034','2986','300078','321','809007','30006','A703','809002','809029',
    'A701','2371','300035','809024','1873','189','809027','809003','809022','809019','809018','279006','300007','669007','20002','669008','1320']
# 提取脂肪肝
data['sp_3'] = data['0102'].copy()

data['0102'] = data['0102'].apply(get_numberliver)

data['2302'] = data['2302'].apply(lambda x:get_zh(x))
data['2302'] = data['2302'].apply(lambda x:str(x).replace('健康,健康','健康'))
data['2302'] = data['2302'].apply(lambda x:str(x).replace('健康,正常疲劳反应','健康'))

data['0403'] = data['0403'].apply(lambda x:str(x).replace('正常 正常','正常'))
data['0403'] = data['0403'].apply(lambda x:str(x).replace('心界正常','正常'))
data['0403'] = data['0403'].apply(lambda x:str(x).replace('无扩大','正常'))
data['0403'] = data['0403'].apply(lambda x:str(x).replace('未见异常','正常'))
data['0403'] = data['0403'].apply(lambda x:str(x).replace('无明显扩大','扩大'))
data['0403'] = data['0403'].apply(lambda x:str(x).replace('左侧心界扩大','扩大'))
data['0403'] = data['0403'].apply(lambda x:str(x).replace('不大','扩大'))
# data['0403'] = data['0403'].apply(lambda x:str(x).replace('左侧心界扩大','扩大'))
data = pd.concat([data,pd.get_dummies(data['0403'])],axis=1)
del data['0403']

print('nlp process')
# del data['0113']
# del data['0114']
del data['0116']
del data['0117']
del data['0118']
print('finish')



# data['0424'] = data['0424'].fillna(-1)
# data['0424'] = data['0424'].apply(lambda x:re.sub(re.compile(r"\D"),'',str(x)))
# data['0424'] = data['0424'].apply(lambda x:x[:3] if len(x)>2 else x)

data['3301'] = data['3301'].fillna(-1)
data['3301'] = data['3301'].apply(lambda x:re.sub(re.compile(r"\D"),'',str(x)))

def get_about(data):
    data = re.findall(r"约(.+?)",str(data))
    if len(data) > 0:
        return data[0]
    else:
        return '-1'

def get_about_baby(data):
    data = re.findall(r"如孕(\d+?)",str(data))
    if len(data) > 0:
        return data[0]
    else:
        return '-1'
data['0516'] = data['0516'].apply(get_about_baby)

def calc_number(data):
    print(data)
    data = re.sub(re.compile(r"[\u4e00-\u9fa5,。，、：:CBDCDFInan()（）mmc[/；]"),'',str(data)).strip()
    data = data.replace('x','*')
    if str(data).__contains__('*'):
        tmp_1 = data.split('*')[0]
        tmp_2 = data.split('*')[1]
        print(tmp_2)
        try:
            tmp_1 = float(tmp_1)
            tmp_2 = float(tmp_2)
        except:
            return -1
        if float(tmp_1) > 100:
            tmp_1 = float(tmp_1)%100
        return tmp_1 * float(tmp_2)
    else:
        return data
    # return data


# data['0114'] = data['0114'].apply(get_about)
# data['0114'] = data['0114'].apply(calc_number)
# data['0114'] = data['0114'].apply(lambda x:str(x).strip())

# data['0113'] = data['0113'].apply(lambda x:re.sub(re.compile(r"[\u4e00-\u9fa5,。，、：:CBDCDFInan()（）]"),'',str(x)))
# data['0113'] = data['0113'].apply(lambda x:1 if str(x)=='' else 0)


# exit()
# 提取数字
def clean_number(data):
    try:
        data = str(data).replace('阴性','')
        data = str(data).replace('Hg','')
        data = str(data).replace('mm','')
        data = str(data).replace('mmHg','')
        data = str(data).replace('＝５.０','5.0')
        data = str(data).replace('＜','')
        data = str(data).replace('kpa','')
        data = str(data).replace('db/m','')
        data = str(data).replace('详见报告单','')
        data = str(data).replace('详见纸质报告','')
        data = str(data).replace('+','')
        data = str(data).replace('<','')
        data = str(data).replace('。','.')
        data = str(data).replace('＞＝１.０３０','1.030')
        data = str(data).replace('<=','')
        data = str(data).replace('＜＝５.０','5.0')
        data = str(data).replace('320.00.','320.00')
        data = str(data).replace('20.908.','20.908')

        data = str(data).replace('77..21','77.21')
        data = str(data).replace('16.7.07','16.7')
        data = str(data).replace('12.01','12.01')
        data = str(data).replace('.3.70','3.70')
        data = str(data).replace('4.14.','4.14')
        data = str(data).replace('2.1.','2.1')
        data = str(data).replace('＜0.60','0.6')
        data = str(data).replace('-','')
        data = str(data).replace('6.81.','6.81')
        data = str(data).replace('5..0.','5')
        data = str(data).replace('20.908..','20.908')

        data = str(data).replace('=','')
        data = str(data).replace('>=','')
        data = str(data).replace('>=','')
        data = str(data).replace('>','')
        data = str(data).replace('1.015.','1.015')
        data = str(data).replace('12.01.','12.01')
        data = str(data).replace('5..0','5')
        data = str(data).replace('2.792.20','2.792')
        data = str(data).replace('8.53.','8.53')
        data = str(data).replace('5.10.','5.10')
        data = str(data).replace('14.4.','14.4')
        data = str(data).replace('116.0.','116.0')
        data = str(data).replace('126.0.','126.0')
        data = str(data).replace('10.0.','10')
        data = str(data).replace('2.70.','2.70')
        data = str(data).replace('%','')
        data = str(data).replace('4.3.','4.3')
        data = str(data).replace('6.31.0.45','6.31')
        data = str(data).replace('.45.21','45.21')
        data = str(data).replace('(μIU/ml)','')
        data = str(data).replace('S','')
        data = str(data).replace('(正常值 1222)','')
        data = float(data)
    except Exception:
        if (str(data)=='降脂后复查'):
            return 500
        if (str(data) == '')|(str(data) == '未做')| (str(data) == '/')|(str(data) == '未见')|(str(data) == 'nan')|(str(data) == '未查') | (str(data) == '弃查')|(str(data)=='标本已退检'):
            # print(data)
            return -1
        else:
            data = str(data).split(" ")
            try:
                data[0] = float(data[0])
                return data[0]
            except:
                try:
                    data[1] = float(data[1])
                    return data[1]
                except:
                    print(data)
                    return -2
    return data


del data['0225']

del data['0983']
del data['0981']
del data['0982']
del data['0984']
# del data['0436']
del data['1330']




# for i in clean_radio_col.index:
for i in number_list:
    # 300008
    print('calc %s'%(i))
    data[i] = data[i].apply(clean_number)
del data['0415']
del data['A301']
del data['A302']
del data['0414']
del data['0987']
# del data['0645']
del data['0422']
del data['0124']
del data['0730']
del data['0731']
del data['0715']
del data['0972']
# del data['3601']
del data['1316']
del data['0703']
del data['0705']
del data['0706']
del data['0707']
del data['0702']
del data['0709']
# del data['4001']

del data['0427']
del data['A601']

del data['0115']
# del data['0101']


del data['1304']
del data['0979']
del data['0978']
del data['0954']
# del data['1308']
del data['A202']
# del data['A201']
del data['0222']
del data['0119']
del data['0947']
del data['0949']

# del data['0215']
del data['0209']
# del data['0973']
del data['1315']

# del data['0216']
# del data['0212']
# del data['0210']
del data['0208']
del data['0201']

del data['0202']
del data['0217']
del data['3400']
# 无意义的两个
del data['0980']
del data['0977']
# del data['0201']
# del data['0119']




del data['1328']
# del data['1103']
del data['1102']

del data['1313']
del data['0206']
del data['0120']
# del data['2501']
del data['0973']
# del data['0209']
del data['0215']
del data['0216']
del data['0212']
del data['0210']


del data['0541']
del data['0537']

del data['0509']
del data['0503']
# del data['0503']
# del data['0539']
del data['0539']
del data['0728']
del data['0123']
del data['0122']

del data['0429']

del data['0726']

# del data['1402']
del data['360']



# exit()


# data['0439'] = data['0439'].apply(sorted(list(data['0439'].unique())).index)

for i in ['0424',]:
    pass

data['0121'] = data['0121'].apply(lambda x:1 if str(x)=='nan' else 0)


syb = re.compile(r"[^-|+|*]+")
data['30007'] = data['30007'].fillna(0)
data['30007'] = data['30007'].apply(lambda x:str(x).replace('度',''))
data = pd.concat([data,pd.get_dummies(data['30007'])],axis=1)
del data['30007']
del data['1329']
del data['0501']

for symble in ['2228','3486','2233','2231','2230','2229','3485','3189','3195','3191','3197','3192','3196','3190',
               '100010','3399','1314','3194','300018']:
    data[symble] = data[symble].apply(lambda x: str(x).replace('未做', '*'))
    data[symble] = data[symble].apply(lambda x: str(x).replace('１＋', '+'))
    data[symble] = data[symble].apply(lambda x: str(x).replace('２＋', '++'))
    data[symble] = data[symble].apply(lambda x: str(x).replace('2+', '++'))
    # data[symble] = data[symble].applu(lambda x: str(x).replace('0(+', '0'))
    # data[symble] = data[symble].applu(lambda x: str(x).replace('5(+', '0'))
    data[symble] = data[symble].apply(lambda x: str(x).replace('3+', '+++'))
    data[symble] = data[symble].apply(lambda x: str(x).replace('阴性', '-'))
    data[symble] = data[symble].apply(lambda x: str(x).replace('正常', '-'))
    data[symble] = data[symble].apply(lambda x: str(x).lower().replace('normal', '-'))
    # data[symble] = data[symble].apply(lambda x: str(x).replace('NormaL', '-'))
    data[symble] = data[symble].apply(lambda x:re.sub(syb,'',str(x)))
    data[symble] = data[symble].apply(sorted(list(data[symble].unique())).index)
    # data[symble] = data[symble].fillna('*')
    # data[symble] = data[symble].apply(lambda x:re.sub(re.compile(r"\d(\+)"),'',str(x)))
    data = pd.concat([data, pd.get_dummies(data[symble], prefix=symble)], axis=1)
    del data[symble]






def get_history(data):
    if str(data).__contains__('内科检查未发现明显异常'):
        return 1
    elif (str(data).__contains__('血压')):
        return 2
    elif (str(data).__contains__('血糖'))|(str(data).__contains__('糖')):
        return 3
    elif (str(data).__contains__('血脂'))|(str(data).__contains__('肝')):
        return 4
    elif (str(data).__contains__('心')):
        return 5
    else:
        return 6

data['sp_2'] = data['0434'].copy()
data['0434'] = data['0434'].apply(get_history)
# data['0439'] = data['0439'].apply(get_history)
data['sp'] = data['0409'].copy()
data['0409'] = data['0409'].apply(get_history)
data['0439'] = data['0439'].apply(get_history)

data = pd.concat([data,pd.get_dummies(data['0409'],prefix='history')],axis=1)
data = pd.concat([data,pd.get_dummies(data['0102'],prefix='0102')],axis=1)
data = pd.concat([data,pd.get_dummies(data['2302'],prefix='2302')],axis=1)
data = pd.concat([data,pd.get_dummies(data['0439'],prefix='0439')],axis=1)
data = pd.concat([data,pd.get_dummies(data['0434'],prefix='0434')],axis=1)
del data['0409']
del data['0102']
del data['2302']
del data['0434']

def set_sp(data):
    if str(data).__contains__('未见异常'):
        return 1
    elif (str(data).__contains__('未闻')):
        return 1
    else:
        return 2
data['sp_1'] = data['0426'].copy()
data['0426'] = data['0426'].apply(set_sp)
data = pd.concat([data,pd.get_dummies(data['0426'],prefix='0426')],axis=1)
del data['0426']



# 修改训练和测试数据
train = pd.read_csv('../data/train.csv',encoding='gbk')
train = pd.merge(train,data,on=['vid'],how='left')


test = pd.read_csv('../data/meinian_round1_test_b_20180505.csv',encoding='gbk')
test = pd.merge(test,data,on=['vid'],how='left')


train.to_csv('../data/train_csv.csv',index=False)
test.to_csv('../data/test_b_csv.csv',index=False)
