'''
function:
description:
'''
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing as pp
from matplotlib.font_manager import FontProperties
import sys,os
import numpy as np
#import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import classification_report
from collections import defaultdict
import traceback
import json
import datetime
import xgboost as xgb


#get_ipython().magic('matplotlib inline')

BASEDIR_1 = r"D:\bond_fault\bond\bond_data"
BASEDIR_2 = r'/bond_fault/bond/bond_data'
BASEDIR_3 = r'/home/zean/bond_risk/data_wash'
CODE_STYLE = r'yahei.ttf'
BASEDIR = BASEDIR_3

# Initial the parameter  

class Times_tooles(object):
    def __init__(self):
        self.desc = "interface to handle times"
        
    def get_day(self, innum):
        '''
        chage num to day, which coule be calculate with dates
        '''
        days = datetime.timedelta(days=innum)
        return days
# ===================================================
'''
'''    
# ===================================================
class Model_bond_fault(object):

    def __init__(self):
        self.init_char()
        self.company_df = None
        self.bond_df = None
        self.level_df = None
        self.bond_halfyear = None
        self.bond_year = None
        self.bond_quarter_one = None
        self.bond_quarter_thr = None
        self.public_sentiment_df = None
        BASEDIR = BASEDIR_1
        print("[x] Initial Model_bond_fault")
        

    def init_char(self):
        '''
        初始化字符集
        '''
        pass
        #myfont=FontProperties(fname=r'C:\Users\Administrator\Downloads\yahei.ttf', size=14)
        #myfont=FontProperties(fname='/usr/share/fonts/truetype/yahei.ttf',size=14)
        mpl.rcParams['axes.unicode_minus']=False # 【笔记】解决中文坐标显示乱码问题
        #sns.set(font=myfont.get_name())

    def path_join(self, pre_path,after_path):
        '''
        生成路径名
        '''
        path = os.path.join(pre_path,after_path)
        return path

    def get_file(self, name, codec='utf-8'):
        '''
        获得文件
        '''        
        file_path = path_join(BASE_PATH, name)
        f = codecs.open(file_path, 'r', codec)
        cont = f.read()
        return cont

    def wash_data_public_sentiment(self, input_df):
            if not pd.isnull(input_df):
                char_lst = []
                words = input_df.split(',')
                for j in words:
                    chars = j.split(r"->")
                    try:
                        char_lst.append(chars[-2]+chars[-1])
                    except:
                        continue
                i = char_lst
#                i = re.sub('["\'","\]","\["]','',i)
#                print(i)
                return ','.join(i)[:-1]

    def dateparse_2(self, dates):
        try:
            return (pd.datetime.strptime(str(dates)[0:10], '%Y-%m-%d'))
        except:
            return ""
        
    def init_data(self):
        '''
        数据准备
        '''
        print ("[x] Init init_data")
        '''
        | obj        | file
        ----------------------------------
        | company_df | "company_info.csv"
        | bond_df"   | "a.csv"
        | level_df   | "level.csv"
        '''
        
        dateparse = lambda dates: pd.datetime.strptime((str(dates)+'0101')[0:8], '%Y%m%d') if not pd.isnull(dates) else None
        self.company_df = pd.read_csv(os.path.join(BASEDIR, "company_info.csv"),
                         parse_dates=['成立日期'],
                         infer_datetime_format = True,
                         date_parser=dateparse)
        #print(company_df)
        self.bond_df = pd.read_csv(os.path.join(BASEDIR, "a.csv"),
                      parse_dates=['报告期'],
                      infer_datetime_format = True,
                      date_parser=dateparse)
        #print(bond_df)

        self.level_df = pd.read_csv(os.path.join(BASEDIR, "level.csv"),
                       parse_dates=['评级日期','公告日期'],
                       infer_datetime_format=True,
                       date_parser=dateparse)

#        dateparse_2 = lambda dates: pd.datetime.strptime((str(dates)+'0101')[0:8], '%Y%m%d') if not pd.isnull(dates) else None
        self.public_sentiment_df = pd.read_csv(os.path.join(BASEDIR, "public_sentiment.csv"),
                       parse_dates=['與情日期'],
                       infer_datetime_format=True,
                       date_parser=self.dateparse_2)
        
        #        dateparse_2 = lambda dates: pd.datetime.strptime((str(dates)+'0101')[0:8], '%Y%m%d') if not pd.isnull(dates) else None
        self.bond_fault_df = pd.read_csv(os.path.join(BASEDIR, "bond_fault.csv"),
                       parse_dates=['发生日期'],
                       infer_datetime_format=True,
                       date_parser=dateparse)


        self.public_sentiment_df['lable_type'] = self.public_sentiment_df['lable_type'].apply(lambda x: md.wash_data_public_sentiment(x))
#        self.public_sentiment_df = md.wash_data_public_sentiment(self.public_sentiment_df)

        self.bond_df['报告类型'] = self.bond_df['报告期'].apply(lambda x: '年报' if x.month==12 else ('半年报' if x.month==6 else ('一季报' if x.month==314 else '三季报')))
        company = pd.merge(self.company_df, self.level_df, how='inner', left_on='公司名称', right_on='公司中文名称')

        # 常规评级、穆迪评级，量化为分数
        mudy_level = {"Aaa":100,"Aa1":95,"Aa2":90,"Aa3":85,"A1":80,"A2":75,"A3":70,"Baa1":65,"Baa2":60,\
                     "Baa3":55, "Ba1":50,"Ba2":45,"Ba3":40,"B1":35,"B2":30,"B3":25,"Caa1":20,"Caa2":15,\
                     "Caa3":10,"Ca":5,"C":0}
        norm_level = {"AAA":100,"AA+":95,"AA":90,"AA-":85,"A+":80,"A":75,"A-":70,"BBB+":65,"BBB":60,\
                     "BBB-":55, "BB+":50,"BB":45,"BB-":40,"B+":35,"B":30,"B-":25,"CCC":20,"CC":15,\
                     "C":10, "RD":5,"D":0}
        company['信用评级分数']=company['信用评级'].apply(lambda x: norm_level[x] if x in norm_level else(mudy_level[x]))
#        print(company['成立日期'])        
        self.bond_df['报告类型'] = self.bond_df['报告期'].apply(lambda x: '年报' if x.month==12 else ('半年报' if x.month==6 else ('一季报' if x.month==3 else '三季报')))
        company['企业年龄']=company['成立日期'].apply(lambda x: (2017 - x.year))
        company = company.drop(['公司中文简介', '信用评级说明', '法人代表', '总经理', \
                    '董事会秘书', '主页', '电子邮箱', '经营范围', '对象ID', '公司ID', \
                    '公司中文名称', '主要产品及业务', '办公地址', '评级展望', '债券主体公司id',\
                     '前次信用评级', '评级变动方向', '城市'], axis=1)

        # 设置列类型  
        company[['公司类别','评级类型','评级机构代码']] = company[['公司类别','评级类型','评级机构代码']].astype(str)
        company['员工总数人'] = company[company['员工总数人'].notnull()]['员工总数人'].astype(int)

        print('[x] 第一次 合并数据 merge')

        # 将company 和 bond_df 绑定为一个表，根据报告类型分为 年 半年 季
        self.bond_year = pd.merge(self.bond_df[self.bond_df['报告类型']=='年报'], company, how='inner', on='公司名称')
        self.bond_halfyear = pd.merge(self.bond_df[self.bond_df['报告类型']=='半年报'], company, how='inner', on='公司名称')
#        bond_quarter = pd.merge(bond_df[bond_df['报告类型']=='季报'], company, how='inner', on='公司名称')
        self.bond_quarter_one = pd.merge(self.bond_df[self.bond_df['报告类型']=='一季报'], company, how='inner', on='公司名称')
        self.bond_quarter_thr = pd.merge(self.bond_df[self.bond_df['报告类型']=='三季报'], company, how='inner', on='公司名称')
#        print(self.bond_quarter_thr)

        # print(type(self.public_sentiment_df))
        # print(self.public_sentiment_df)
        print('[x] 第二次 合并数据 merge')
        self.bond_year = pd.merge(self.bond_year, self.public_sentiment_df, how='inner', left_on=['公司名称','公告日期'],right_on=['enterprise_name','與情日期'])
        #self.bond_halfyear = pd.merge(self.bond_halfyear, self.public_sentiment_df, how='inner', left_on='公司名称',right_on='enterprise_name')
        #self.bond_quarter_thr = pd.merge(self.bond_quarter_thr, self.public_sentiment_df, how='inner', left_on='公司名称',right_on='enterprise_name')
        #self.bond_quarter_one = pd.merge(self.bond_quarter_one, self.public_sentiment_df, how='inner', left_on='公司名称',right_on='enterprise_name')
        print('[x] 数据初始化完毕')
#        print(bond_year_2)
        self.bond_year['舆情日期'] = self.bond_year['评级日期'].apply(lambda x : self.calcu_date(x,30))
        self.bond_year = self.groupby_sentiment(self.bond_year)
        #self.bond_halfyear = self.groupby_sentiment(self.bond_halfyear)
        #self.bond_quarter_thr = self.groupby_sentiment(self.bond_quarter_thr)
        #self.bond_quarter_one = self.groupby_sentiment(self.bond_quarter_one)
#        self.bond_year['舆情日期vs报告'] = self.bond_year['报告期'].apply(lambda x : self.calcu_date(x,30))

        self.bond_year = pd.merge(self.bond_year, self.bond_fault_df, how='left', left_on=['公司名称'],right_on=['发行人'])
        self.bond_halfyear = pd.merge(self.bond_halfyear, self.bond_fault_df, how='left', left_on=['公司名称'],right_on=['发行人'])
        self.bond_quarter_thr = pd.merge(self.bond_quarter_thr, self.bond_fault_df, how='left', left_on=['公司名称'],right_on=['发行人'])
        self.bond_quarter_one = pd.merge(self.bond_quarter_one, self.bond_fault_df, how='left', left_on=['公司名称'],right_on=['发行人'])
#        print(self.bond_year)


    def process_data(self, bond):
        # 删除评级日期<报告期的记录+一个月滞后期(暂时没有实现)
        # 按公司名称，报告期分组排序
        bond['sort_id'] =  bond.loc[bond['报告期']<bond['评级日期'],['评级日期']].groupby([bond['公司名称'], bond['报告期']]).rank(ascending=False)
        #    print( bond.loc[bond['报告期']<bond['评级日期'],['评级日期']])
        #    for i in bond.loc[bond['报告期']<bond['评级日期'],['评级日期']].groupby([bond['公司名称'], bond['报告期']]):
        #        print(i)
        #    print(bond.loc[bond['sort_id']>7,['sort_id']])

        fin_columns = ['公司名称','报告期','基本每股收益','每股未分配利润','每股净资产','净资产收益率',
                       '总资产报酬率','总资产净利率（杜邦分析）','经营活动净收益/利润总额','营业外收支净额/利润总额',
                       '销售商品提供劳务收到的现金/营业收入','经营活动产生的现金流量净额/营业收入','资产负债率',
                       '权益乘数(用于杜邦分析) ','流动资产/总资产','归属于母公司的股东权益/全部投入资本','流动比率',
                       '速动比率','产权比率','经营活动产生的现金流量净额/负债合计','存货周转率','应收账款周转率',
                       '总资产周转率','同比增长率-基本每股收益(%)','同比增长率-营业利润(%)',
                       '同比增长率-经营活动产生的现金流量净额(%)']
        # 评级前前期财务指标
        bond_2 = bond.loc[bond['sort_id'] == 2, fin_columns]
        #print(bond_2)
        # 评级前前前期财务指标
        bond_3 = bond.loc[bond['sort_id'] == 3, fin_columns]
        #print(bond_3)
        # 只保留评级日期和报告期最接近的一条记录
        bond_tmp = bond[bond['sort_id'] == 1]
        # 合并评级日期前两期财务指标
        bond_tmp = pd.merge(bond_tmp, bond_2, how='left', on=['公司名称','报告期'])
        '''
        【笔记】axis 1 列， 0 行
        #    print(bond_tmp.drop(['公司代码', '公司名称'], axis=1))
        '''
        return bond_tmp.drop(['公司代码', '公司名称'], axis=1)

        # 数值型字段标准化        
        
    def data_normal(self, bond):
        columns_normal = [
            '基本每股收益',
            '每股未分配利润',
            '每股净资产',
            '净资产收益率',
            '总资产报酬率',
            '总资产净利率（杜邦分析）',
            '经营活动净收益/利润总额',
            '营业外收支净额/利润总额',
            '销售商品提供劳务收到的现金/营业收入',
            '经营活动产生的现金流量净额/营业收入',
            '资产负债率',
            '权益乘数(用于杜邦分析) ',
            '流动资产/总资产',
            '归属于母公司的股东权益/全部投入资本',
            '流动比率',
            '速动比率',
            '产权比率',
            '经营活动产生的现金流量净额/负债合计',
            '存货周转率','应收账款周转率',
            '总资产周转率',
            '同比增长率-基本每股收益(%)',
            '同比增长率-营业利润(%)',
            '同比增长率-经营活动产生的现金流量净额(%)',
            '员工总数人',
            '信用评级分数',
            '注册资本万元',
        ]
        
        for x in columns_normal:
            bond[x] = bond[x].transform(lambda x: (x - x.mean()) / x.std())
        
            
        '''
        【笔记】
        Z-score 将偏离中心概率的值量化表示
        F-score 落入范围的概率
        增加信用评级分数
        该处增加企业年龄
        '''
        # data_normal(bond_year)
        
    def get_day(self, innum):
        days = datetime.timedelta(days=innum)
        return days
    
    def calcu_date(self, x, num):
#        print((x - self.get_day(num)))
        return (x - self.get_day(num)) 
    
    def count_label(self,bond):
        label_dict={}
        label_lst = []
        label_set = []
        for item in bond.loc[:,'lable_type']:
            item = re.sub('\'','',item)
            for j in item.split(','):
                label_lst.append(j)
#        print(list(set(label_lst)))
        label_set = list(set(label_lst))
#        label_set = list(set(label_lst))
        for i in label_set:
#            print(i)
            my_count = label_lst.count(i)
            label_dict[i] = my_count
#            label_dict.keys()
            one_hot=pd.get_dummies(list(label_dict.keys()))
#            print(type(one_hot))
        return[one_hot, label_dict]  
    
    def calcu_label(self, bond):
        print('[x] calcu_label')
#        print(bond)
        bond_tmp = bond.copy()
#        print(bond_tmp)
        for i in bond_tmp.index:
            itm = bond_tmp.loc[i]
#            print(itm)
            startdate = itm['评级日期']
            enddate = itm['舆情日期']
            print('')
 #           print(startdate,enddate)
            bond_tmp_tmp = bond_tmp[bond_tmp["pub_date"]>startdate & bond_tmp["pub_date"]<enddate]
#            print(bond_tmp_tmp)
#            lable_type = bond_tmp.iloc[:,'lable_type']
  #          print(lable_type)
            
        '''
#        for i in bond_tmp.iloc[:,:]:
#            pass 
#            print(type(i))
#            print(bond_tmp.iloc[i,:])
        
            startdate = bond_tmp.loc[i]['舆情日期']
            enddate = bond_tmp.loc[i]['评级日期']
            bond_tmp = bond_tmp.loc[bond_tmp["pub_date"]>startdate & bond_tmp["pub_date"]<enddate]
            for i in bond_tmp.loc[:,'lable_type']:
                print('读取lable_type')
                print(i)
        '''
        
#        print(bond_tmp)
        
        '''
        label_dict = {}
        bond.loc
        x['评级日期'] = x['评级日期'] - self.get_day(num)
        bond_month_ago = bond[x['评级日期']<bond['pub_date']]
        bond_month_ago = bond_month_ago[bond_month_ago['pub_date']<x['评级日期']]
        bond_month_ago = bond_month_ago[bond_month_ago['公司名称']==x['公司名称']]
#        group_bond = bond_month_ago.groupby('lable_type')
        label_lst = []
        label_set = []
        for item in bond.loc[:,'lable_type']:
            item = re.sub('\'','',item)
            for j in item.split(','):
                label_lst.append(j) 
        label_set = list(set(label_lst))
        for i in label_set:
#            print(i)
            my_count = label_lst.count(i)
            label_dict[i] = my_count
        return(str(label_dict))
        '''        
    
    def groupby_sentiment(self, inbond):
        bond = inbond.copy()
        [onehot, label_count] = self.count_label(inbond)
        for i in label_count.keys():
            bond[i] = 0
#        bond['label_sum'] = bond[['评级日期','lable_type','舆情日期']].apply(lambda x : self.calcu_label(x))       
        for i in bond.index:
#            print("[x] 以此打印index i = ",i)
            line = bond.iloc[i,:]
#            print(line)
            sta = line['舆情日期']
#            print(sta)
            end = line['评级日期']
#            print(end)
            bond_tmp = bond[(bond['pub_date']> sta) & (bond['pub_date']< end)]
            lable_type = bond_tmp.loc[:,'lable_type']
            lst = []
            for ii in lable_type:
                for iii in ii.split(','):
                    lst.append(iii)
 #           print('[x] groupby_sentiment')
#            print(lst)            
            ser_lst =[]
            part_count = pd.Series(lst).value_counts()
#            print('* 将一类舆情标签 和 数量作为一个columns写入df')
            for j in part_count.keys():
#                print('写入新值', i, j, part_count[j])
                bond.ix[i,j] = part_count[j]
#                ser_lst.append(onehot[i].append(pd.Series()))
#            print('* 打印第一个整理好的舆情标签')
#            print(pd.DataFrame(ser_lst))
#            ls = pd.DataFrame(ser_lst)
#            print(ls)
#            line['label_sum'] = ls
        return bond
            
            
            
            
 #       return bond
        
    def encode_feature_base(self, bond):
        '''
        对信用评级的评分，前述已完成，此方法暂时未使用
        '''
        bond_tmp = bond.copy()
        # 合并B,C分类, 并取消+/-微调
        bond_tmp['信用评级'] = bond_tmp['信用评级'].apply(lambda x: x[0] if x[0].upper() in ['B', 'C'] else (x[0:-1] if x[-1] in ['+','-'] else x)) 
        # 显示各分类的数量
        #sns.factorplot(y='信用评级', data=bond_tmp, kind='count', size=7)
        # 对信用评级进行编码
        le = pp.LabelEncoder()
        le.fit(bond_tmp['信用评级'])
        bond_tmp['信用评级'] = le.transform(bond_tmp['信用评级'])
        #    print(bond_tmp['信用评级'])

        # 对类型字段进行onehot编码
        bond_tmp['lable_type']
        bond_tmp=pd.get_dummies(bond_tmp)
        self.count_label(bond_tmp)

        # bond shuffle
        return bond_tmp.sample(frac=1).reset_index(drop=True), le.classes_
        #encode_feature(bond_year)

    def encode_feature(self, bond):
        '''
        对信用评级的评分，前述已完成，此方法暂时未使用
        '''
        bond_tmp = bond.copy()
        # 合并B,C分类, 并取消+/-微调
        bond_tmp['信用评级'] = bond_tmp['信用评级'].apply(lambda x: x[0] if x[0].upper() in ['B', 'C'] else (x[0:-1] if x[-1] in ['+','-'] else x)) 
        # 显示各分类的数量
        # sns.factorplot(y='信用评级', data=bond_tmp, kind='count', size=7)
        # 对信用评级进行编码
        le = pp.LabelEncoder()
        le.fit(bond_tmp['信用评级'])
        bond_tmp['信用评级'] = le.transform(bond_tmp['信用评级'])
        bond_tmp=pd.get_dummies(bond_tmp)

        # bond shuffle
        return bond_tmp.sample(frac=1).reset_index(drop=True), le.classes_
        #encode_feature(bond_year)

    # 训练模型
    def run(self, data, label, num_round=2000):
        '''
        【笔记】
        导入data label 的 columns 是什么？

        '''
        sz = data.shape
        cid = data.shape[1]-1
        train = data.iloc[:int(sz[0] * 0.7), :]
        test = data.iloc[int(sz[0] * 0.7):, :]
        train_X = train.drop(['信用评级','成立日期','评级日期','公告日期','报告期'], axis=1).values
        train_Y = train['信用评级'].values
        test_X = test.drop(['信用评级','成立日期','评级日期','公告日期','报告期'], axis=1).values
        test_Y = test['信用评级'].values

        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)
        '''
        【笔记】转换成xgb数据格式
        
        配置xgb参数
        通用 参数
        boost 参数
        学习目标参数
        '''
        param = {
            'objective': 'multi:softmax', #目标函数
            'eval_metric': 'mlogloss',# 评估函数
            'gamma': 0.1, # 裕度
            'max_depth': 20,# 树形深度 default 6 越大越拟合
            'lambda': 10, # L2正则 惩罚项
            'subsample': 0.7,# 采样样本、总样本比
            'colsample_bytree': 0.7,  # 在建立树时对特征随机采样的比例
            'colsample_bylevel': 0.7, # 决定每次节点划分时子样例的比例
            'eta': 0.01, # 收缩补偿 0.01~0.2
            'tree_method': 'exact',
            'seed': 0,# 伪随机数种子
            'scale_pos_weight': 5, # 样本不均衡
            'num_class': 5  # 几分类
        }

        watchlist = [(xg_train,'train'), (xg_test, 'test')]
        bst = xgb.train(param, xg_train, num_round, watchlist )
        # 保存训练模型
        bst.save_model('./bond_data/xgb.model')
        # 计算分类预测准确性
        pred = bst.predict(xg_test)
        # 打印评估结果

        print(classification_report(test_Y, pred, target_names=label))

    # 计算特征权重
    def calc_fscore(self, model_file='./bond_data/xgb.model'):
        import operator
        bst = xgb.Booster()
        bst.load_model(model_file)
        importance = bst.get_fscore('./bond_data/xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore']).sort_values(by='fscore')
        print(df)

    # 保存特征列表
    def create_feature_map(self, features):
        with open('./bond_data/xgb.fmap', 'w', encoding='utf-8') as outfile:
            for i, feat in enumerate(features):
                outfile.write('{0}\t{1}\tq\n'.format(i, feat))

    # 测试结果，更多的特征有利于模型准确性提升
    # 查看各特征相关性
    def show_corr(self, bond):
        colormap = plt.cm.viridis
        plt.figure(figsize=(15,15))
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        sns.heatmap(bond.iloc[:,:-1].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

    def ana_model_pred(self):
        # 归一化
        self.data_normal(self.bond_year)
#        self.data_normal(self.bond_halfyear)
#        self.data_normal(self.bond_quarter_one)
#        self.data_normal(self.bond_quarter_thr)

        # 数据关联
        b_year = self.process_data(self.bond_year)
#        b_halfyear = self.process_data(self.bond_halfyear)
#        b_quarter_one = self.process_data(self.bond_quarter_one)
#        b_quarter_thr = self.process_data(self.bond_quarter_thr)


        # 查看数值列分布
        print(b_year.describe())
        # 查看特征数据量
        print(b_year.info())


        # 是否存在同一公司的多条记录？
        # 是否省份，地市会提高准确率？
        b, label =self.encode_feature(b_year)
        #self.show_corr(b.iloc[:,:10])
        self.run(b, label, 5000)
        self.create_feature_map(b.columns)
        # calc_fscore()

        '''
        # 是否省份，地市会提高准确率？
        b, label = self.encode_feature(b_halfyear)
        #self.show_corr(b.iloc[:,:10])
        self.run(b, label, 5000)
        self.create_feature_map(b.columns)
        # calc_fscore()
        # 是否省份，地市会提高准确率？
        b, label =self.encode_feature(b_quarter_one)
        #self.show_corr(b.iloc[:,:10])
        self.run(b, label, 5000)
        self.create_feature_map(b.columns)
        
        b, label =self.encode_feature(b_quarter_thr)
        #self.show_corr(b.iloc[:,:10])
        self.run(b, label, 5000)
        self.create_feature_map(b.columns)
        '''

        
    def wr2csv(self, df_conf, file_name):
        import csv
        df_conf.to_csv(file_name, index=True, sep=',',encoding = "utf-8")  

def read_clear_file():
    out_put = {}

    company_df = pd.read_csv(os.path.join(BASEDIR, "company_df.csv"),
                     parse_dates=['成立日期'],
                     infer_datetime_format = True,
                     date_parser=dateparse_2)
    #print(company_df)
    bond_df = pd.read_csv(os.path.join(BASEDIR, "bond_df.csv"),
                  parse_dates=['报告期'],
                  infer_datetime_format = True,
                  date_parser=dateparse_2)
    #print(bond_df)

    level_df = pd.read_csv(os.path.join(BASEDIR, "level_df.csv"),
                   parse_dates=['评级日期','公告日期'],
                   infer_datetime_format=True,
                   date_parser=dateparse_2)

#        dateparse_2 = lambda dates: pd.datetime.strptime((str(dates)+'0101')[0:8], '%Y%m%d') if not pd.isnull(dates) else None
    public_sentiment_df = pd.read_csv(os.path.join(BASEDIR, "public_sentiment_df.csv"),
                   parse_dates=['舆情日期'],
                   infer_datetime_format=True,
                   date_parser=dateparse_2)
    
    #        dateparse_2 = lambda dates: pd.datetime.strptime((str(dates)+'0101')[0:8], '%Y%m%d') if not pd.isnull(dates) else None
    bond_fault_df = pd.read_csv(os.path.join(BASEDIR, "bond_fault_df.csv"),
                   parse_dates=['发生日期'],
                   infer_datetime_format=True,
                   date_parser=dateparse_2)

    out_put['company_df'] = company_df
    out_put['bond_df'] = bond_df
    out_put['level_df'] = level_df
    out_put['public_sentiment_df'] = public_sentiment_df
    out_put['bond_fault_df'] = bond_fault_df
    return out_put

def merge_all():
    pass
    
def dateparse_2(dates):
    try:
        return (pd.datetime.strptime(str(dates)[0:10], '%Y-%m-%d'))
    except:
        return None


class Trade_risk_module():
    def __init__(self):
        pass

    def get_data(self):
        data = {}
        data['pred'] = pd.read_csv('/home/zean/Downloads/train.csv')
        data['train'] = pd.read_csv('/home/zean/Downloads/train.csv')
        train = data['train'].sample(frac=0.3)
        test = data['train'].sample(frac=0.3)
        test = test.drop_duplicates(train)

        train_X = train.iloc[1:,:-1]
        train_Y = train.iloc[1:,-1]
        test_X = test.iloc[1:,:-1]
        test_Y = test.iloc[1:,-1]
        pred_X = data['pred'].iloc[1:,:-1]
        pred_Y = data['pred'].iloc[1:,-1]
        return [train_X, train_Y, test_X, test_Y, pred_X, pred_Y]

    def run(self, train_X, train_Y,test_X, test_Y, pred_X, pred_Y, num_round=5000):

        xg_train = xgb.DMatrix(train_X, label=train_Y)
        xg_test = xgb.DMatrix(test_X, label=test_Y)
        param = {
            'objective': 'multi:softmax', #目标函数
            'eval_metric': 'mlogloss',# 评估函数
            'gamma': 0.1, # 裕度
            'max_depth': 20,# 树形深度 default 6 越大越拟合
            'lambda': 10, # L2正则 惩罚项
            'subsample': 0.7,# 采样样本、总样本比
            'colsample_bytree': 0.7,  # 在建立树时对特征随机采样的比例
            'colsample_bylevel': 0.7, # 决定每次节点划分时子样例的比例
            'eta': 0.01, # 收缩补偿 0.01~0.2
            'tree_method': 'exact',
            'seed': 0,# 伪随机数种子
            'scale_pos_weight': 5, # 样本不均衡
            'num_class': 2  # 几分类
        }

        watchlist = [(xg_train,'train'), (xg_test, 'test')]
        bst = xgb.train(param, xg_train, num_round, watchlist )
        # 保存训练模型
        bst.save_model('/home/zean/bond_risk/bond_data/xgb.model')
        # 计算分类预测准确性
        pred = bst.predict(xg_test)
        # 打印评估结果
        print(classification_report(test_Y, pred, target_names="01"))

    def show_corr(self, bond):
        colormap = plt.cm.viridis
        plt.figure(figsize=(15,15))
        plt.title('Pearson Correlation of Features', y=1.05, size=15)
        sns.heatmap(bond.iloc[:,:-1].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


if __name__ == '__main__':
    
     try:
        mod = Trade_risk_module()
        dLst = mod.get_data()

        mod.show_corr(dLst[0].iloc[:,:-1])

        mod.run(dLst[0],dLst[1],dLst[2],dLst[3],dLst[4],dLst[5])
        mod.create_feature_map(dLst.columns)
        # output = read_clear_file()
        # for i in output.keys():
        #     print(output[i].columns)
        
#         print('[x] START ===')
#         md = Model_bond_fault()
#         md.init_data()
#         print('[x] 数据处理完毕')
# #        print(md.bond_year)
# #        print(md.bond_year)
#         md.ana_model_pred()
#     #    md.wr2csv(md.bond_df, 'bond_df.csv')
#     #    md.wr2csv(md.level_df, 'level_df.csv')
#     #    md.wr2csv(md.company_df, 'company_df.csv')
#     #    md.wr2csv(md.bond_halfyear, 'bond_halfyear.csv')
#     #    md.wr2csv(md.bond_quarter_one, 'bond_quarter_one.csv')
#     #    md.wr2csv(md.bond_quarter_thr, 'bond_quarter_thr.csv')
#         md.wr2csv(md.bond_year, 'bond_year.csv')
    
     except :#Exception, e:
         traceback.print_exc()


