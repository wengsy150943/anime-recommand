
#import warnings
#warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


#获取数据
ratings_df = pd.read_csv('real_ratings.csv')
movies_df = pd.read_csv('movies.csv')

userNo = max(ratings_df['userId'])+1
movieNo = max(ratings_df['movieRow'])+1
print(userNo,movieNo)

#创建电影评分表
rating = np.zeros((userNo,movieNo))

for index,row in ratings_df.iterrows():
    rating[int(row['userId']),int(row['movieRow'])]=row['rating']

def recommend(userID,lr,alpha,d,n_iter,data):
    '''
    userID(int):推荐用户ID
    lr(float):学习率
    alpha(float):权重衰减系数
    d(int):矩阵分解因子(即元素个数)
    n_iter(int):训练轮数
    data(ndarray):用户-电影评分矩阵
    ''' 
    #获取用户数与电影数
    m,n = data.shape 
    #初始化参数  
    x = np.random.uniform(0,1,(m,d))
    w = np.random.uniform(0,1,(d,n))
    #创建评分记录表，无评分记为0，有评分记为1
    record = np.array(data>0,dtype=int)
    #梯度下降，更新参数           
    for i in range(n_iter):
        x_grads = np.dot(np.multiply(record,np.dot(x,w)-data),w.T)
        w_grads = np.dot(x.T,np.multiply(record,np.dot(x,w)-data))
        x = alpha*x - lr*x_grads
        w = alpha*w - lr*w_grads
    #预测
    predict = np.dot(x,w)
    #将用户未看过的电影分值从低到高进行排列
    for i in range(n):
        if record[userID-1][i] == 1 :
            predict[userID-1][i] = 0 
    recommend = np.argsort(predict[userID-1])
    a = recommend[-1]
    b = recommend[-2]
    c = recommend[-3]
    d = recommend[-4]
    e = recommend[-5]
    print('为用户%d推荐的电影为：\n1:%s\n2:%s\n3:%s\n4:%s\n5:%s。'\
          %(userID,movies_df['title'][a],movies_df['title'][b],movies_df['title'][c],movies_df['title'][d],movies_df['title'][e]))   

recommend(123,1e-4,0.999,20,100,rating) 