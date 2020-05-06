import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='SimHei', size=13)

import os,gc,re,warnings,sys
warnings.filterwarnings("ignore")

path = 'data/'

##### train
train_user_df = pd.read_csv(path+'underexpose_train/underexpose_user_feat.csv', names=['user_id','user_age_level','user_gender','user_city_level'])
train_item_df = pd.read_csv(path+'underexpose_train/underexpose_item_feat.csv')
train_click_0_df = pd.read_csv(path+'underexpose_train/underexpose_train_click-0.csv',names=['user_id','item_id','time'])
# print(len(train_click_0_df['user_id']),len(train_click_0_df['user_id'].unique()),len(train_click_0_df['item_id'].unique()))
# exit(0)

##### test
test_qtime_0_df = pd.read_csv(path+'underexpose_test/underexpose_test_qtime-0.csv', names=['user_id','query_time'])
test_click_0_df = pd.read_csv(path+'underexpose_test/underexpose_test_click-0.csv', names=['user_id','item_id','time'])

train_item_df.columns = ['item_id'] + ['txt_vec'+str(i) for i in range(128)] + ['img_vec'+str(i) for i in range(128)]
train_item_df['txt_vec0'] = train_item_df['txt_vec0'].apply(lambda x:float(x[1:]))
train_item_df['txt_vec127'] = train_item_df['txt_vec127'].apply(lambda x:float(x[:-1]))
train_item_df['img_vec0'] = train_item_df['img_vec0'].apply(lambda x:float(x[1:]))
train_item_df['img_vec127'] = train_item_df['img_vec127'].apply(lambda x:float(x[:-1]))

#print(len(train_item_df['item_id'].unique()))
#print(len(train_item_df['item_id']))

#exit(0)

def rank_and_count():
    train_click_0_df['rank'] = train_click_0_df.groupby(['user_id'])['time'].rank(ascending=False).astype(int)
    test_click_0_df['rank'] = test_click_0_df.groupby(['user_id'])['time'].rank(ascending=False).astype(int)

    # click cnts
    # print(train_click_0_df.groupby(['user_id']).transform('count'))
    # print(train_click_0_df.groupby(['user_id'])['time'].transform('count'))

    train_click_0_df['click_cnts'] = train_click_0_df.groupby(['user_id'])['time'].transform('count')
    test_click_0_df['click_cnts'] = test_click_0_df.groupby(['user_id'])['time'].transform('count')



    print(len(train_click_0_df['user_id']))
    train_click_0_df = train_click_0_df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')  
    #print(len(train_click_0_df['user_id'].drop_duplicates(keep='first')))
    print(len(train_click_0_df))

    print(len(train_user_df['user_id']),len(train_click_0_df['user_id'].drop_duplicates(keep='first')))
    inner_join = pd.merge( train_user_df['user_id'], train_click_0_df['user_id'].drop_duplicates(keep='first'),how='inner')

    print(len(inner_join))
    exit(0)

    #print(len(train_item_df[~train_item_df['txt_vec127'].isnull()]))

    #exit(0)
    #for item in train_item_df:
    #    if item

    #print(train_item_df[:3]['img_vec123'])
    #print(train_item_df.sort_values('item_id'))


# 商品共现频次：连续出现
def item_count():
    tmp = train_click_0_df.sort_values('time')
    tmp['next_item'] = tmp.groupby(['user_id'])['item_id'].transform(lambda x:x.shift(-1))
    union_item = tmp.groupby(['item_id','next_item'])['time'].agg({'count'}).reset_index().sort_values('count', ascending=False)
    #print(union_item[['count']].describe())


#查看缺失向量
def Show_missing():
    # print(train_click_0_df)
    # tmp = train_click_0_df[train_click_0_df['user_id']==5701] 
    #train_click_0_df['user_id'] = train_click_0_df['user_id'].drop_duplicates()

    print(len(train_click_0_df['user_id']))
    train_click_0_df = train_click_0_df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')  
    #print(len(train_click_0_df['user_id'].drop_duplicates(keep='first')))
    print(len(train_click_0_df))

    print(len(train_user_df['user_id']),len(train_click_0_df['user_id'].drop_duplicates(keep='first')))
    inner_join = pd.merge( train_user_df['user_id'], train_click_0_df['user_id'].drop_duplicates(keep='first'),how='inner')

    print(len(inner_join))
    exit(0)

    tmp = tmp.merge(train_user_df, on='user_id', how='left').drop_duplicates()

    #print(tmp[~tmp['txt_vec0'].isnull()])
    # print(tmp[tmp['txt_vec0'].isnull()])
    print(len(tmp))

#分析 某用户前后点击 商品相似度
def show_click_item_sim(user_id = 5708):
    tmp = train_click_0_df[train_click_0_df['user_id']==user_id]
    tmp = tmp.merge(train_item_df,on = 'item_id', how = 'left')

    nonull_tmp = tmp[~tmp['txt_vec0'].isnull()]
    print(nonull_tmp)
    exit(0)
    sim_list = []
    for i in range(0, nonull_tmp.shape[0]-1):
        emb1 = nonull_tmp.values[i][-128-128:-128]
        emb2 = nonull_tmp.values[i+1][-128-128:-128]
        sim_list.append(np.dot(emb1,emb2)/(np.linalg.norm(emb1)*(np.linalg.norm(emb2))))
    sim_list.append(0)

    plt.figure()
    plt.figure(figsize=(10, 6))
    fig = sns.lineplot(x=[i for i in range(len(sim_list))], y=sim_list)
    for item in fig.get_xticklabels():
        item.set_rotation(90)
    plt.tight_layout()
    plt.title('用户点击序列前后txt相似性')
    #plt.show()
    plt.savefig('用户点击序列前后txt相似性'+str(user_id)+'.png',dpi=600)
    plt.cla()


def Creat_data_for_CTR(now_phase):
    whole_click_train = pd.DataFrame()  
    whole_click_test = pd.DataFrame()  
    for c in range(now_phase + 1):  
        print('phase:', c)  
        
        click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,  names=['user_id', 'item_id', 'time'])  
        click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c,c), header=None,  names=['user_id', 'item_id', 'time'])  
        
        whole_click_train = whole_click_train.append(click_train)
        whole_click_test = whole_click_test.append(click_test)
        
        all_click = click_train.append(click_test)  
        whole_click = whole_click.append(all_click)  
        whole_click = whole_click.drop_duplicates(subset=['user_id','item_id','time'],keep='last')
        whole_click = whole_click.sort_values('time')

        #item_sim_list, user_item = get_sim_item(whole_click, 'user_id', 'item_id', use_iif=False)  

        for i in tqdm(click_test['user_id'].unique()):  
            rank_item = recommend(item_sim_list, user_item, i, 500, 500)  
            for j in rank_item:  
                recom_item.append([i, j[0], j[1]])  
    
    result.to_csv('data/train_ctr/train_click_P_N.csv', index=False, header=None)
    result.to_csv('data/train_ctr/test_click_P_N.csv', index=False, header=None)