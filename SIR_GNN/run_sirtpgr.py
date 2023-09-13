import numpy as np
import tensorflow as tf
import pandas as pd
from sir.SirSharedTree import *
import datetime
from funk_svd import SVD

np.random.seed(4)
tf.set_random_seed(4)

dataset = 'netflix_1_64.dat'
other_information = ''
path = '/dataset/netflix_1_64/'

#dataset = 'ratings_1m.dat'
#other_information = 'movies_1m.dat'
#path = '/dataset/ML-1m/'

#dataset = 'ratings_10m.dat'
#other_information = 'movies_10m.dat'
#path = '/dataset/ML-10m/'


batch_size = 16
max_seq_length = 8
hidden_size = 64
feature_size = 64
learning_rate = 5e-4
discount_factor = 0.8
gnn_layer = 1
epsilon = 3 #global图关联窗口大小
global_topk = 10    #global图关系中取最重要的k个

data = pd.read_table(path+dataset, sep='::', names=['u_id', 'i_id', 'rating', 'timestep'])
if dataset != 'netflix_1_64.dat':
    movie = pd.read_table(path+other_information, sep='::', names=['i_id', 'title', 'genre'])


user_idx = list(data['u_id'].unique())  # id for all the user
np.random.shuffle(user_idx)
#user_idx = user_idx[:len(user_idx)//4]  #10m用1/4
train_id = user_idx[:int(len(user_idx) * 0.8)]
test_id = user_idx[int(len(user_idx) * 0.8):]

# count the movies
movie_id = [-1]
for idx1 in user_idx:  # 针对train_id中的每个用户
    user_record = data[data['u_id'] == idx1]
    for idx2, row in user_record.iterrows():
        # 检查list中是否有该电影的id
        if row['i_id'] in movie_id:
            idx = movie_id.index(row['i_id'])  # 找到该位置
        else:
            # 否则新加入movie_id
            movie_id.append(row['i_id'])

item_size = len(movie_id)
if dataset != 'netflix_1_64.dat':
# movie information

    y_min = 2100 # 1919_1m
    y_max = 1900 # 2000_1m
    movie_years = {}
    movie_genre = {}
    cnt_genre = 0 # 18
    genres = []
    for _, row in movie.iterrows():
        if row['i_id'] in movie_id:
            idx = movie_id.index(row['i_id'])
            movie_years[idx] = int(str(row['title'])[-5:-1])
            genre_list = str(row['genre']).split('|')
            movie_genre[idx] = genre_list
            for g in genre_list:
                if g not in genres:
                    genres.append(g)
                    cnt_genre += 1
            if movie_years[idx]>y_max:
                y_max=movie_years[idx]
            if movie_years[idx]<y_min:
                y_min=movie_years[idx]

print('Total movie count:', item_size-1)

#global图的邻接矩阵和邻接表
def get_all_adj(train_data):
    adj_list_weight = np.zeros([item_size, item_size])
    adj_list = []
    for id in train_data:
        user_record = data[data['u_id'] == id]
        user_record = user_record.sort_values(by='timestep')
        item_list = []
        rating_list = []
        for _, row in user_record.iterrows():
            item_list.append(movie_id.index(row['i_id']))
            rating_list.append(row['rating'])
        for i in range(len(item_list) - epsilon):
            for j in range(i+1,i+epsilon+1):
                #if item_relation_ASG(item_list[i],item_list[j])>0:
                adj_list_weight[item_list[i]][item_list[j]] += 1
                adj_list_weight[item_list[j]][item_list[i]] += 1

    for i in range(len(adj_list_weight)):
        tmp = adj_list_weight[i].copy()
        tmp.sort()
        tmp = tmp[::-1]
        last = tmp[min(len(tmp), global_topk) - 1]
        left = []
        for j in range(len(adj_list_weight[i])):
            if adj_list_weight[i][j] >= last and adj_list_weight[i][j] > 0:
                left.append(j)
            else:
                adj_list_weight[i][j] = 0
        left += [0] * (global_topk - len(left))
        left = left[:global_topk]
        adj_list.append(left)

    return adj_list_weight, np.array(adj_list)


def normalize(rating):
    max_rating = 5
    min_rating = 0
    return -1 + 2 * (rating - min_rating) / (max_rating - min_rating)

def item_relation_ASG(item_1,item_2):
    if item_1 == 0 or item_2 == 0 or item_1 == item_2:
        return 0
    score = 0

    if dataset == 'netflix_1_64.dat':
        return 1

    for g in movie_genre[item_1]:
        if g in movie_genre[item_2]:
            score += 1
    '''
    if abs(movie_years[item_1]-movie_years[item_2]) < 5:
        score += 1
    '''
    return score

def item_relation_WSG(item_1,item_2):
    if item_1 == 0 or item_2 == 0 or item_1 == item_2:
        return 0
    score = 0

    for g in movie_genre[item_1]:
        if g in movie_genre[item_2]:
            score += 1

    if abs(movie_years[item_1]-movie_years[item_2]) < 5:
        score += 1

    return score

def process_data(item_list, rating_list, Q_value):
    if len(item_list)>0:
        action = item_list.pop()
        reward = Q_value.pop()
    else:
        action = 0
        reward = 0
    mask = [1.] * len(item_list)
    state_len=max_seq_length-1
    while len(item_list)<state_len:
        item_list.append(0)
        rating_list.append(-1)
        mask.append(0.)

    items = np.unique(item_list).tolist()
    items = items + [0]*(state_len-len(items))
    alias_inputs = [items.index(i) for i in item_list]
    WSG = np.zeros((state_len, state_len), dtype=np.float32)
    AIG = np.zeros((state_len, state_len), dtype=np.float32)
    '''
    for i in range(state_len):
        for j in range(state_len):
            u = alias_inputs[i]
            v = alias_inputs[j]
            if item_list[i] != 0 and item_list[j]!=0:
                A[u][v] += item_relation(item_list[i], item_list[j])
    '''
    for i in range(state_len-1):
        u = alias_inputs[i]
        v = alias_inputs[i+1]
        AIG[u][v] += 1

    for i in range(state_len-1):
        u = alias_inputs[i]
        for j in range(i+1,state_len):
            v = alias_inputs[j]
            if rating_list[i] >= 3 and rating_list[j] >= 3:
            #if item_relation_WSG(item_list[i],item_list[j]) > 0 or (rating_list[i] >= 3 and rating_list[j] >= 3):
                WSG[u][v] = 1

    sum_in = np.sum(WSG, 0)
    sum_in[np.where(sum_in == 0)] = 1
    WSG_in = np.divide(WSG, sum_in)
    sum_out = np.sum(WSG, 1)
    sum_out[np.where(sum_out == 0)] = 1
    WSG_out = np.divide(WSG.transpose(), sum_out)

    sum_in = np.sum(AIG, 0)
    sum_in[np.where(sum_in == 0)] = 1
    AIG_in = np.divide(AIG, sum_in)
    sum_out = np.sum(AIG, 1)
    sum_out[np.where(sum_out == 0)] = 1
    AIG_out = np.divide(AIG.transpose(), sum_out)

    return WSG_in, WSG_out, AIG_in, AIG_out, items, alias_inputs, mask, action, reward

def evaluate(recommend_id, item_id, rating, top_N):
    '''
    evalute the recommend result for each user.
    :param recommend_id: the recommend_result for each item, a list that contains the results for each item.
    :param item_id: item id.
    :param rating: user's rating on item.
    :param top_N: N, a real number of N for evaluation.
    :return: reward@N, recall@N, MRR@N
    '''
    session_length = len(recommend_id)
    relevant = 0
    recommend_relevant = 0
    selected = 0
    output_reward = 0
    mrr = 0
    if session_length == 0:
        return 0, 0, 0, 0
    for ti in range(session_length):
        current_recommend_id = list(recommend_id[ti])[:top_N]
        current_item = item_id[ti]
        current_rating = rating[ti]
        if current_rating > 3.5:
            relevant += 1
            if current_item in current_recommend_id:
                recommend_relevant += 1
        if current_item in current_recommend_id:
            selected += 1
            output_reward += normalize(current_rating)
            rank = current_recommend_id.index(current_item)
            mrr += 1.0 / (rank + 1)
    recall = recommend_relevant / relevant if relevant != 0 else 0
    precision = recommend_relevant / session_length
    return output_reward / session_length, precision, recall, mrr / session_length



print('Begin training the tree policy.')
start = datetime.datetime.now()
train_step = 0
Loss_list = []
#result_analysis = []
adj_w, adj = get_all_adj(train_id)

agent = SharedTreePolicy(adj_w=adj_w, adj=adj, layer=3, branch=int(np.ceil(item_size ** (1 / 3))), learning_rate=1e-4,
                         max_seq_length=max_seq_length-1, hidden_size=hidden_size, batch_size=batch_size,
                         feature_size=feature_size, gnn_layer=gnn_layer, topK=global_topk)

LIST_WSG_in = []
LIST_WSG_out = []
LIST_AIG_in = []
LIST_AIG_out = []
LIST_item = []
LIST_alias_inputs = []
LIST_mask = []
LIST_reward = []
LIST_action = []
for id1 in train_id:
    user_record = data[data['u_id'] == id1]
    user_record = user_record.sort_values(by='timestep')
    item_list = []
    rating_list = []
    Q_value = []
    for _, row in user_record.iterrows():
        item_list.append(movie_id.index(row['i_id']))
        rating_list.append(row['rating'])
    for rating in rating_list:
        Q_value.append(normalize(rating))
    for i in range(len(Q_value)-1,-1,-1):
        Q_value[i-1] += discount_factor*Q_value[i]

    Loss_list = []
    for i in range(len(item_list)-max_seq_length+1):
        WSG_in, WSG_out, AIG_in, AIG_out, items, alias_inputs, mask, action, reward = process_data(item_list[i:i+max_seq_length],
                                                                      rating_list[i:i+max_seq_length],
                                                                      Q_value[i:i+max_seq_length])
        LIST_WSG_in.append(WSG_in)
        LIST_WSG_out.append(WSG_out)
        LIST_AIG_in.append(AIG_in)
        LIST_AIG_out.append(AIG_out)
        LIST_item.append(items)
        LIST_alias_inputs.append(alias_inputs)
        LIST_mask.append(mask)
        LIST_reward.append(reward)
        LIST_action.append(action)
        if len(LIST_action)==batch_size:
            loss = agent.learn(LIST_WSG_in, LIST_WSG_out, LIST_AIG_in, LIST_AIG_out, LIST_item, LIST_alias_inputs, LIST_mask, LIST_reward, LIST_action)
            Loss_list.append(loss)
            LIST_WSG_in = []
            LIST_WSG_out = []
            LIST_AIG_in = []
            LIST_AIG_out = []
            LIST_item = []
            LIST_alias_inputs = []
            LIST_mask = []
            LIST_reward = []
            LIST_action = []

    train_step += 1
    print('User ', train_step, 'Loss: ', np.mean(Loss_list))

while(len(LIST_action)>0 and len(LIST_action)<batch_size):
    WSG_in, WSG_out, AIG_in, AIG_out, items, alias_inputs, mask, action, reward = process_data([], [], [])
    LIST_WSG_in.append(WSG_in)
    LIST_WSG_out.append(WSG_out)
    LIST_AIG_in.append(AIG_in)
    LIST_AIG_out.append(AIG_out)
    LIST_item.append(items)
    LIST_alias_inputs.append(alias_inputs)
    LIST_mask.append(mask)
    LIST_reward.append(reward)
    LIST_action.append(action)
if len(LIST_action)==batch_size:
    loss = agent.learn(LIST_WSG_in, LIST_WSG_out, LIST_AIG_in, LIST_AIG_out, LIST_item, LIST_alias_inputs, LIST_mask, LIST_reward, LIST_action)

end = datetime.datetime.now()
training_time = (end - start).seconds

print('Begin Test')
test_count = 0
result = []
#result_analysis = []
total_testing_steps = 0
start = datetime.datetime.now()

for id1 in test_id:
    user_record = data[data['u_id'] == id1]
    user_record = user_record.sort_values(by='timestep')
    item_list = []
    rating_list = []
    Q_value = []
    test_count += 1
    all_item = []
    all_rating = []
    recommend = []

    LIST_WSG_in = []
    LIST_WSG_out = []
    LIST_AIG_in = []
    LIST_AIG_out = []
    LIST_item = []
    LIST_alias_inputs = []
    LIST_mask = []
    LIST_rating = []
    LIST_action = []

    for _, row in user_record.iterrows():
        item_list.append(movie_id.index(row['i_id']))
        rating_list.append(row['rating'])
    for rating in rating_list:
        Q_value.append(normalize(rating))

    for i in range(len(item_list)-max_seq_length+1):
        WSG_in, WSG_out, AIG_in, AIG_out, items, alias_inputs, mask, action, reward = process_data(item_list[i:i+max_seq_length],
                                                                      rating_list[i:i+max_seq_length],
                                                                      Q_value[i:i+max_seq_length])
        LIST_WSG_in.append(WSG_in)
        LIST_WSG_out.append(WSG_out)
        LIST_AIG_in.append(AIG_in)
        LIST_AIG_out.append(AIG_out)
        LIST_item.append(items)
        LIST_alias_inputs.append(alias_inputs)
        LIST_mask.append(mask)
        LIST_rating.append(rating_list[i+max_seq_length-1])
        LIST_action.append(action)
        if len(LIST_action) == batch_size:
            output_action = agent.get_action_prob(LIST_WSG_in, LIST_WSG_out, LIST_AIG_in, LIST_AIG_out, LIST_item, LIST_alias_inputs, LIST_mask)
            for j in range(len(output_action)):
                if LIST_action[j] == 0:
                    break
                recommend_idx = np.argsort(-output_action[j])[:50]
                recommend.append(recommend_idx)
                all_item.append(LIST_action[j])
                all_rating.append(LIST_rating[j])
            LIST_WSG_in = []
            LIST_WSG_out = []
            LIST_AIG_in = []
            LIST_AIG_out = []
            LIST_item = []
            LIST_alias_inputs = []
            LIST_mask = []
            LIST_rating = []
            LIST_action = []

    while (len(LIST_action) > 0 and len(LIST_action) < batch_size):
        WSG_in, WSG_out, AIG_in, AIG_out, items, alias_inputs, mask, action, reward = process_data([], [], [])
        LIST_WSG_in.append(WSG_in)
        LIST_WSG_out.append(WSG_out)
        LIST_AIG_in.append(AIG_in)
        LIST_AIG_out.append(AIG_out)
        LIST_item.append(items)
        LIST_alias_inputs.append(alias_inputs)
        LIST_mask.append(mask)
        LIST_rating.append(0)
        LIST_action.append(action)
    if len(LIST_action) == batch_size:
        output_action = agent.get_action_prob(LIST_WSG_in, LIST_WSG_out, LIST_AIG_in, LIST_AIG_out, LIST_item, LIST_alias_inputs, LIST_mask)
        for j in range(len(output_action)):
            if LIST_action[j] == 0:
                break
            recommend_idx = np.argsort(-output_action[j])[:50]
            recommend.append(recommend_idx)
            all_item.append(LIST_action[j])
            all_rating.append(LIST_rating[j])

    if len(all_rating) > 0:
        reward_10, precision_10, recall_10, mkk_10 = evaluate(recommend, all_item, all_rating, 10)
        reward_30, precision_30, recall_30, mkk_30 = evaluate(recommend, all_item, all_rating, 30)
        #result_analysis.append((recommend, all_item, all_rating))
        print('Test user #', test_count, '/', len(test_id))
        print('Reward@10: %.4f, Precision@10: %.4f, Recall@10: %.4f, MRR@10: %4f'
              % (reward_10, precision_10, recall_10, mkk_10))
        print('Reward@30: %.4f, Precision@30: %.4f, Recall@30: %.4f, MRR@30: %4f'
              % (reward_30, precision_30, recall_30, mkk_30))
        result.append([reward_10, precision_10, recall_10, mkk_10, reward_30, precision_30, recall_30, mkk_30])
end = datetime.datetime.now()
testing_time = (end - start).seconds

print('###############')
print('Learning finished')
print('Result:')
display = np.mean(np.array(result).reshape([-1, 8]), axis=0)
eval_mat = ["Reward@10", "Precision@10", "Recall@10", "MRR@10", "Reward@30", "Precision@30", "Recall@30", "MRR@30"]
for i in range(len(display)):
    print('%.5f' % display[i])
