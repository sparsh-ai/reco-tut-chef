import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm


def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList

def load_negative_file(filename):
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1: ]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users, num_items = train.shape
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def load_rating_file_as_matrix(filename):
    '''
    Read .rating file and Return dok matrix.
    The first line of .rating file is: num_users\t num_items
    '''
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    # Construct matrix
    mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if (rating > 0):
                mat[user, item] = 1.0
            line = f.readline()    
    return mat


def load_data(filepath):
    df = pd.read_csv(filepath,
                     sep="::",
                     header=None,
                     engine='python',
                     names=['userId', 'movieId', 'rating', 'time'])
    df = df.drop('time', axis=1)
    df['userId'] = df['userId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)
    df['rating'] = df['rating'].astype(float)
    
    df = df[['userId', 'movieId', 'rating']]
    df['rating'] = 1.
    m_codes = df['movieId'].astype('category').cat.codes
    u_codes = df['userId'].astype('category').cat.codes
    df['movieId'] = m_codes
    df['userId'] = u_codes
    
    return df


def make_triplet(df):
    df_ = df.copy()
    user_id = df['userId'].unique()
    item_id = df['movieId'].unique()
    
    negs = np.zeros(len(df), dtype=int)
    for u in tqdm(user_id):
        user_idx = list(df[df['userId']==u].index)
        n_choose = len(user_idx)
        available_negative = list(set(item_id) - set(df[df['userId']==u]['movieId'].values))
        new = np.random.choice(available_negative, n_choose, replace=True)
        
        negs[user_idx] = new
    df_['negative'] = negs
    
    return df_


def extract_from_df(df, n_positive, n_negative):
    df_ = df.copy()
    rtd = []
    
    user_id = df['userId'].unique()
    
    for i in tqdm(user_id):
        rtd += list(np.random.choice(df[df['userId']==i][df['rating']==1]['movieId'].index, n_positive, replace=False))
        rtd += list(np.random.choice(df[df['userId']==i][df['rating']==0]['movieId'].index, n_negative, replace=False))
        
    return rtd


# train = load_rating_file_as_matrix('./data/ml-1m.train.rating')
# print(train.toarray())
# print(train.toarray().shape)
# # rating = get_train_instances(train, 0)
# # print(np.array(rating[-1]))

# # values = np.ones(len(rating))

# # users = rating[:,0]
# # items = rating[:,1]
# # X = csr_matrix((values, (users, items)))
# # print(X.toarray().shape)