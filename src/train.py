import os
import sys
import yaml
import pickle
import pandas as pd

from src.models import PopularityRecommender, KNNRecommender


params = yaml.safe_load(open("params.yaml"))["train"]


def load_data(datapath):
    path_train = os.path.join(datapath, 'train.parquet.snappy')
    path_test = os.path.join(datapath, 'test.parquet.snappy')
    train = pd.read_parquet(path_train)
    test = pd.read_parquet(path_test)
    return train, test


def train_model(modelname, train):
    models = {'itempop': PopularityRecommender(),
              'knn': KNNRecommender(model='sknn', k=10)}
    model = models[modelname]
    model.fit(train)
    return model


def save_model(model, modelname, modelpath):
    pickle.dump(model, open(os.path.join(modelpath, modelname+'.pkl'), 'wb'))


if __name__ == "__main__":
    # load the params
    modelname = params['model_name']
    datapath = str(sys.argv[1])
    modelpath = str(sys.argv[2])
    # load the data
    train, test = load_data(datapath)
    # train the model
    model = train_model(modelname, train)
    # save the model
    save_model(model, modelname, modelpath)