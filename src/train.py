import os
import sys
import yaml
import pickle
import pandas as pd

from src.models import PopularityRecommender

params = yaml.safe_load(open("params.yaml"))["train"]

models = {
    'itempop': PopularityRecommender()
}


def load_data(path):
    path_train = os.path.join(path, 'train.parquet.snappy')
    path_test = os.path.join(path, 'test.parquet.snappy')
    train = pd.read_parquet(path_train)
    test = pd.read_parquet(path_test)
    return train, test


def load_model(name):
    model = models[name]
    return model


if __name__ == "__main__":
    model_name = params['model_name']
    data_path = str(sys.argv[1])
    model_path = str(sys.argv[2])
    train, test = load_data(data_path)
    model = load_model(model_name)
    model.fit(train)
    pickle.dump(model, open(os.path.join(model_path, model_name+'.pkl'), 'wb'))