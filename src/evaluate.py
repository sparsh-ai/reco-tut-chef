import os
import sys
import pickle
import pandas as pd

from src.eval import SequentialEvaluator


def load_data(datapath):
    path_train = os.path.join(datapath, 'train.parquet.snappy')
    path_test = os.path.join(datapath, 'test.parquet.snappy')
    train = pd.read_parquet(path_train)
    test = pd.read_parquet(path_test)
    return train, test


def load_model(modelpath):
    model = pickle.load(open(modelpath, 'rb'))
    return model


def save_results(evaluator, resultspath):
    results = {}
    results['seq_reveal'] = evaluator.eval_seqreveal()
    results['static_profile'] = evaluator.eval_staticprofile()
    results['rec_length'] = evaluator.eval_reclength()
    results['profile_length'] = evaluator.eval_profilelength()
    pickle.dump(results, open(resultspath, 'wb'))


if __name__ == "__main__":
    # load the params
    datapath = str(sys.argv[1])
    modelpath = str(sys.argv[2])
    resultspath = str(sys.argv[3])
    # load the data
    train, test = load_data(datapath)
    # load the model
    model = load_model(modelpath)
    # evaluate and save the results
    evaluator = SequentialEvaluator(train, test, model)
    save_results(evaluator, resultspath)