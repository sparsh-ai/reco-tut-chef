
import os
import sys
import pickle
import pandas as pd

from src.eval.seqeval import SequentialEvaluator

data_source_path = str(sys.argv[1])
data_source_path_train = os.path.join(data_source_path, 'train.parquet.snappy')
data_source_path_test = os.path.join(data_source_path, 'test.parquet.snappy')

train_data = pd.read_parquet(data_source_path_train)
test_data = pd.read_parquet(data_source_path_test)

model_path = str(sys.argv[2])
model = pickle.load(open(model_path, 'rb'))

evaluator = SequentialEvaluator(train_data, test_data, model)

results = {}
results['seq_reveal'] = evaluator.eval_seqreveal()
results['static_profile'] = evaluator.eval_staticprofile()
results['rec_length'] = evaluator.eval_reclength()
results['profile_length'] = evaluator.eval_profilelength()

pickle.dump(results, open('./artifacts/30music/results/itempop.pkl', 'wb'))