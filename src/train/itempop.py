
import os
import sys
import pickle
import pandas as pd

from src.models.itempop import PopularityRecommender

data_source_path = str(sys.argv[1])
data_source_path_train = os.path.join(data_source_path, 'train.parquet.snappy')
data_source_path_test = os.path.join(data_source_path, 'test.parquet.snappy')

train_data = pd.read_parquet(data_source_path_train)
test_data = pd.read_parquet(data_source_path_test)

pop_recommender = PopularityRecommender()

pop_recommender.fit(train_data)

pickle.dump(pop_recommender, open('./artifacts/30music/models/itempop.pkl', 'wb'))