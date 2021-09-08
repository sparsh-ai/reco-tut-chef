import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

import os
from pathlib import Path

from collections import defaultdict

import numpy as np
from tqdm import tqdm
import scipy.sparse as sparse

import tracemalloc
from datetime import datetime
from time import time

from functools import partial
from multiprocessing.pool import ThreadPool


class DataSplit:
    def __init__(self, seed=42):
        self.seed = seed

    def splitting_functions_factory(self, function_name):
        """Returns splitting function based on name"""
        if function_name == "by_time":
            return self.split_by_time

    def split_by_time(self, interactions, fraction_test):
        """
        Splits interactions by time. Returns tuple of dataframes: train and test.
        """

        np.random.seed(self.seed)

        test_min_timestamp = np.percentile(
            interactions["timestamp"], 100 * (1 - fraction_test)
        )

        train = interactions[interactions["timestamp"] < test_min_timestamp]
        test = interactions[interactions["timestamp"] >= test_min_timestamp]

        return train, test

    @staticmethod
    def filtering_restrict_to_train_users(train, test):
        """
        Returns test DataFrame restricted to users from train set.
        """
        train_users = set(train["user"])
        return test[test["user"].isin(train_users)]

    @staticmethod
    def filtering_already_interacted_items(train, test):
        """
        Filters out (user, item) pairs from the test set if the given user interacted with a given item in train set.
        """
        columns = test.columns
        already_interacted_items = train[["user", "item"]].drop_duplicates()
        merged = pd.merge(
            test, already_interacted_items, on=["user", "item"], how="left", indicator=True
        )
        test = merged[merged["_merge"] == "left_only"]
        return test[columns]

    @staticmethod
    def filtering_restrict_to_unique_user_item_pair(dataframe):
        """
        Returns pd.DataFrame where each (user, item) pair appears only once.
        A list of corresponding events is stores instead of a single event.
        Returned timestamp is the timestamp of the first (user, item) interaction.
        """
        return (
            dataframe.groupby(["user", "item"])
            .agg({"event": list, "timestamp": "min"})
            .reset_index()
        )

    def split(
        self,
        interactions,
        splitting_config=None,
        restrict_to_train_users=True,
        filter_out_already_interacted_items=True,
        restrict_train_to_unique_user_item_pairs=True,
        restrict_test_to_unique_user_item_pairs=True,
        replace_events_by_ones=True,
    ):
        """
        Main function used for splitting the dataset into the train and test sets.
        Parameters
        ----------
        interactions: pd.DataFrame
            Interactions dataframe
        splitting_config : dict, optional
            Dict with name and parameters passed to splitting function.
            Currently only name="by_time" supported.
        restrict_to_train_users : boolean, optional
            Whether to restrict users in the test set only to users from the train set.
        filter_out_already_interacted_items : boolean, optional
            Whether to filter out (user, item) pairs from the test set if the given user interacted with a given item
            in the train set.
        restrict_test_to_unique_user_item_pairs
            Whether to return only one row per (user, item) pair in test set.
        """

        if splitting_config is None:
            splitting_config = {
                "name": "by_time",
                "fraction_test": 0.2,
            }

        splitting_name = splitting_config["name"]
        splitting_config = {k: v for k, v in splitting_config.items() if k != "name"}

        train, test = self.splitting_functions_factory(splitting_name)(
            interactions=interactions, **splitting_config
        )

        if restrict_to_train_users:
            test = self.filtering_restrict_to_train_users(train, test)

        if filter_out_already_interacted_items:
            test = self.filtering_already_interacted_items(train, test)

        if restrict_train_to_unique_user_item_pairs:
            train = self.filtering_restrict_to_unique_user_item_pair(train)

        if restrict_test_to_unique_user_item_pairs:
            test = self.filtering_restrict_to_unique_user_item_pair(test)

        if replace_events_by_ones:
            train["event"] = 1
            test["event"] = 1

        return train, test


class Dataset:
    def __init__(self,
                 data=None,
                 train=None,
                 valid=None,
                 test=None,
                 val_target_users_size=10000,
                 random_seed=42):
        self.data = data
        self.train = train
        self.valid = valid
        self.test = test
        self.val_target_users_size = val_target_users_size
        self.splitter = DataSplit()
        self.train_ui = None
        self.encode_maps = {}
        self.random_seed = random_seed

    def load_interactions(self, data_path, format='csv'):
        if format=='csv':
            self.data = pd.read_csv(data_path, compression='gzip', header=0)
        elif format=='parquet':
            self.data = pd.read_parquet(data_path)
        self.data.columns = ["user", "item", "event", "timestamp"]
        self.data = self.data.astype({"user": str, "item": str, "event": str, "timestamp": int})

    def split(self):
        train_and_valid, self.test = self.splitter.split(self.data)
        self.train, self.valid = self.splitter.split(train_and_valid)

    def prepare(self):
        data = self.train.copy()
        np.random.seed(self.random_seed)

        user_code_id = dict(enumerate(data["user"].unique()))
        user_id_code = {v: k for k, v in user_code_id.items()}
        data["user_code"] = data["user"].apply(user_id_code.get)

        item_code_id = dict(enumerate(data["item"].unique()))
        item_id_code = {v: k for k, v in item_code_id.items()}
        data["item_code"] = data["item"].apply(item_id_code.get)

        self.train_ui = sparse.csr_matrix(
            (np.ones(len(data)), (data["user_code"], data["item_code"]))
        )
        self.encode_maps = {'user_code_id':user_code_id,
                        'user_id_code':user_id_code,
                        'item_code_id':item_code_id,
                        'item_id_code':item_id_code}

        self.target_users_all = self.test["user"].drop_duplicates()
        validation_users = self.valid["user"].drop_duplicates()
        self.target_users_subset_validation = validation_users.sample(n=min(self.val_target_users_size, len(validation_users)))

    @staticmethod
    def load_target_users(path):
        return list(pd.read_csv(path, compression="gzip", header=None).astype(str).iloc[:, 0])

    @staticmethod
    def _get_subset_by_column(interactions, column, fraction):
        column_df = interactions[column].unique()
        subset = set(np.random.choice(column_df, int(len(column_df) * fraction)))
        return interactions[interactions[column].isin(subset)]

    @staticmethod
    def get_interactions_subset(
        interactions, fraction_users, fraction_items, random_seed=10
    ):
        """
        Select subset from interactions based on fraction of users and items
        :param interactions: Original interactions
        :param fraction_users: Fraction of users
        :param fraction_items: Fraction of items
        :param random_seed: Random seed
        :return: Dataframe with subset of interactions
        """

        np.random.seed(random_seed)
        if fraction_users < 1:
            interactions = _get_subset_by_column(interactions, "user", fraction_users)

        if fraction_items < 1:
            interactions = _get_subset_by_column(interactions, "item", fraction_items)

        return interactions