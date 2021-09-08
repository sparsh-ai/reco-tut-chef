from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from src.models import BaseRecommender


class TopPop(BaseRecommender):
    """
    TopPop recommender, which recommends the most popular items
    """

    def __init__(self, train_ui, encode_maps, show_progress=True):
        super().__init__()

        self.popular_items = None

        self.train_ui = train_ui
        self.user_id_code = encode_maps['user_id_code']
        self.user_code_id = encode_maps['user_code_id']
        self.item_code_id = encode_maps['item_code_id']

        self.show_progress = show_progress

    def fit(self):
        """
        Fit the model
        """
        self.popular_items = (-self.train_ui.sum(axis=0).A.ravel()).argsort()

    def recommend(
        self,
        target_users,
        n_recommendations,
        filter_out_interacted_items=True,
    ) -> pd.DataFrame:
        """
            Recommends n_recommendations items for target_users
        :return:
            pd.DataFrame (user, item_1, item_2, ..., item_n)
        """

        with ThreadPool() as thread_pool:
            recommendations = list(
                tqdm(
                    thread_pool.imap(
                        partial(
                            self.recommend_per_user,
                            n_recommendations=n_recommendations,
                            filter_out_interacted_items=filter_out_interacted_items,
                        ),
                        target_users,
                    ),
                    disable=not self.show_progress,
                )
            )

        return pd.DataFrame(recommendations)

    def recommend_per_user(
        self, user, n_recommendations, filter_out_interacted_items=True
    ):
        """
        Recommends n items per user
        :param user: User id
        :param n_recommendations: Number of recommendations
        :param filter_out_interacted_items: boolean value to filter interacted items
        :return: list of format [user_id, item1, item2 ...]
        """
        u_code = self.user_id_code.get(user)
        u_recommended_items = []

        if u_code is not None:
            exclude_items = []
            if filter_out_interacted_items:
                exclude_items = self.train_ui.indices[
                    self.train_ui.indptr[u_code] : self.train_ui.indptr[u_code + 1]
                ]

            u_recommended_items = self.popular_items[
                : n_recommendations + len(exclude_items)
            ]

            u_recommended_items = [
                self.item_code_id[i]
                for i in u_recommended_items
                if i not in exclude_items
            ]

            u_recommended_items = u_recommended_items[:n_recommendations]

        return (
            [user]
            + u_recommended_items
            + [None] * (n_recommendations - len(u_recommended_items))
        )