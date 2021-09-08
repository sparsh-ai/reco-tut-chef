from functools import partial
from multiprocessing.pool import ThreadPool

import implicit
import pandas as pd
from scipy import sparse
from tqdm import tqdm

from src.models import BaseRecommender


class ALS(BaseRecommender):
    """
    Module implementing a wrapper for the ALS model
    Wrapper over ALS model
    """

    def __init__(self, train_ui,
                 encode_maps,
                 factors=100,
                 regularization=0.01,
                 use_gpu=False,
                 iterations=15,
                 event_weights_multiplier=100,
                 show_progress=True,
                 ):
        """
        Source of descriptions:
        https://github.com/benfred/implicit/blob/master/implicit/als.py
        Alternating Least Squares
        A Recommendation Model based on the algorithms described in the paper
        'Collaborative Filtering for Implicit Feedback Datasets'
        with performance optimizations described in 'Applications of the
        Conjugate Gradient Method for Implicit Feedback Collaborative Filtering.'
        Parameters
        ----------
        factors : int, optional
            The number of latent factors to compute
        regularization : float, optional
            The regularization factor to use
        use_gpu : bool, optional
            Fit on the GPU if available, default is to run on CPU
        iterations : int, optional
            The number of ALS iterations to use when fitting data
        event_weights_multiplier: int, optional
            The multiplier of weights.
            Used to find a tradeoff between the importance of interacted and not interacted items.
        """

        super().__init__()

        self.train_ui = train_ui
        self.user_id_code = encode_maps['user_id_code']
        self.user_code_id = encode_maps['user_code_id']
        self.item_code_id = encode_maps['item_code_id']
        self.mapping_user_test_items = None
        self.similarity_matrix = None

        self.show_progress = show_progress

        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            use_gpu=use_gpu,
            iterations=iterations,
        )

        self.event_weights_multiplier = event_weights_multiplier

    def fit(self):
        """
        Fit the model
        """
        self.model.fit(self.train_ui.T, show_progress=self.show_progress)

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

            u_recommended_items = list(
                zip(
                    *self.model.recommend(
                        u_code,
                        self.train_ui,
                        N=n_recommendations,
                        filter_already_liked_items=filter_out_interacted_items,
                    )
                )
            )[0]

            u_recommended_items = [self.item_code_id[i] for i in u_recommended_items]

        return (
            [user]
            + u_recommended_items
            + [None] * (n_recommendations - len(u_recommended_items))
        )