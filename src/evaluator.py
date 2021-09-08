from pathlib import Path
import scipy.sparse as sparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class Metrics:
    def __init__(self):
        pass

    def ranking_metrics(self, test_matrix, recommendations, k=10):
        """
        Calculates ranking metrics (precision, recall, F1, F0.5, NDCG, mAP, MRR, LAUC, HR)
        based on test interactions matrix and recommendations
        :param test_matrix: Test interactions matrix
        :param recommendations: Recommendations
        :param k: Number of top recommendations to calculate metrics on
        :return: Dataframe with metrics
        """

        items_number = test_matrix.shape[1]
        metrics = {
            "precision": 0,
            "recall": 0,
            "F_1": 0,
            "F_05": 0,
            "ndcg": 0,
            "mAP": 0,
            "MRR": 0,
            "LAUC": 0,
            "HR": 0,
        }

        denominators = {
            "relevant_users": 0,
        }

        for (user_count, user) in tqdm(enumerate(recommendations[:, 0])):
            u_interacted_items = self.get_interacted_items(test_matrix, user)
            interacted_items_amount = len(u_interacted_items)

            if interacted_items_amount > 0:  # skip users with no items in test set
                denominators["relevant_users"] += 1

                # evaluation
                success_statistics = self.calculate_successes(
                    k, recommendations, u_interacted_items, user_count
                )

                user_metrics = self.calculate_ranking_metrics(
                    success_statistics,
                    interacted_items_amount,
                    items_number,
                    k,
                )

                for metric_name in metrics:
                    metrics[metric_name] += user_metrics[metric_name]

        metrics = {
            name: metric / denominators["relevant_users"]
            for name, metric in metrics.items()
        }

        return pd.DataFrame.from_dict(metrics, orient="index").T

    def calculate_ranking_metrics(
        self,
        success_statistics,
        interacted_items_amount,
        items_number,
        k,
    ):
        """
        Calculates ranking metrics based on success statistics
        :param success_statistics: Success statistics dictionary
        :param interacted_items_amount:
        :param items_number:
        :param k: Number of top recommendations to calculate metrics on
        :return: Dictionary with metrics
        """
        precision = success_statistics["total_amount"] / k
        recall = success_statistics["total_amount"] / interacted_items_amount
        user_metrics = dict(
            precision=precision,
            recall=recall,
            F_1=self.calculate_f(precision, recall, 1),
            F_05=self.calculate_f(precision, recall, 0.5),
            ndcg=self.calculate_ndcg(interacted_items_amount, k, success_statistics["total"]),
            mAP=self.calculate_map(success_statistics, interacted_items_amount, k),
            MRR=self.calculate_mrr(success_statistics["total"]),
            LAUC=self.calculate_lauc(
                success_statistics, interacted_items_amount, items_number, k
            ),
            HR=success_statistics["total_amount"] > 0,
        )
        return user_metrics

    @staticmethod
    def calculate_mrr(user_successes):
        return (
            1 / (user_successes.nonzero()[0][0] + 1)
            if user_successes.nonzero()[0].size > 0
            else 0
        )

    @staticmethod
    def calculate_f(precision, recall, f):
        return (
            (f ** 2 + 1) * (precision * recall) / (f ** 2 * precision + recall)
            if precision + recall > 0
            else 0
        )

    @staticmethod
    def calculate_lauc(successes, interacted_items_amount, items_number, k):
        return (
            np.dot(successes["cumsum"], 1 - successes["total"])
            + (successes["total_amount"] + interacted_items_amount)
            / 2
            * ((items_number - interacted_items_amount) - (k - successes["total_amount"]))
        ) / ((items_number - interacted_items_amount) * interacted_items_amount)

    @staticmethod
    def calculate_map(successes, interacted_items_amount, k):
        return np.dot(successes["cumsum"] / np.arange(1, k + 1), successes["total"]) / min(
            k, interacted_items_amount
        )

    @staticmethod
    def calculate_ndcg(interacted_items_amount, k, user_successes):
        cumulative_gain = 1.0 / np.log2(np.arange(2, k + 2))
        cg_sum = np.cumsum(cumulative_gain)
        return (
            np.dot(user_successes, cumulative_gain)
            / cg_sum[min(k, interacted_items_amount) - 1]
        )

    @staticmethod
    def calculate_successes(k, recommendations, u_interacted_items, user_count):

        items = recommendations[user_count, 1 : k + 1]
        user_successes = np.isin(items, u_interacted_items)

        return dict(
            total=user_successes.astype(int),
            total_amount=user_successes.sum(),
            cumsum=np.cumsum(user_successes),
        )

    @staticmethod
    def get_reactions(test_matrix, user):
        return test_matrix.data[test_matrix.indptr[user] : test_matrix.indptr[user + 1]]

    @staticmethod
    def get_interacted_items(test_matrix, user):
        return test_matrix.indices[test_matrix.indptr[user] : test_matrix.indptr[user + 1]]

    def diversity_metrics(
        self, test_matrix, formatted_recommendations, original_recommendations, k=10
    ):
        """
        Calculates diversity metrics
        (% if recommendations in test, test coverage, Shannon, Gini, users without recommendations)
        based on test interactions matrix and recommendations
        :param test_matrix: user/item interactions' matrix
        :param formatted_recommendations: recommendations where user and item ids were replaced by respective codes based on test_matrix
        :param original_recommendations: original format recommendations
        :param k: Number of top recommendations to calculate metrics on
        :return: Dataframe with metrics
        """

        formatted_recommendations = formatted_recommendations[:, : k + 1]

        frequency_statistics = self.calculate_frequencies(formatted_recommendations, test_matrix)

        with np.errstate(
            divide="ignore"
        ):  # let's put zeros we items with 0 frequency and ignore division warning
            log_frequencies = np.nan_to_num(
                np.log(frequency_statistics["frequencies"]), posinf=0, neginf=0
            )

        metrics = dict(
            reco_in_test=frequency_statistics["recommendations_in_test_n"]
            / frequency_statistics["total_recommendations_n"],
            test_coverage=frequency_statistics["recommended_items_n"]
            / test_matrix.shape[1],
            Shannon=-np.dot(frequency_statistics["frequencies"], log_frequencies),
            Gini=self.calculate_gini(
                frequency_statistics["frequencies"], frequency_statistics["items_in_test_n"]
            ),
            users_without_reco=original_recommendations.iloc[:, 1].isna().sum()
            / len(original_recommendations),
            users_without_k_reco=original_recommendations.iloc[:, k - 1].isna().sum()
            / len(original_recommendations),
        )

        return pd.DataFrame.from_dict(metrics, orient="index").T

    @staticmethod
    def calculate_gini(frequencies, items_in_test_n):
        return (
            np.dot(
                frequencies,
                np.arange(
                    1 - items_in_test_n,
                    items_in_test_n,
                    2,
                ),
            )
            / (items_in_test_n - 1)
        )

    @staticmethod
    def calculate_frequencies(formatted_recommendations, test_matrix):
        frequencies = defaultdict(
            int, [(item, 0) for item in list(set(test_matrix.indices))]
        )
        for item in formatted_recommendations[:, 1:].flat:
            frequencies[item] += 1
        recommendations_out_test_n = frequencies[-1]
        del frequencies[-1]
        frequencies = np.array(list(frequencies.values()))
        items_in_test_n = len(frequencies)
        recommended_items_n = len(frequencies[frequencies > 0])
        recommendations_in_test_n = np.sum(frequencies)
        frequencies = frequencies / np.sum(frequencies)
        frequencies = np.sort(frequencies)
        return dict(
            frequencies=frequencies,
            items_in_test_n=items_in_test_n,
            recommended_items_n=recommended_items_n,
            recommendations_in_test_n=recommendations_in_test_n,
            total_recommendations_n=recommendations_out_test_n + recommendations_in_test_n,
        )


class Evaluator:
    """
    Class used for models evaluation
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        recommendations_path: Path,
        test_path: Path,
        k,
        models_to_evaluate,
    ):
        self.recommendations_path = recommendations_path
        self.test_path = test_path
        self.k = k
        self.models_to_evaluate = models_to_evaluate
        self.located_models = None
        self.test = None
        self.user_map = None
        self.item_map = None
        self.test_matrix = None
        self.evaluation_results = []

        self.metrics = Metrics()

    @staticmethod
    def _map_column(test, column):
        test[f"{column}_code"] = test[column].astype("category").cat.codes
        return dict(zip(test[column], test[f"{column}_code"]))

    @staticmethod
    def _get_models(models_to_evaluate, recommendations_path):
        models = [
            (file_name.split(".")[0], file_name)
            for file_name in os.listdir(recommendations_path)
        ]
        if models_to_evaluate:
            return [model for model in models if model[0] in models_to_evaluate]
        return models
    
    def _read_recommendations(self, file_name):
        return pd.read_csv(
            os.path.join(self.recommendations_path, file_name),
            header=None,
            compression="gzip",
            dtype=str,
        )

    @staticmethod
    def _format_recommendations(recommendations, user_id_code, item_id_code):
        users = recommendations.iloc[:, :1].applymap(
            lambda x: user_id_code.setdefault(str(x), -1)
        )
        items = recommendations.iloc[:, 1:].applymap(
            lambda x: -1 if pd.isna(x) else item_id_code.setdefault(x, -1)
        )
        return np.array(pd.concat([users, items], axis=1))

    def preprocess_test(self, test: pd.DataFrame):
        """
        Preprocesses test set to speed up evaluation
        """

        test = test.copy()

        test.columns = ["user", "item", "event", "timestamp"]
        user_map = self._map_column(test, "user")
        item_map = self._map_column(test, "item")

        test_matrix = sparse.csr_matrix(
            (np.ones(len(test)), (test["user_code"], test["item_code"]))
        )
        return user_map, item_map, test_matrix

    def prepare(self):
        """
        Prepares test set and models to evaluate
        """

        self.test = pd.read_csv(self.test_path, compression="gzip").astype(
            {"user": str, "item": str}
        )
        self.user_map, self.item_map, self.test_matrix = self.preprocess_test(self.test)

        self.located_models = self._get_models(
            self.models_to_evaluate, self.recommendations_path
        )

    def evaluate_models(self):
        """
        Evaluating multiple models
        """

        for model, file_name in self.located_models:
            recommendations = self._read_recommendations(file_name)
            evaluation_result = self.evaluate(
                original_recommendations=recommendations,
            )

            evaluation_result.insert(0, "model_name", model)
            self.evaluation_results.append(evaluation_result)
        self.evaluation_results = pd.concat(self.evaluation_results).set_index(
            "model_name"
        )
        if "precision" in self.evaluation_results.columns:
            self.evaluation_results = self.evaluation_results.sort_values(
                by="precision", ascending=False
            )

    def evaluate(
        self,
        original_recommendations: pd.DataFrame,
    ):
        """
        Evaluate single model
        """

        original_recommendations = original_recommendations.iloc[:, : self.k + 1].copy()

        formatted_recommendations = self._format_recommendations(
            original_recommendations, self.user_map, self.item_map
        )

        evaluation_results = pd.concat(
            [
                self.metrics.ranking_metrics(
                    self.test_matrix,
                    formatted_recommendations,
                    k=self.k,
                ),
                self.metrics.diversity_metrics(
                    self.test_matrix,
                    formatted_recommendations,
                    original_recommendations,
                    self.k,
                ),
            ],
            axis=1,
        )

        return evaluation_results