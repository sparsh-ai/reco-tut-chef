
class EvalMetrics:
    def __init__(self):
        pass

    def precision(self, ground_truth, prediction):
        """
        Compute Precision metric
        :param ground_truth: the ground truth set or sequence
        :param prediction: the predicted set or sequence
        :return: the value of the metric
        """
        ground_truth = self._remove_duplicates(ground_truth)
        prediction = self._remove_duplicates(prediction)
        precision_score = self._count_a_in_b_unique(prediction, ground_truth) / float(len(prediction))
        assert 0 <= precision_score <= 1
        return precision_score

    def recall(self, ground_truth, prediction):
        """
        Compute Recall metric
        :param ground_truth: the ground truth set or sequence
        :param prediction: the predicted set or sequence
        :return: the value of the metric
        """
        ground_truth = self._remove_duplicates(ground_truth)
        prediction = self._remove_duplicates(prediction)
        recall_score = 0 if len(prediction) == 0 else self._count_a_in_b_unique(prediction, ground_truth) / float(
            len(ground_truth))
        assert 0 <= recall_score <= 1
        return recall_score

    def mrr(self, ground_truth, prediction):
        """
        Compute Mean Reciprocal Rank metric. Reciprocal Rank is set 0 if no predicted item is in contained the ground truth.
        :param ground_truth: the ground truth set or sequence
        :param prediction: the predicted set or sequence
        :return: the value of the metric
        """
        rr = 0.
        for rank, p in enumerate(prediction):
            if p in ground_truth:
                rr = 1. / (rank + 1)
                break
        return rr

    @staticmethod
    def _count_a_in_b_unique(a, b):
        """
        :param a: list of lists
        :param b: list of lists
        :return: number of elements of a in b
        """
        count = 0
        for el in a:
            if el in b:
                count += 1
        return count

    @staticmethod
    def _remove_duplicates(l):
        return [list(x) for x in set(tuple(x) for x in l)]