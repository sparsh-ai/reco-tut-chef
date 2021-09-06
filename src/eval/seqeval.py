import operator
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.eval.metrics import EvalMetrics


class SequentialEvaluator:
    """
    In the evaluation of sequence-aware recommenders, each sequence in the test set is split into:
    - the user profile, used to compute recommendations, is composed by the first k events in the sequence;
    - the ground truth, used for performance evaluation, is composed by the remainder of the sequence.
    
    you can control the dimension of the user profile by assigning a positive value to GIVEN_K,
    which correspond to the number of events from the beginning of the sequence that will be assigned
    to the initial user profile. This ensures that each user profile in the test set will have exactly
    the same initial size, but the size of the ground truth will change for every sequence.

    Alternatively, by assigning a negative value to GIVEN_K, you will set the initial size of the ground truth.
    In this way the ground truth will have the same size for all sequences, but the dimension of the user
    profile will differ.
    """

    def __init__(self, train_data, test_data, recommender):
        self.test_data = test_data
        self.recommender = recommender
        self.train_users = train_data['user_id'].values
        self.evalmetrics = EvalMetrics()
        self.evaluation_functions = {'precision':self.evalmetrics.precision,
                        'recall':self.evalmetrics.recall,
                        'mrr': self.evalmetrics.mrr}

    def get_test_sequences(self, given_k):
        # we can run evaluation only over sequences longer than abs(LAST_K)
        test_sequences = self.test_data.loc[self.test_data['sequence'].map(len) > abs(given_k), 'sequence'].values
        return test_sequences

    def get_test_sequences_and_users(self, given_k):
        # we can run evaluation only over sequences longer than abs(LAST_K)
        mask = self.test_data['sequence'].map(len) > abs(given_k)
        mask &= self.test_data['user_id'].isin(self.train_users)
        test_sequences = self.test_data.loc[mask, 'sequence'].values
        test_users = self.test_data.loc[mask, 'user_id'].values
        return test_sequences, test_users

    def sequential_evaluation(self, test_sequences, users=None, given_k=1,
                              look_ahead=1, top_n=10, scroll=True, step=1):
        """
        Runs sequential evaluation of a recommender over a set of test sequences
        :param recommender: the instance of the recommender to test
        :param test_sequences: the set of test sequences
        :param evaluation_functions: list of evaluation metric functions
        :param users: (optional) the list of user ids associated to each test sequence. Required by personalized models like FPMC.
        :param given_k: (optional) the initial size of each user profile, starting from the first interaction in the sequence.
                        If <0, start counting from the end of the sequence. It must be != 0.
        :param look_ahead: (optional) number of subsequent interactions in the sequence to be considered as ground truth.
                        It can be any positive number or 'all' to extend the ground truth until the end of the sequence.
        :param top_n: (optional) size of the recommendation list
        :param scroll: (optional) whether to scroll the ground truth until the end of the sequence.
                    If True, expand the user profile and move the ground truth forward of `step` interactions. Recompute and evaluate recommendations every time.
                    If False, evaluate recommendations once per sequence without expanding the user profile.
        :param step: (optional) number of interactions that will be added to the user profile at each step of the sequential evaluation.
        :return: the list of the average values for each evaluation metric
        """
        if given_k == 0:
            raise ValueError('given_k must be != 0')

        evaluation_functions = self.evaluation_functions.values()

        metrics = np.zeros(len(evaluation_functions))
        with tqdm(total=len(test_sequences)) as pbar:
            for i, test_seq in enumerate(test_sequences):
                if users is not None:
                    user = users[i]
                else:
                    user = None
                if scroll:
                    metrics += self.sequence_sequential_evaluation(test_seq,
                                                                   user,
                                                                   given_k,
                                                                   look_ahead,
                                                                   top_n,
                                                                   step)
                else:
                    metrics += self.evaluate_sequence(test_seq, 
                                                      user,
                                                      given_k,
                                                      look_ahead,
                                                      top_n)
                pbar.update(1)

        return metrics / len(test_sequences)

    def evaluate_sequence(self, seq, user, given_k, look_ahead, top_n):
        """
        :param recommender: which recommender to use
        :param seq: the user_profile/ context
        :param given_k: last element used as ground truth. NB if <0 it is interpreted as first elements to keep
        :param evaluation_functions: which function to use to evaluate the rec performance
        :param look_ahead: number of elements in ground truth to consider. if look_ahead = 'all' then all the ground_truth sequence is considered
        :return: performance of recommender
        """
        # safety checks
        if given_k < 0:
            given_k = len(seq) + given_k

        user_profile = seq[:given_k]
        ground_truth = seq[given_k:]

        # restrict ground truth to look_ahead
        ground_truth = ground_truth[:look_ahead] if look_ahead != 'all' else ground_truth
        ground_truth = list(map(lambda x: [x], ground_truth))  # list of list format

        user_profile = list(user_profile)
        ground_truth = list(ground_truth)
        evaluation_functions = self.evaluation_functions.values()

        if not user_profile or not ground_truth:
            # if any of the two missing all evaluation functions are 0
            return np.zeros(len(evaluation_functions))

        r = self.recommender.recommend(user_profile, user)[:top_n]

        if not r:
            # no recommendation found
            return np.zeros(len(evaluation_functions))
        reco_list = self.recommender.get_recommendation_list(r)

        tmp_results = []
        for f in evaluation_functions:
            tmp_results.append(f(ground_truth, reco_list))
        return np.array(tmp_results)

    def sequence_sequential_evaluation(self, seq, user, given_k, look_ahead, top_n, step):
        if given_k < 0:
            given_k = len(seq) + given_k

        eval_res = 0.0
        eval_cnt = 0
        for gk in range(given_k, len(seq), step):
            eval_res += self.evaluate_sequence(seq,
                                                user,
                                                gk,
                                                look_ahead,
                                                top_n)
            eval_cnt += 1
        return eval_res / eval_cnt

    def eval_seqreveal(self, user_flg=0, GIVEN_K=1, LOOK_AHEAD=1, STEP=1, TOPN=20):
        if user_flg:
            test_sequences, test_users = self.get_test_sequences_and_users(GIVEN_K)
            print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))
            results = self.sequential_evaluation(test_sequences,
                                                users=test_users,
                                                given_k=GIVEN_K,
                                                look_ahead=LOOK_AHEAD,
                                                top_n=TOPN,
                                                scroll=True,  # scrolling averages metrics over all profile lengths
                                                step=STEP)
        else:
            test_sequences = self.get_test_sequences(GIVEN_K)
            print('{} sequences available for evaluation'.format(len(test_sequences)))
            results = self.sequential_evaluation(test_sequences,
                                                given_k=GIVEN_K,
                                                look_ahead=LOOK_AHEAD,
                                                top_n=TOPN,
                                                scroll=True,  # scrolling averages metrics over all profile lengths
                                                step=STEP)
        
        # print('Sequential evaluation (GIVEN_K={}, LOOK_AHEAD={}, STEP={})'.format(GIVEN_K, LOOK_AHEAD, STEP))
        # for mname, mvalue in zip(self.evaluation_functions.keys(), results):
        #     print('\t{}@{}: {:.4f}'.format(mname, TOPN, mvalue))
        return [results, GIVEN_K, LOOK_AHEAD, STEP]  


    def eval_staticprofile(self, user_flg=0, GIVEN_K=1, LOOK_AHEAD='all', STEP=1, TOPN=20):
        if user_flg:
            test_sequences, test_users = self.get_test_sequences_and_users(GIVEN_K) # we need user ids now!
            print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))
            results = self.sequential_evaluation(test_sequences,
                                                users=test_users,
                                                given_k=GIVEN_K,
                                                look_ahead=LOOK_AHEAD,
                                                top_n=TOPN,
                                                scroll=False  # notice that scrolling is disabled!
                                            )                                
        else:
            test_sequences = self.get_test_sequences(GIVEN_K)
            print('{} sequences available for evaluation'.format(len(test_sequences)))
            results = self.sequential_evaluation(test_sequences,
                                                 given_k=GIVEN_K,
                                                 look_ahead=LOOK_AHEAD,
                                                 top_n=TOPN,
                                                 scroll=False  # notice that scrolling is disabled!
                                                 )
            
        return [results, GIVEN_K, LOOK_AHEAD, STEP] 

    def eval_reclength(self, user_flg=0, GIVEN_K=1, LOOK_AHEAD=1, STEP=1,
                       topn_list=[1,5,10,20,50,100], TOPN=20):
        res_list = []

        if user_flg:
            test_sequences, test_users = self.get_test_sequences_and_users(GIVEN_K) # we need user ids now!
            print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))
            for topn in topn_list:
                print('Evaluating recommendation lists with length: {}'.format(topn)) 
                res_tmp = self.sequential_evaluation(test_sequences,
                                                        users=test_users,
                                                        given_k=GIVEN_K,
                                                        look_ahead=LOOK_AHEAD,
                                                        top_n=topn,
                                                        scroll=True,  # here we average over all profile lengths
                                                        step=STEP
                                                )
                mvalues = list(zip(self.evaluation_functions.keys(), res_tmp))
                res_list.append((topn, mvalues))                            
        else:
            test_sequences = self.get_test_sequences(GIVEN_K)
            print('{} sequences available for evaluation'.format(len(test_sequences)))
            for topn in topn_list:
                print('Evaluating recommendation lists with length: {}'.format(topn))      
                res_tmp = self.sequential_evaluation(test_sequences,
                                                    given_k=GIVEN_K,
                                                    look_ahead=LOOK_AHEAD,
                                                    top_n=topn,
                                                    scroll=True,  # here we average over all profile lengths
                                                    step=STEP)
                mvalues = list(zip(self.evaluation_functions.keys(), res_tmp))
                res_list.append((topn, mvalues))

        # show separate plots per metric
        # fig, axes = plt.subplots(nrows=1, ncols=len(self.evaluation_functions), figsize=(15,5))
        res_list_t = list(zip(*res_list))
        results = []
        # for midx, metric in enumerate(self.evaluation_functions):
        #     mvalues = [res_list_t[1][j][midx][1] for j in range(len(res_list_t[1]))]
        #     fig, ax = plt.subplots(figsize=(5,5))
        #     ax.plot(topn_list, mvalues)
        #     ax.set_title(metric)
        #     ax.set_xticks(topn_list)
        #     ax.set_xlabel('List length')
        #     fig.tight_layout()
        #     results.append(fig)
        return [results, GIVEN_K, LOOK_AHEAD, STEP]

    def eval_profilelength(self, user_flg=0, given_k_list=[1,2,3,4], 
                           LOOK_AHEAD=1, STEP=1, TOPN=20):
        res_list = []

        if user_flg:
            test_sequences, test_users = self.get_test_sequences_and_users(max(given_k_list)) # we need user ids now!
            print('{} sequences available for evaluation ({} users)'.format(len(test_sequences), len(np.unique(test_users))))
            for gk in given_k_list:
                print('Evaluating profiles having length: {}'.format(gk))
                res_tmp = self.sequential_evaluation(test_sequences,
                                                            users=test_users,
                                                            given_k=gk,
                                                            look_ahead=LOOK_AHEAD,
                                                            top_n=TOPN,
                                                            scroll=False,  # here we stop at each profile length
                                                            step=STEP)
                mvalues = list(zip(self.evaluation_functions.keys(), res_tmp))
                res_list.append((gk, mvalues))                          
        else:
            test_sequences = self.get_test_sequences(max(given_k_list))
            print('{} sequences available for evaluation'.format(len(test_sequences)))
            for gk in given_k_list:
                print('Evaluating profiles having length: {}'.format(gk))
                res_tmp = self.sequential_evaluation(test_sequences,
                                                            given_k=gk,
                                                            look_ahead=LOOK_AHEAD,
                                                            top_n=TOPN,
                                                            scroll=False,  # here we stop at each profile length
                                                            step=STEP)
                mvalues = list(zip(self.evaluation_functions.keys(), res_tmp))
                res_list.append((gk, mvalues))

        # show separate plots per metric
        # fig, axes = plt.subplots(nrows=1, ncols=len(self.evaluation_functions), figsize=(15,5))
        res_list_t = list(zip(*res_list))
        results = []
        # for midx, metric in enumerate(self.evaluation_functions):
        #     mvalues = [res_list_t[1][j][midx][1] for j in range(len(res_list_t[1]))]
        #     fig, ax = plt.subplots(figsize=(5,5))
        #     ax.plot(given_k_list, mvalues)
        #     ax.set_title(metric)
        #     ax.set_xticks(given_k_list)
        #     ax.set_xlabel('Profile length')
        #     fig.tight_layout()
        #     results.append(fig)
        return [results, TOPN, LOOK_AHEAD, STEP]