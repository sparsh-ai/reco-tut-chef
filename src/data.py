import numpy as np
import pandas as pd
import datetime
import calendar
import time
from collections import Counter


class SessionDataset:
    def __init__(self, df, seed=42):
        self.data = df.copy()
        self._standardize()
        self.seed = seed
        self.train = None
        self.test = None

    def _standardize(self):
        col_names = ['session_id', 'user_id', 'item_id', 'ts'] + self.data.columns.values.tolist()[4:]
        self.data.columns = col_names

    def _add_months(self, sourcedate, months):
        month = sourcedate.month - 1 + months
        year = int(sourcedate.year + month / 12)
        month = month % 12 + 1
        day = min(sourcedate.day, calendar.monthrange(year, month)[1])
        return datetime.date(year, month, day)

    def filter_by_time(self, last_months=0):
        if last_months > 0:
            lastdate = datetime.datetime.fromtimestamp(self.data.ts.max())
            firstdate = self._add_months(lastdate, -last_months)
            initial_unix = time.mktime(firstdate.timetuple())
            self.data = self.data[self.data['ts'] >= initial_unix]

    def convert_to_sequence(self, topk=0):
        c = Counter(list(self.data['item_id']))
        if topk > 1:
            keeper = set([x[0] for x in c.most_common(topk)])
            self.data = self.data[self.data['item_id'].isin(keeper)]

        # group by session id and concat song_id
        groups = self.data.groupby('session_id')

        # convert item ids to string, then aggregate them to lists
        aggregated = groups['item_id'].agg(sequence = lambda x: list(map(str, x)))
        init_ts = groups['ts'].min()
        users = groups['user_id'].min()  # it's just fast, min doesn't actually make sense

        self.data = aggregated.join(init_ts).join(users)
        self.data.reset_index(inplace=True)

    def get_stats(self):
        cnt = Counter()
        _stats = []
        self.data.sequence.map(cnt.update);
        sequence_length = self.data.sequence.map(len).values
        n_sessions_per_user = self.data.groupby('user_id').size()

        _stats.append('Number of items: {}'.format(len(cnt)))
        _stats.append('Number of users: {}'.format(self.data.user_id.nunique()))
        _stats.append('Number of sessions: {}'.format(len(self.data)) )

        _stats.append('Session length:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
            sequence_length.mean(), 
            np.quantile(sequence_length, 0.5), 
            sequence_length.min(), 
            sequence_length.max()))

        _stats.append('Sessions per user:\n\tAverage: {:.2f}\n\tMedian: {}\n\tMin: {}\n\tMax: {}'.format(
            n_sessions_per_user.mean(), 
            np.quantile(n_sessions_per_user, 0.5), 
            n_sessions_per_user.min(), 
            n_sessions_per_user.max()))

        _stats.append('Most popular items: {}'.format(cnt.most_common(5)))
        _stats =  '\n'.join(_stats)
        
        return _stats

    def random_holdout(self, split=0.8):
        """
        Split sequence data randomly
        :param split: the training percentange
        """
        self.data = self.data.sample(frac=1, random_state=self.seed)
        nseqs = len(self.data)
        train_size = int(nseqs * split)
        self.train = self.data[:train_size]
        self.test = self.data[train_size:]

    def temporal_holdout(self, ts_threshold):
        """
        Split sequence data using timestamps
        :param ts_threshold: the timestamp from which test sequences will start
        """
        self.train = self.data.loc[self.data['ts'] < ts_threshold]
        self.test = self.data.loc[self.data['ts'] >= ts_threshold]
        self.train, self.test = self._clean_split(self.train, self.test)

    def last_session_out_split(self,
                               user_key='user_id', 
                               session_key='session_id',
                               time_key='ts'):
        """
        Assign the last session of every user to the test set and the remaining ones to the training set
        """
        sessions = self.data.sort_values(by=[user_key, time_key]).groupby(user_key)[session_key]
        last_session = sessions.last()
        self.train = self.data[~self.data.session_id.isin(last_session.values)].copy()
        self.test = self.data[self.data.session_id.isin(last_session.values)].copy()
        self.train, self.test = self._clean_split(self.train, self.test)

    def _clean_split(self, train, test):
        """
        Remove new items from the test set.
        :param train: The training set.
        :param test: The test set.
        :return: The cleaned training and test sets.
        """
        train = train.copy()
        test = test.copy()
        train_items = set()
        train['sequence'].apply(lambda seq: train_items.update(set(seq)))
        test['sequence'] = test['sequence'].apply(lambda seq: [it for it in seq if it in train_items])
        return train, test