"""
`model`: One among the following KNN models:
    - `iknn`: ItemKNN, item-to-item KNN based on the *last* item in the session to determine the items to be recommended.
    - `sknn`: SessionKNN, compares the *entire* current session with the past sessions in the training data to determine the items to be recommended.
    - `v-sknn`: VMSessionKNN, use linearly decayed real-valued vectors to encode the current session, then compares the current session with the past sessions in the training data using the dot-product to determine the items to be recommended.
    - `s-sknn`: SeqSessionKNN, this variant also puts more weight on elements that appear later in the session by using a custom scoring function (see the paper by Ludewng and Jannach).
    - `sf-sknn`: SeqFilterSessionKNN, this variant also puts more weight on elements that appear later in the session in a more restrictive way by using a custom scoring function (see the paper by Ludewng and Jannach).
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd

from src.models.abstract import ISeqRecommender


def dataset_to_gru4rec_format(dataset):
    """
    Convert a list of sequences to GRU4Rec format.
    Based on this StackOverflow answer: https://stackoverflow.com/a/48532692

    :param dataset: the dataset to be transformed
    """

    lst_col = 'sequence'
    df = dataset.reset_index()
    unstacked = pd.DataFrame({
        col: np.repeat(df[col].values, df[lst_col].str.len()) for col in df.columns.drop(lst_col)}
    ).assign(**{lst_col: np.concatenate(df[lst_col].values)})[df.columns]
    # ensure that events in the session have increasing timestamps
    unstacked['ts'] = unstacked['ts'] + unstacked.groupby('user_id').cumcount()
    unstacked.rename(columns={'sequence': 'item_id'}, inplace=True)
    return unstacked


class ItemKNN:
    '''
    ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')
    
    Item-to-item predictor that computes the the similarity to all items to the given item.
    
    Similarity of two items is given by:
    
    .. math::
        s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}
        
    Parameters
    --------
    n_sims : int
        Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    lmbd : float
        Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
    alpha : float
        Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')
    
    '''

    def __init__(self, n_sims=100, lmbd=20, alpha=0.5, session_key='SessionId', item_key='ItemId', time_key='Time'):
        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key

    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        data.set_index(np.arange(len(data)), inplace=True)
        self.itemids = data[self.item_key].unique()
        n_items = len(self.itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: self.itemids, 'ItemIdx': np.arange(len(self.itemids))}),
                        on=self.item_key, how='inner')
        sessionids = data[self.session_key].unique()
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': np.arange(len(sessionids))}),
                        on=self.session_key, how='inner')
        supp = data.groupby('SessionIdx').size()
        session_offsets = np.zeros(len(supp) + 1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()
        index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values
        supp = data.groupby('ItemIdx').size()
        item_offsets = np.zeros(n_items + 1, dtype=np.int32)
        item_offsets[1:] = supp.cumsum()
        index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values
        self.sims = dict()
        for i in range(n_items):
            iarray = np.zeros(n_items)
            start = item_offsets[i]
            end = item_offsets[i + 1]
            for e in index_by_items[start:end]:
                uidx = data.SessionIdx.values[e]
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx + 1]
                user_events = index_by_sessions[ustart:uend]
                iarray[data.ItemIdx.values[user_events]] += 1
            iarray[i] = 0
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1 - self.n_sims:-1]
            self.sims[self.itemids[i]] = pd.Series(data=iarray[indices], index=self.itemids[indices])

    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        if predict_for_item_ids is None:
            predict_for_item_ids = self.itemids
        preds = np.zeros(len(predict_for_item_ids))
        sim_list = self.sims[input_item_id]
        mask = np.in1d(predict_for_item_ids, sim_list.index)
        preds[mask] = sim_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)


#collapse
class SeqFilterSessionKNN:
    '''
    SessionKNN( k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__(self, k, sample_size=1000, sampling='recent', similarity='jaccard', remind=False, pop_boost=0,
                 extend=False, normalize=True, session_key='SessionId', item_key='ItemId', time_key='Time'):

        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.pop_boost = pop_boost
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()
        self.followed_by = dict()

        self.sim_time = 0

    def fit(self, train, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''

        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_time = train.columns.get_loc(self.time_key)
        self.itemids = train[self.item_key].unique()

        session = -1
        session_items = set()
        last_item = -1
        time = -1
        # cnt = 0
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session: session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session: time})
                session = row[index_session]
                session_items = set()
            else:
                if last_item != -1:  # fill followed by map for filtering of candidate items
                    if not last_item in self.followed_by:
                        self.followed_by[last_item] = set()
                    self.followed_by[last_item].add(row[index_item])

            time = row[index_time]
            session_items.add(row[index_item])

            # cache sessions involving an item
            map_is = self.item_session_map.get(row[index_item])
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item]: map_is})
            map_is.add(row[index_session])

            last_item = row[index_item]

        # Add the last tuple    
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: time})

    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''

        #         gc.collect()
        #         process = psutil.Process(os.getpid())
        #         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')

        if (self.session != session_id):  # new session

            if (self.extend):
                item_set = set(self.session_items)
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get(item)
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item: map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time.update({self.session: ts})

                last_item = -1
                for item in self.session_items:
                    if last_item != -1:
                        if not last_item in self.followed_by:
                            self.followed_by[last_item] = set()
                        self.followed_by[last_item].add(item)
                    last_item = item

            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append(input_item_id)

        if skip:
            return

        neighbors = self.find_neighbors(set(self.session_items), input_item_id, session_id)
        scores = self.score_items(neighbors, input_item_id)

        # add some reminders
        if self.remind:

            reminderScore = 5
            takeLastN = 3

            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                # reminderScore = reminderScore + (cnt/100)

                oldScore = scores.get(elem)
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                # print 'old score ', oldScore
                # update the score and add a small number for the position 
                newScore = (newScore * reminderScore) + (cnt / 100)

                scores.update({elem: newScore})

        # push popular ones
        if self.pop_boost > 0:

            pop = self.item_pop(neighbors)
            # Iterate over the item neighbors
            # print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key: (scores[key] + (self.pop_boost * item_pop))})

        # Create things in the format ..
        if predict_for_item_ids is None:
            predict_for_item_ids = self.itemids
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, list(scores.keys()))

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.normalize:
            series = series / series.max()

        return series

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)
        
        Parameters
        --------
        sessions: set
        
        Returns
        --------
        out : dict            
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session(session)
            for item in items:

                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})

                if (result.get(item) > max_pop):
                    max_pop = result.get(item)

        for key in result:
            result.update({key: (result[key] / max_pop)})

        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second)
        res = intersection / union

        self.sim_time += (time.clock() - sc)

        return res

    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result

    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / (la + lb - li)

        return result

    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first & second)
        b = len(first)
        c = len(second)

        result = (2 * a) / ((2 * a) + b + c)

        return result

    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);

    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get(item_id)

    def most_recent_sessions(self, sessions, number):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get(session)
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        # print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        # print 'returning sample of size ', len(sample)
        return sample

    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''

        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id);

        if self.sample_size == 0:  # use all session as possible neighbors

            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else:  # sample some sessions

            self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id);

            if len(self.relevant_sessions) > self.sample_size:

                if self.sampling == 'recent':
                    sample = self.most_recent_sessions(self.relevant_sessions, self.sample_size)
                elif self.sampling == 'random':
                    sample = random.sample(self.relevant_sessions, self.sample_size)
                else:
                    sample = self.relevant_sessions[:self.sample_size]

                return sample
            else:
                return self.relevant_sessions

    def calc_similarity(self, session_items, sessions):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''

        # print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            session_items_test = self.items_for_session(session)

            similarity = getattr(self, self.similarity)(session_items_test, session_items)
            if similarity > 0:
                neighbors.append((session, similarity))

        return neighbors

    # -----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    # -----------------
    def find_neighbors(self, session_items, input_item_id, session_id):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        possible_neighbors = self.possible_neighbor_sessions(session_items, input_item_id, session_id)
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors)

        possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
        possible_neighbors = possible_neighbors[:self.k]

        return possible_neighbors

    def score_items(self, neighbors, input_item_id):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session(session[0])

            for item in items:

                if input_item_id in self.followed_by and item in self.followed_by[
                    input_item_id]:  # hard filter the candidates

                    old_score = scores.get(item)
                    new_score = session[1]

                    if old_score is None:
                        scores.update({item: new_score})
                    else:
                        new_score = old_score + new_score
                        scores.update({item: new_score})

        return scores


#collapse
class VMSessionKNN:
    '''
    VMSessionKNN( k, sample_size=1000, sampling='recent', similarity='cosine', weighting='div', dwelling_time=False, last_n_days=None, last_n_clicks=None, extend=False, weighting_score='div_score', weighting_time=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    weighting : string
        Decay function to determine the importance/weight of individual actions in the current session (linear, same, div, log, quadratic). (default: div)
    weighting_score : string
        Decay function to lower the score of candidate items from a neighboring sessions that were selected by less recently clicked items in the current session. (linear, same, div, log, quadratic). (default: div_score)
    weighting_time : boolean
        Experimental function to give less weight to items from older sessions (default: False)
    dwelling_time : boolean
        Experimental function to use the dwelling time for item view actions as a weight in the similarity calculation. (default: False)
    last_n_days : int
        Use only data from the last N days. (default: None)
    last_n_clicks : int
        Use only the last N clicks of the current session when recommending. (default: None)
    extend : bool
        Add evaluated sessions to the maps.
    normalize : bool
        Normalize the scores in the end.
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__(self, k, sample_size=1000, sampling='recent', similarity='cosine', weighting='div',
                 dwelling_time=False, last_n_days=None, last_n_clicks=None, extend=False, weighting_score='div_score',
                 weighting_time=False, normalize=True, session_key='SessionId', item_key='ItemId', time_key='Time'):

        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.weighting = weighting
        self.dwelling_time = dwelling_time
        self.weighting_score = weighting_score
        self.weighting_time = weighting_time
        self.similarity = similarity
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize
        self.last_n_days = last_n_days
        self.last_n_clicks = last_n_clicks

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()
        self.min_time = -1

        self.sim_time = 0

    def fit(self, data, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''

        if self.last_n_days != None:

            max_time = dt.fromtimestamp(data[self.time_key].max())
            date_threshold = max_time.date() - td(self.last_n_days)
            stamp = dt.combine(date_threshold, dt.min.time()).timestamp()
            train = data[data[self.time_key] >= stamp]

        else:
            train = data

        self.num_items = train[self.item_key].max()

        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_time = train.columns.get_loc(self.time_key)
        self.itemids = train[self.item_key].unique()

        session = -1
        session_items = set()
        time = -1
        # cnt = 0
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session: session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session: time})
                    if time < self.min_time:
                        self.min_time = time
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])

            # cache sessions involving an item
            map_is = self.item_session_map.get(row[index_item])
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item]: map_is})
            map_is.add(row[index_session])

        # Add the last tuple    
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: time})

    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''

        #         gc.collect()
        #         process = psutil.Process(os.getpid())
        #         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')

        if (self.session != session_id):  # new session

            if (self.extend):
                item_set = set(self.session_items)
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get(item)
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item: map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time.update({self.session: ts})

            self.last_ts = -1
            self.session = session_id
            self.session_items = list()
            self.dwelling_times = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append(input_item_id)
            if self.dwelling_time:
                if self.last_ts > 0:
                    self.dwelling_times.append(timestamp - self.last_ts)
                self.last_ts = timestamp

        if skip:
            return

        items = self.session_items if self.last_n_clicks is None else self.session_items[-self.last_n_clicks:]
        neighbors = self.find_neighbors(items, input_item_id, session_id, self.dwelling_times, timestamp)
        scores = self.score_items(neighbors, items, timestamp)

        # Create things in the format ..
        if predict_for_item_ids is None:
            predict_for_item_ids = self.itemids
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, list(scores.keys()))

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.normalize:
            series = series / series.max()

        return series

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)
        
        Parameters
        --------
        sessions: set
        
        Returns
        --------
        out : dict            
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session(session)
            for item in items:

                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})

                if (result.get(item) > max_pop):
                    max_pop = result.get(item)

        for key in result:
            result.update({key: (result[key] / max_pop)})

        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second)
        res = intersection / union

        self.sim_time += (time.clock() - sc)

        return res

    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result

    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / (la + lb - li)

        return result

    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first & second)
        b = len(first)
        c = len(second)

        result = (2 * a) / ((2 * a) + b + c)

        return result

    def vec(self, first, second, map):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = first & second
        sum = 0
        for i in a:
            sum += map[i]

        result = sum / len(map)

        return result

    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);

    def vec_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_vec_map.get(session);

    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get(item_id) if item_id in self.item_session_map else set()

    def most_recent_sessions(self, sessions, number):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get(session)
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        # print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        # print 'returning sample of size ', len(sample)
        return sample

    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''

        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id)

        if self.sample_size == 0:  # use all session as possible neighbors

            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else:  # sample some sessions

            if len(self.relevant_sessions) > self.sample_size:

                if self.sampling == 'recent':
                    sample = self.most_recent_sessions(self.relevant_sessions, self.sample_size)
                elif self.sampling == 'random':
                    sample = random.sample(self.relevant_sessions, self.sample_size)
                else:
                    sample = self.relevant_sessions[:self.sample_size]

                return sample
            else:
                return self.relevant_sessions

    def calc_similarity(self, session_items, sessions, dwelling_times, timestamp):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''

        pos_map = {}
        length = len(session_items)

        count = 1
        for item in session_items:
            if self.weighting is not None:
                pos_map[item] = getattr(self, self.weighting)(count, length)
                count += 1
            else:
                pos_map[item] = 1

        dt = dwelling_times.copy()
        dt.append(0)
        dt = pd.Series(dt, index=session_items)
        dt = dt / dt.max()
        # dt[session_items[-1]] = dt.mean() if len(session_items) > 1 else 1
        dt[session_items[-1]] = 1

        if self.dwelling_time:
            # print(dt)
            for i in range(len(dt)):
                pos_map[session_items[i]] *= dt.iloc[i]
            # print(pos_map)
        # print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        items = set(session_items)
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            n_items = self.items_for_session(session)
            sts = self.session_time[session]

            similarity = self.vec(items, n_items, pos_map)
            if similarity > 0:

                if self.weighting_time:
                    diff = timestamp - sts
                    days = round(diff / 60 / 60 / 24)
                    decay = pow(7 / 8, days)
                    similarity *= decay

                # print("days:",days," => ",decay)

                neighbors.append((session, similarity))

        return neighbors

    # -----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    # -----------------
    def find_neighbors(self, session_items, input_item_id, session_id, dwelling_times, timestamp):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        possible_neighbors = self.possible_neighbor_sessions(session_items, input_item_id, session_id)
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors, dwelling_times, timestamp)

        possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
        possible_neighbors = possible_neighbors[:self.k]

        return possible_neighbors

    def score_items(self, neighbors, current_session, timestamp):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session(session[0])
            step = 1

            for item in reversed(current_session):
                if item in items:
                    decay = getattr(self, self.weighting_score)(step)
                    break
                step += 1

            for item in items:
                old_score = scores.get(item)
                similarity = session[1]

                if old_score is None:
                    scores.update({item: (similarity * decay)})
                else:
                    new_score = old_score + (similarity * decay)
                    scores.update({item: new_score})

        return scores

    def linear_score(self, i):
        return 1 - (0.1 * i) if i <= 100 else 0

    def same_score(self, i):
        return 1

    def div_score(self, i):
        return 1 / i

    def log_score(self, i):
        return 1 / (log10(i + 1.7))

    def quadratic_score(self, i):
        return 1 / (i * i)

    def linear(self, i, length):
        return 1 - (0.1 * (length - i)) if i <= 10 else 0

    def same(self, i, length):
        return 1

    def div(self, i, length):
        return i / length

    def log(self, i, length):
        return 1 / (log10((length - i) + 1.7))

    def quadratic(self, i, length):
        return (i / length) ** 2


#collapse
class SessionKNN:
    '''
    SessionKNN( k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__(self, k, sample_size=1000, sampling='recent', similarity='jaccard', remind=False, pop_boost=0,
                 extend=False, normalize=True, session_key='SessionId', item_key='ItemId', time_key='Time'):

        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.pop_boost = pop_boost
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()

        self.sim_time = 0

    def fit(self, train):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''

        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_time = train.columns.get_loc(self.time_key)
        self.itemids = train[self.item_key].unique()

        session = -1
        session_items = set()
        time = -1
        # cnt = 0
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session: session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session: time})
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])

            # cache sessions involving an item
            map_is = self.item_session_map.get(row[index_item])
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item]: map_is})
            map_is.add(row[index_session])

        # Add the last tuple    
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: time})

    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''

        #         gc.collect()
        #         process = psutil.Process(os.getpid())
        #         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')

        if (self.session != session_id):  # new session

            if (self.extend):
                item_set = set(self.session_items)
                self.session_item_map[self.session] = item_set;
                for item in item_set:
                    map_is = self.item_session_map.get(item)
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item: map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time.update({self.session: ts})

            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append(input_item_id)

        if skip:
            return

        neighbors = self.find_neighbors(set(self.session_items), input_item_id, session_id)
        scores = self.score_items(neighbors)

        # add some reminders
        if self.remind:

            reminderScore = 5
            takeLastN = 3

            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                # reminderScore = reminderScore + (cnt/100)

                oldScore = scores.get(elem)
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                # print 'old score ', oldScore
                # update the score and add a small number for the position 
                newScore = (newScore * reminderScore) + (cnt / 100)

                scores.update({elem: newScore})

        # push popular ones
        if self.pop_boost > 0:

            pop = self.item_pop(neighbors)
            # Iterate over the item neighbors
            # print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key: (scores[key] + (self.pop_boost * item_pop))})

        # Create things in the format ..
        if predict_for_item_ids is None:
            predict_for_item_ids = self.itemids
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, list(scores.keys()))

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.normalize:
            series = series / series.max()

        return series

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)
        
        Parameters
        --------
        sessions: set
        
        Returns
        --------
        out : dict            
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session(session)
            for item in items:

                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})

                if (result.get(item) > max_pop):
                    max_pop = result.get(item)

        for key in result:
            result.update({key: (result[key] / max_pop)})

        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second)
        res = intersection / union

        self.sim_time += (time.clock() - sc)

        return res

    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result

    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / (la + lb - li)

        return result

    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first & second)
        b = len(first)
        c = len(second)

        result = (2 * a) / ((2 * a) + b + c)

        return result

    def random(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        return random.random()

    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);

    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get(item_id)

    def most_recent_sessions(self, sessions, number):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get(session)
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        # print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        # print 'returning sample of size ', len(sample)
        return sample

    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''

        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id);

        if self.sample_size == 0:  # use all session as possible neighbors

            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else:  # sample some sessions

            self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id);

            if len(self.relevant_sessions) > self.sample_size:

                if self.sampling == 'recent':
                    sample = self.most_recent_sessions(self.relevant_sessions, self.sample_size)
                elif self.sampling == 'random':
                    sample = random.sample(self.relevant_sessions, self.sample_size)
                else:
                    sample = self.relevant_sessions[:self.sample_size]

                return sample
            else:
                return self.relevant_sessions

    def calc_similarity(self, session_items, sessions):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''

        # print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            session_items_test = self.items_for_session(session)

            similarity = getattr(self, self.similarity)(session_items_test, session_items)
            if similarity > 0:
                neighbors.append((session, similarity))

        return neighbors

    # -----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    # -----------------
    def find_neighbors(self, session_items, input_item_id, session_id):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        possible_neighbors = self.possible_neighbor_sessions(session_items, input_item_id, session_id)
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors)

        possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
        possible_neighbors = possible_neighbors[:self.k]

        return possible_neighbors

    def score_items(self, neighbors):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session(session[0])

            for item in items:
                old_score = scores.get(item)
                new_score = session[1]

                if old_score is None:
                    scores.update({item: new_score})
                else:
                    new_score = old_score + new_score
                    scores.update({item: new_score})

        return scores


#collapse
class SeqSessionKNN:
    '''
    SeqSessionKNN( k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__(self, k, sample_size=1000, sampling='recent', similarity='jaccard', weighting='div', remind=False,
                 pop_boost=0, extend=False, normalize=True, session_key='SessionId', item_key='ItemId',
                 time_key='Time'):

        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.weighting = weighting
        self.similarity = similarity
        self.pop_boost = pop_boost
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.extend = extend
        self.normalize = normalize

        # updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()

        self.sim_time = 0

    def fit(self, train, items=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''

        index_session = train.columns.get_loc(self.session_key)
        index_item = train.columns.get_loc(self.item_key)
        index_time = train.columns.get_loc(self.time_key)
        self.itemids = train[self.item_key].unique()

        session = -1
        session_items = set()
        time = -1
        # cnt = 0
        for row in train.itertuples(index=False):
            # cache items of sessions
            if row[index_session] != session:
                if len(session_items) > 0:
                    self.session_item_map.update({session: session_items})
                    # cache the last time stamp of the session
                    self.session_time.update({session: time})
                session = row[index_session]
                session_items = set()
            time = row[index_time]
            session_items.add(row[index_item])

            # cache sessions involving an item
            map_is = self.item_session_map.get(row[index_item])
            if map_is is None:
                map_is = set()
                self.item_session_map.update({row[index_item]: map_is})
            map_is.add(row[index_session])

        # Add the last tuple    
        self.session_item_map.update({session: session_items})
        self.session_time.update({session: time})

    def predict_next(self, session_id, input_item_id, predict_for_item_ids=None, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''

        #         gc.collect()
        #         process = psutil.Process(os.getpid())
        #         print( 'cknn.predict_next: ', process.memory_info().rss, ' memory used')

        if (self.session != session_id):  # new session

            if (self.extend):
                item_set = set(self.session_items)
                self.session_item_map[self.session] = item_set
                for item in item_set:
                    map_is = self.item_session_map.get(item)
                    if map_is is None:
                        map_is = set()
                        self.item_session_map.update({item: map_is})
                    map_is.add(self.session)

                ts = time.time()
                self.session_time.update({self.session: ts})

            self.session = session_id
            self.session_items = list()
            self.relevant_sessions = set()

        if type == 'view':
            self.session_items.append(input_item_id)

        if skip:
            return

        neighbors = self.find_neighbors(set(self.session_items), input_item_id, session_id)
        scores = self.score_items(neighbors, self.session_items)

        # add some reminders
        if self.remind:

            reminderScore = 5
            takeLastN = 3

            cnt = 0
            for elem in self.session_items[-takeLastN:]:
                cnt = cnt + 1
                # reminderScore = reminderScore + (cnt/100)

                oldScore = scores.get(elem)
                newScore = 0
                if oldScore is None:
                    newScore = reminderScore
                else:
                    newScore = oldScore + reminderScore
                # print 'old score ', oldScore
                # update the score and add a small number for the position 
                newScore = (newScore * reminderScore) + (cnt / 100)

                scores.update({elem: newScore})

        # push popular ones
        if self.pop_boost > 0:

            pop = self.item_pop(neighbors)
            # Iterate over the item neighbors
            # print itemScores
            for key in scores:
                item_pop = pop.get(key)
                # Gives some minimal MRR boost?
                scores.update({key: (scores[key] + (self.pop_boost * item_pop))})

        # Create things in the format ..
        if predict_for_item_ids is None:
            predict_for_item_ids = self.itemids
        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, list(scores.keys()))

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.normalize:
            series = series / series.max()

        return series

    def item_pop(self, sessions):
        '''
        Returns a dict(item,score) of the item popularity for the given list of sessions (only a set of ids)
        
        Parameters
        --------
        sessions: set
        
        Returns
        --------
        out : dict            
        '''
        result = dict()
        max_pop = 0
        for session, weight in sessions:
            items = self.items_for_session(session)
            for item in items:

                count = result.get(item)
                if count is None:
                    result.update({item: 1})
                else:
                    result.update({item: count + 1})

                if (result.get(item) > max_pop):
                    max_pop = result.get(item)

        for key in result:
            result.update({key: (result[key] / max_pop)})

        return result

    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second)
        res = intersection / union

        self.sim_time += (time.clock() - sc)

        return res

    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / sqrt(la) * sqrt(lb)

        return result

    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first & second)
        la = len(first)
        lb = len(second)
        result = li / (la + lb - li)

        return result

    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first & second)
        b = len(first)
        c = len(second)

        result = (2 * a) / ((2 * a) + b + c)

        return result

    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);

    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get(item_id)

    def most_recent_sessions(self, sessions, number):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get(session)
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))

        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        # print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add(element[0])
        # print 'returning sample of size ', len(sample)
        return sample

    def possible_neighbor_sessions(self, session_items, input_item_id, session_id):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''

        self.relevant_sessions = self.relevant_sessions | self.sessions_for_item(input_item_id);

        if self.sample_size == 0:  # use all session as possible neighbors

            print('!!!!! runnig KNN without a sample size (check config)')
            return self.relevant_sessions

        else:  # sample some sessions

            if len(self.relevant_sessions) > self.sample_size:

                if self.sampling == 'recent':
                    sample = self.most_recent_sessions(self.relevant_sessions, self.sample_size)
                elif self.sampling == 'random':
                    sample = random.sample(self.relevant_sessions, self.sample_size)
                else:
                    sample = self.relevant_sessions[:self.sample_size]

                return sample
            else:
                return self.relevant_sessions

    def calc_similarity(self, session_items, sessions):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''

        # print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        neighbors = []
        cnt = 0
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            session_items_test = self.items_for_session(session)

            similarity = getattr(self, self.similarity)(session_items_test, session_items)
            if similarity > 0:
                neighbors.append((session, similarity))

        return neighbors

    # -----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    # -----------------
    def find_neighbors(self, session_items, input_item_id, session_id):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        possible_neighbors = self.possible_neighbor_sessions(session_items, input_item_id, session_id)
        possible_neighbors = self.calc_similarity(session_items, possible_neighbors)

        possible_neighbors = sorted(possible_neighbors, reverse=True, key=lambda x: x[1])
        possible_neighbors = possible_neighbors[:self.k]

        return possible_neighbors

    def score_items(self, neighbors, current_session):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session(session[0])
            step = 1

            for item in reversed(current_session):
                if item in items:
                    decay = getattr(self, self.weighting)(step)
                    break
                step += 1

            for item in items:
                old_score = scores.get(item)
                similarity = session[1]

                if old_score is None:
                    scores.update({item: (similarity * decay)})
                else:
                    new_score = old_score + (similarity * decay)
                    scores.update({item: new_score})

        return scores

    def linear(self, i):
        return 1 - (0.1 * i) if i <= 100 else 0

    def same(self, i):
        return 1

    def div(self, i):
        return 1 / i

    def log(self, i):
        return 1 / (log10(i + 1.7))

    def quadratic(self, i):
        return 1 / (i * i)


#collapse
class KNNRecommender(ISeqRecommender):
    """
    Interface to ItemKNN and Session-based KNN methods. Based on:

    Evaluation of Session-based Recommendation Algorithms, Malte Ludewig and Dietmar Jannach
    """
    knn_models = {
        'iknn': ItemKNN,
        'sknn': SessionKNN,
        'v-sknn': VMSessionKNN,
        's-sknn': SeqSessionKNN,
        'sf-sknn': SeqFilterSessionKNN
    }

    def __init__(self,
                 model='cknn',
                 **init_args):
        """
        :param model: One among the following KNN models:
            - iknn: ItemKNN, item-to-item KNN based on the *last* item in the session to determine the items to be recommended.
            - sknn: SessionKNN, compares the *entire* current session with the past sessions in the training data to
                    determine the items to be recommended.
            - v-sknn: VMSessionKNN, use linearly decayed real-valued vectors to encode the current session,
                    then compares the current session with the past sessions in the training data using the dot-product
                    to determine the items to be recommended.
            - s-sknn: SeqSessionKNN, this variant also puts more weight on elements that appear later in the session by
                using a custom scoring function (see the paper by Ludewng and Jannach).
            - sf-sknn: SeqFilterSessionKNN, this variant also puts more weight on elements that appear later in the session
                in a more restrictive way by using a custom scoring function (see the paper by Ludewng and Jannach).

        :param init_args: The model initialization arguments. See the following initializations or
            check `util.knn` for more details on each model:
            - iknn: ItemKNN(n_sims=100, lmbd=20, alpha=0.5)
            - sknn: SessionKNN(k, sample_size=500, sampling='recent', similarity='jaccard', remind=False, pop_boost=0)
            - v-sknn: VMSessionKNN(k, sample_size=1000, sampling='recent', similarity='cosine', weighting='div',
                 dwelling_time=False, last_n_days=None, last_n_clicks=None, extend=False, weighting_score='div_score',
                 weighting_time=False, normalize=True)
            - s-knn: SeqSessionKNN(k, sample_size=1000, sampling='recent', similarity='jaccard', weighting='div',
                remind=False, pop_boost=0, extend=False, normalize=True)
            - sf-sknn: SeqFilterSessionKNN(k, sample_size=1000, sampling='recent', similarity='jaccard', remind=False, pop_boost=0,
                 extend=False, normalize=True)
        """
        super(KNNRecommender).__init__()
        if model not in self.knn_models:
            raise ValueError("Unknown KNN model '{}'. The available ones are: {}".format(
                model, list(self.knn_models.keys())
            ))
        self.init_args = init_args
        self.init_args.update(dict(session_key='session_id',
                                   item_key='item_id',
                                   time_key='ts'))
        self.model = self.knn_models[model](**self.init_args)
        self.pseudo_session_id = 0

    def __str__(self):
        return str(self.model)

    def fit(self, train_data):
        self.logger.info('Converting training data to GRU4Rec format')
        # parse training data to GRU4Rec format
        train_data = dataset_to_gru4rec_format(dataset=train_data)

        self.logger.info('Training started')
        self.model.fit(train_data)
        self.logger.info('Training completed')
        self.pseudo_session_id = 0

    def recommend(self, user_profile, user_id=None):
        for item in user_profile:
            pred = self.model.predict_next(session_id=self.pseudo_session_id,
                                           input_item_id=item)
        # sort items by predicted score
        pred.sort_values(0, ascending=False, inplace=True)
        # increase the psuedo-session id so that future call to recommend() won't be connected
        self.pseudo_session_id += 1
        # convert to the required output format
        return [([x.index], x._2) for x in pred.reset_index().itertuples()]