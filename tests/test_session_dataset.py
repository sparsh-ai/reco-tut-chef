import unittest
import datetime
import pandas as pd
from pandas.testing import assert_frame_equal

from src.data import SessionDataset


test_df = [[1,1,1,'2015-01-13',10],
             [2,1,1,'2015-02-13',20],
             [2,1,3,'2015-02-13',5],
             [3,1,3,'2015-02-14',15],
             [4,2,1,'2014-12-13',10],
             [5,2,2,'2015-02-10',2],
             [5,2,1,'2015-02-10',9],
             [5,2,3,'2015-02-10',3],
             [5,2,3,'2015-02-10',7],
             ]
test_df = pd.DataFrame(test_df)
test_df.columns = ['session_id', 'user_id', 'item_id', 'ts', 'playtime']


def _dt_int(dt, tm='00:00:00'):
    """converts date (& time) to integer"""
    return int(datetime.datetime.strptime('{} {}'.format(dt,tm), '%Y-%m-%d %H:%M:%S').strftime("%s"))

test_df.ts = test_df.ts.apply(_dt_int)


class TestMoney(unittest.TestCase):
    def setUp(self):
        pass

    def testFilterByTimeNoFilter(self):
        """If month=0, do not remove any rows
        passing first n rows of the test_df,
        expected not to remove any rows
        """
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.filter_by_time(last_months=0)
        assert_frame_equal(test_df.iloc[:,:], _dataset.data)

    def testFilterByTimeFilter(self):
        """If month>0, remove rows
        passing first n rows of the test_df,
        expected to remove some rows
        """
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.filter_by_time(last_months=1)
        assert_frame_equal(test_df.iloc[[1,2,3,5,6,7,8],:], _dataset.data)

    def testItemConversionToSequence(self):
        """convert items to a list in time-based sequence
        passing first n rows of the test_df,
        expected as per dictionary frame defined below
        """
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _expecteddf = pd.DataFrame.from_dict({
            'session_id': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
            'sequence': {0: ['1'], 1: ['1', '3'], 2: ['3'], 3: ['1'], 4: ['2', '1', '3', '3']},
            'ts': {0: 1421107200, 1: 1423785600, 2: 1423872000, 3: 1418428800, 4: 1423526400},
            'user_id': {0: 1, 1: 1, 2: 1, 3: 2, 4: 2}})
        assert_frame_equal(_expecteddf, _dataset.data)        

    def testItemConversionToSequenceTopK(self):
        """convert items to a list in time-based sequence
        filters topk most interacted items
        passing first n rows of the test_df with topk=2,
        expected as per dictionary frame defined below
        """
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence(topk=2)
        _expecteddf = pd.DataFrame.from_dict({
            'session_id': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
            'sequence': {0: ['1'], 1: ['1', '3'], 2: ['3'], 3: ['1'], 4: ['1', '3', '3']},
            'ts': {0: 1421107200, 1: 1423785600, 2: 1423872000, 3: 1418428800, 4: 1423526400},
            'user_id': {0: 1, 1: 1, 2: 1, 3: 2, 4: 2}})
        assert_frame_equal(_expecteddf, _dataset.data)   

    def testDataStatistics(self):
        """generate statistics of the dataset
        passing first n rows of the test_df,
        expected as per string defined below
        expected:
        Number of items: 3\nNumber of users: 2\nNumber of sessions: 
        5\nSession length:\n\tAverage: 1.80\n\tMedian: 1.0\n\tMin: 
        1\n\tMax: 4\nSessions per user:\n\tAverage: 2.50\n\tMedian: 
        2.5\n\tMin: 2\n\tMax: 3\nMost popular items: 
        [('1', 4), ('3', 4), ('2', 1)]"""
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _generated = _dataset.get_stats()        
        self.assertIn("Number of items: 3", _generated)    
        self.assertIn("Most popular items: [('1', 4), ('3', 4), ('2', 1)]", _generated)    
        self.assertIn("Session length:\n\tAverage: 1.80\n\tMedian: 1.0", _generated)    
        self.assertNotIn("Number of items: 4", _generated)     

    def testRandomSplit(self):
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _dataset.random_holdout(0.6)
        _expecteddf = pd.DataFrame.from_dict(
            {'session_id': {1: 2, 2: 3, 4: 5},
            'sequence': {1: ['1', '3'], 2: ['3'], 4: ['2', '1', '3', '3']},
            'ts': {1: 1423785600, 2: 1423872000, 4: 1423526400},
            'user_id': {1: 1, 2: 1, 4: 2}}
            )
        _expecteddf = _expecteddf.reindex([1,4,2])
        assert_frame_equal(_expecteddf, _dataset.train)

    def testTemporalSplitThreshold1(self):
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _dataset.temporal_holdout(1423600000)
        _expecteddf = pd.DataFrame.from_dict(
            {'session_id': {0: 1, 3: 4, 4: 5},
            'sequence': {0: ['1'], 3: ['1'], 4: ['2', '1', '3', '3']},
            'ts': {0: 1421107200, 3: 1418428800, 4: 1423526400},
            'user_id': {0: 1, 3: 2, 4: 2}}
            )
        assert_frame_equal(_expecteddf, _dataset.train) 

    def testTemporalSplitThreshold2(self):
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _dataset.temporal_holdout(1423500000)
        _expecteddf = pd.DataFrame.from_dict(
            {'session_id': {0: 1, 3: 4},
            'sequence': {0: ['1'], 3: ['1']},
            'ts': {0: 1421107200, 3: 1418428800},
            'user_id': {0: 1, 3: 2}}
        )
        assert_frame_equal(_expecteddf, _dataset.train) 

    def testSessionOutSplit(self):
        _dataset = SessionDataset(test_df.iloc[:9,:])
        _dataset.convert_to_sequence()
        _dataset.last_session_out_split()
        _expecteddf = pd.DataFrame.from_dict(
            {'session_id': {0: 1, 1: 2, 3: 4},
            'sequence': {0: ['1'], 1: ['1', '3'], 3: ['1']},
            'ts': {0: 1421107200, 1: 1423785600, 3: 1418428800},
            'user_id': {0: 1, 1: 1, 3: 2}}
        )
        assert_frame_equal(_expecteddf, _dataset.train)


if __name__ == '__main__':
    unittest.main()