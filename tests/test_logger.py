import unittest
import tempfile
import os

from src.logger import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger_name = tempfile.NamedTemporaryFile(suffix='.log').name
        self.logger = Logger(self.logger_name)
        self.logger.info('Unittest Message 1')
        self.logger.warning('Unittest Message 2')
        with open(self.logger_name, 'r') as l:
            self.msg = l.readlines()

    def testLogFileCreated(self):
        self.assertTrue(os.path.exists(self.logger_name))

    def testLogsWritten(self):
        self.assertIn('Unittest Message 1',self.msg[0])
        self.assertIn('Unittest Message 2',self.msg[1])
    
    def testLogTypes(self):
        self.assertIn('INFO:',self.msg[0])
        self.assertIn('WARNING:',self.msg[1])