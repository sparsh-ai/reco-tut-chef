import unittest

from src.config import Configurator


class TestConfigurator(unittest.TestCase):
    def setUp(self):
        self.conf_default = Configurator("tests/data/config.properties", default_section="default")
        self.conf_mlp = Configurator("tests/data/config.properties", default_section="mlp")

    def testRecommenderName(self):
        self.assertEqual(self.conf_default["recommender"], "config_MF")
    
    def testRecommenderNameType(self):
        self.assertEqual(str(type(self.conf_default["recommender"])), "<type 'unicode'>")

    def testIntegerType(self):
        self.assertEqual(str(type(self.conf_default["epochs"])), "<type 'int'>")

    def testListType(self):
        self.assertEqual(str(type(self.conf_mlp["layers"])), "<type 'list'>")
    
    def testUnwantedTypeConversion(self):
        self.assertEqual(str(type(self.conf_default["batch_size"])), "<type 'str'>")

    def testDefaultSectionValue(self):
        self.assertEqual(self.conf_default["epochs"], 300)

    def testCustomSectionValue(self):
        self.assertEqual(self.conf_mlp["epochs"], 100)
    
    def testListValue(self):
        self.assertEqual(self.conf_mlp["layers"], [32, 64, 1])