from unittest import TestCase

class DummyTest(TestCase):
    def testDummy(self):
        self.assertEqual("dummy".upper(), "DUMMY")