""" Tests """

# matplotlib will open windows during testing unless you do the following
import matplotlib
matplotlib.use("AGG") 

import data
import models

class TestClass:
    def setUp(self):
        pass

    def test_simple_data(self):
        vars = data.simple_hierarchical_model([10,10,10])
        assert 'y' in vars
