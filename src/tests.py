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
        vars = data.simple_hierarchical_data([10,10,10])
        assert 'y' in vars
        
    def test_simple_model(self):
        p = [[1,2,3], [4,5,6,7]]
        vars = models.simple_hierarchical_model(p)
        assert 'y' in vars
