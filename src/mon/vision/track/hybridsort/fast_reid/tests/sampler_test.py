import sys
import unittest

sys.path.append('.')
from fast_reid.fastreid.data.samplers import TrainingSampler


class SamplerTestCase(unittest.TestCase):
    def test_training_sampler(self):
        sampler = TrainingSampler(5)
        for i in sampler:
            from ipdb import set_trace; set_trace()
            print(i)


if __name__ == '__main__':
    unittest.main()
