import numpy as np
from munkres import munkres, max_cost_munkres
from munkres import iterative_munkres

def test_simple():
    a = np.array([i for i in range(64)], dtype=np.double).reshape((8, 8))
    b = munkres(a)
    truth = np.zeros((8, 8), dtype=np.bool)
    for i in range(8):
        truth[7 - i, i] = True
    np.testing.assert_array_equal(b, truth, 'simple 8x8 case failed, b=%s, truth=%s' % (str(b), str(truth)))

def test_basic():
    a = np.array(map(float, '7 4 3 6 8 5 9 4 4'.split()), dtype=np.double).reshape((3, 3))
    b = munkres(a)
    truth = np.array([[False, False, True],
                      [ True, False, False],
                      [False, True, False]])
    np.testing.assert_array_equal(b, truth, 
                                  'basic 3x3 case failed\na=%s\ntruth=%s\ncost=%s\ntrue_cost=%s' % (str(a), str(truth), str(a[b]), str(a[truth])))

def test_more_candidates():

    a = np.array(map(float, '7 4 3 6 8 5 9 4 4 10 10 10'.split()), dtype=np.double).reshape((4, 3))
    b = munkres(a)
    truth = np.array([[False, False, True],
                      [True, False, False],
                      [False, True, False],
                      [False, False, False]])
    
    np.testing.assert_array_equal(b, truth, 
                                  'basic 4x3 case failed\na=%s\ntruth=%s\ncost=%s\ntrue_cost=%s' % (str(a), str(truth), str(a[b]), str(a[truth])))
    

def test_more_jobs():
    a = np.array(map(float, '7 4 3 3 6 8 5 3 9 4 4 3'.split()), dtype=np.double).reshape((3, 4))
    b = munkres(a)

    truth = np.array([[False, False, True, False],
                      [False, False, False, True],
                      [False, True, False, False]])
    np.testing.assert_array_equal(b, truth, 
                                  'basic 3x4 case failed\na=%s\ntruth=%s\ncost=%s\ntrue_cost=%s' % (str(a), str(truth), str(a[b]), str(a[truth])))
                  

def test_big():
    a = np.empty((100,100))
    for i in range(100):
        for j in range(100):
            a[i,j] = (i+1)*(j+1)
    b = munkres(a)

    truth = np.zeros((100, 100), dtype=np.bool)
    for i in range(100):
        truth[99 - i, i] = True
    np.testing.assert_array_equal(b, truth, 'simple 100x100 case failed, b=%s, truth=%s' % (str(b), str(truth)))

def test_max_cost():
    
    a = np.array([i for i in range(16)], dtype=np.double).reshape((4,4))
    b = max_cost_munkres(a,9)
    print b
    truth = np.zeros((4, 4), dtype=np.bool)
    truth[0,1] = True
    truth[1,0] = True
    
    np.testing.assert_array_equal(b, truth, 'basic 3x4 case failed\na=%s\ntruth=%s\ncost=%s\ntrue_cost=%s' % (str(a), str(truth), str(a[b]), str(a[truth])))


def test_iterative():
    cost = np.array([[1,1,1,1,1,9],[9,9,9,9,9,9]], dtype=np.double)
    a = iterative_munkres(cost, 2)
    truth = np.array([[True, True, True, True, True, False],
                      [False,False,False,False,False,False]])
    
    np.testing.assert_array_equal(a,truth, 'iterative assignment failed')
#    cost = cost.T
#    b = iterative_munkres(cost, 2)
#    np.testing.assert_array_equal(b,truth.T, 'iterative assignment failed')
#    np.testing.assert_array_equal(a,b.T, 'iterative assignment failed')
if __name__ == '__main__':
    import nose
    nose.run()
