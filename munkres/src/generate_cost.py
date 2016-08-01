'''
Created on May 6, 2013

@author: jolly
'''
import numpy as np
from scipy.spatial.distance import cdist
import fcm.statistics as stats
from munkres import _get_cost
from kldiv import true_kldiv as kldiv

def mean_distance(ref, test, use_means=None):

    if isinstance(ref, stats.ModalDPMixture) and isinstance(test, stats.ModalDPMixture) and not use_means:
        x = ref.modes()
        y = test.modes()
    else:
        x = ref.mus
        y = test.mus

    return cdist(x, y)

def classification_distance(ref, test, test_data=None, ndraw=100000):
    if test_data is None:
        test_data = test.draw(ndraw)

    t_x = test.classify(test_data)
    r_x = ref.classify(test_data)

    cost = test_data.shape[0] * np.ones((len(test.clusters), len(ref.clusters)), dtype=np.double)

    _get_cost(t_x, r_x, cost)

    return (cost / test_data.shape[0]).T.copy()

def kldiv_distance(ref, test, use_means=None, ndraws=100000):
    if isinstance(ref, stats.ModalDPMixture) and isinstance(test, stats.ModalDPMixture) and not use_means:
        ref_list = []
        ref_sample = ref.draw(ndraws)
        ref_x = ref.classify(ref_sample)
        pi = 0.0
        
        for i in sorted(ref.cmap):
            sub_x = ref_sample[ref_x == i]
            ref_list.append(stats.DPCluster(pi, sub_x.mean(0), np.cov(sub_x.T)))

        test_list = []
        test_sample = test.draw(ndraws)
        test_x = test.classify(test_sample)

        for i in sorted(test.cmap):
            sub_x = test_sample[test_x == i]
            test_list.append(stats.DPCluster(pi, sub_x.mean(0), np.cov(sub_x.T)))

        ref = stats.DPMixture(ref_list)
        test = stats.DPMixture(test_list)

    cost = np.zeros((len(ref.clusters), len(test.clusters)))

    for i in range(len(ref.clusters)):
        for j in range(len(test.clusters)):
            cost[i, j] = kldiv(ref[i].mu, test[j].mu, ref[i].sigma, test[j].sigma)

    return cost

if __name__ == '__main__':
    cluster1 = stats.DPCluster(.5, np.array([0, 0]), np.eye(2))
    cluster2 = stats.DPCluster(.5, np.array([0, 4]), np.eye(2))
    cluster3 = stats.DPCluster(.25, np.array([0, 0]), np.eye(2))
    cluster4 = stats.DPCluster(.25, np.array([4, 0]), np.eye(2))
    cluster5 = stats.DPCluster(.5, np.array([0, 4]), np.eye(2))
    A = stats.DPMixture([cluster1, cluster2])
    B = stats.DPMixture([cluster3, cluster4, cluster5])
    from munkres import munkres
    print 'Ref has means', A.mus, 'with weights', A.pis
    print 'Test has means', B.mus, 'with weights', B.pis
    print 'mean distance'
    print mean_distance(A, B)
    print munkres(mean_distance(A, B))
    mA = A.make_modal()
    mB = B.make_modal()
    print 'modal distance'
    print mean_distance(mA, mB)
    print munkres(mean_distance(mA,mB))
    print 'modal using means'
    print mean_distance(mA, mB, use_means=True)
    print munkres(mean_distance(mA,mB, use_means=True))

    print 'classification'
    print classification_distance(A, B)
    print munkres(classification_distance(A,B))
    print 'modal classification'
    print classification_distance(mA, mB)
    print munkres(classification_distance(mA,mB))

    print 'kldiv'
    print kldiv_distance(A, B)
    print munkres(kldiv_distance(A,B))

    print 'modal kldiv'
    print kldiv_distance(mA, mB)
    print munkres(kldiv_distance(mA,mB))
