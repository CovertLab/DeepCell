import numpy as np
from munkres import munkres

def test_big(k):
    a = np.empty((k,k))
    for i in range(k):
        for j in range(k):
            a[i,j] = (i+1)*(j+1)
    b = munkres(a)
    print k, b


if __name__ == '__main__':
	for i in range(256):
		test_big(i+1)
