import numpy as np
from os.path import join
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.io as sio


def main():
	trainData = pd.read_csv('data/ml-100k/u1.base', sep='\t', names=['user', 'movie', 'rating', 'time'], usecols=[0,1,2,3]).as_matrix()
	testData = pd.read_csv('data/ml-100k/u1.test', sep='\t', names=['user', 'movie', 'rating', 'time'], usecols=[0,1,2,3]).as_matrix()

	maxT = max(np.max(trainData[:,3]), np.max(testData[:,3]))
	minT = min(np.min(trainData[:,3]), np.min(testData[:,3]))
	nUsers = max(np.max(trainData[:,0]), np.max(testData[:,0]))
	nItems = max(np.max(trainData[:,1]), np.max(testData[:,1]))

	N = 10
	(trainLabeled, trainBinned) = toBins(trainData, minT, maxT, N, nUsers, nItems)
	(testLabeled, testBinned) = toBins(testData, minT, maxT, N, nUsers, nItems)

	sio.savemat('trainTimed', {'trainLabeled' : trainLabeled, 'trainBinned' : trainBinned})
	sio.savemat('testTimed', {'testLabeled' : testLabeled, 'testBinned' : testBinned})

def toBins(xs, minT, maxT, n, nUsers, nItems):
    edges = np.linspace(minT-1, maxT, num=n+1)
    binned = np.zeros((nUsers, nItems, n), dtype=np.int32)
    labeled = np.copy(xs)
    for i in range(1, len(edges)):
        inds = xs[:,3] <= edges[i]
        # cvt to matrix
        users = xs[inds, 0] - 1
        movies = xs[inds, 1] - 1
        ratings = xs[inds, 2]
        binned[:,:,i-1] = csr_matrix((ratings, (users, movies)), shape=(nUsers, nItems)).toarray()
        
        inds = np.logical_and(inds, xs[:,3] > edges[i-1])
        labeled[inds,3] = i
        
    return (labeled, binned)

if __name__ == '__main__':
	main()