#library
import numpy as np
from numpy import genfromtxt
from sklearn.manifold import TSNE

#data
X = genfromtxt('sc_train.csv', delimiter=',')

#tsne
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
X = model.fit_transform(X)

np.savetxt("TSNE_output.csv", X, delimiter=",")

#library example parameter
'''class sklearn.manifold.TSNE(n_components=2, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0, n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5)'''

