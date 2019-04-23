import sys
sys.path.append("/data/yosef2/users/chenling/harmonization/MAGAN/MAGAN/")

import tensorflow as tf
from model import MAGAN

def correspondence_loss(b1, b2):
    """
    The correspondence loss.
    :param b1: a tensor representing the object in the graph of the current minibatch from domain one
    :param b2: a tensor representing the object in the graph of the current minibatch from domain two
    :returns a scalar tensor of the correspondence loss
    """
    domain1cols = [0]
    domain2cols = [0]
    loss = tf.constant(0.)
    for c1, c2 in zip(domain1cols, domain2cols):
        loss += tf.reduce_mean((b1[:, c1] - b2[:, c2])**2)

    return loss

magan = MAGAN(dim_b1=2, dim_b2=2, correspondence_loss=correspondence_loss)


class Magan():
    def __init__(self, batch_size=100,n_epochs=100000, loss=correspondence_loss):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.loss = loss

    def fit_transform(self, X, batch, list_b):
        index_0 = np.where(batch == list_b[0])[0]
        index_1 = np.where(batch == list_b[1])[0]
        X1 = np.log(1 + X[index_0])
        X2 = np.log(1 + X[index_1])

        loadb1 = Loader(X1, shuffle=True)
        loadb2 = Loader(X2, shuffle=True)
        # Build the tf graph
        magan = MAGAN(dim_b1=X1.shape[1], dim_b2=X2.shape[1], correspondence_loss=self.loss)
        # Train
        for i in range(1, self.n_epochs):
            if i % 100 == 0: print("Iter {} ({})".format(i, now()))
            xb1_ = loadb1.next_batch(self.batch_size)
            xb2_ = loadb2.next_batch(self.batch_size)
            magan.train(xb1_, xb2_)

        xb1_ = loadb1.next_batch(len(index_0))
        xb2_ = loadb2.next_batch(len(index_1))
        lstring = magan.get_loss(xb1_, xb2_)
        print("{} {}".format(magan.get_loss_names(), lstring))
        xb1 = magan.get_layer(xb1_, xb2_, 'xb1')
        xb2 = magan.get_layer(xb1_, xb2_, 'xb2')
        Gb1 = magan.get_layer(xb1_, xb2_, 'Gb1')
        Gb2 = magan.get_layer(xb1_, xb2_, 'Gb2')
        arr1 = np.zeros_like(X, dtype=np.float)
        arr1[index_0] = xb1
        arr1[index_1] = Gb2
        arr2 = np.zeros_like(X, dtype=np.float)
        arr2[index_0] = Gb1
        arr2[index_1] = xb2
        return arr1, arr2
