import tensorflow as tf

""" This is copied from 
https://github.com/ragavvenkatesan/cse591-practicals/blob/master/miniprojects/miniproject2/dataset.py
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from tools.feed import feed

from globals import TRAIN_SET_SIZE, TEST_SET_SIZE

def to_one_hot(indices, depth):
    """
    Convert ints to one-hot encoding.

    Args:
        indices: Which index to put code.
        depth: How many dimensions of code tou want?

    Returns:
        np.ndarray: Output of length of indices, depth.
    """
    targets = np.array(indices).reshape(-1)
    return np.eye(depth)[targets]

class xor_feeder(object):
    """
    This is the xor feeder. 

    Attributes:
        test.images: Should contain test images.
        test.labels: Should contain test labels.
        train.next_batch(mini_batch): Should return a mini_batch given
                some mini_batch values

    Args:
        tuple: number of samples for train and test
                         (Default ``(10000,1000)``).
    """
    def __init__(self, dataset_size = (TRAIN_SET_SIZE, TEST_SET_SIZE)):
        """
        Class constructor
        """
        dataset = xor_data_generator()
        images, labels = dataset.query_data(samples = dataset_size[0])        
        dataset.plot(images, np.argmax(labels, 1) )        
        self.train = feed(images, labels)

        images, labels = dataset.query_data(samples = dataset_size[1])
        self.test = feed(images, labels)

class xor_data_generator(object):
    """ 
    Class that creates a xor dataset. Note that for the grading of the project, this method 
    might be changed, although it's output format will not be. This implies we might use other
    methods to create data. You must assume that the dataset will be blind and your machine is 
    capable of running any dataset. Although the dataset will not be changed drastically and will
    hold the XOR style . 
    """    
    def __init__(self):

        self.dimensions = 2
        self.positive_means = [[-1,-1],[1,1]]
        self.negative_means = [[-1,1],[1,-1]]
        self.covariance = [[0.1, 0.0], [0.0, 0.1]]

    def query_data(self, **kwargs):
        """
        Once initialized, this method will create more data.
        Args:
            samples: number of samples of data needed (optional, default randomly 10k - 50k)   
        Returns:
            tuple: data a tuple, ``(x,y)``
                ``x`` is a two dimensional ndarray ordered such that axis 0 is independent 
                data and data is spread along axis 1. 
                ``y`` is a 1D ndarray it will be of the same length as axis 0 of x.                          
        """
        if 'samples' in kwargs.keys():
            samples = kwargs['samples']
        else:
            samples = np.random.randint(low = 1000, high = 5000)

        # make positive samples
        dim1, dim2 = np.random.multivariate_normal( self.positive_means[0], 
                                                    self.covariance, samples/4).T
        positive = np.stack((dim1,dim2),axis = 1)
        dim1, dim2 = np.random.multivariate_normal( self.positive_means[1], 
                                                    self.covariance, samples/4).T            
        positive = np.concatenate((positive,np.stack((dim1,dim2),axis = 1)),axis = 0)
        labels = np.ones(positive.shape[0], dtype = 'int32')

        # make the negative samples
        dim1, dim2 = np.random.multivariate_normal( self.negative_means[0], 
                                                    self.covariance, samples/4).T
        negative = np.stack((dim1,dim2),axis = 1)
        dim1, dim2 = np.random.multivariate_normal( self.negative_means[1], 
                                                    self.covariance, samples/4).T            
        negative = np.concatenate((negative,np.stack((dim1,dim2),axis = 1)),axis = 0)    
        labels = np.concatenate((labels,np.zeros(negative.shape[0], dtype = 'int32')), axis = 0)

        data = np.concatenate((positive, negative),axis = 0)        
        assert data.shape[0] == labels.shape[0]
  
        perm = np.random.permutation(labels.shape[0])
        data = data[perm,:]
        labels = labels[perm]

        return (data, to_one_hot(labels, 2) )

    def plot(self, data,labels):
        """
        This method will plot the data as created by this dataset generator.
        Args:
            data: as produced by the ``query_data`` method's first element.
            labels: as produced by the ``query_data`` method's second element.
        """
        positive = data[labels == 1,:]
        negative = data[labels == 0,:]

        plt.plot(positive[:,0], positive[:,1], 'bo', negative[:,0], negative[:,1], 'rs')        
        plt.axis('equal')      
        plt.title('XOR Dataset')  
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()

    def _demo (self):
        """
        This is a demonstration method that will plot a version of the dataset on the screen.
        """        
        data, labels = self.query_data(samples = 5000) 
        self.plot(data, labels)        

class xor(object):
    """
    Class for the xor objects

    Class Properties:
    
        These are variables of the class that are available outside. 
        
        *   ``images``: This is the placeholder for images. This needs to be fed in.
        *   ``labels``: This is the placeholder for images. This needs to be fed in.     
        *   ``feed``: This is a feeder from mnist tutorials of tensorflow.      
    """
    def __init__(self):
        """
        Class constructor
        """     
        self.feed = xor_feeder()

        #Placeholders
        with tf.variable_scope('dataset_inputs') as scope:
            self.images = tf.placeholder(tf.float32, shape=[None, 2], name = 'images')
            self.labels = tf.placeholder(tf.float32, shape = [None, 2], name = 'labels')         

if __name__ == '__main__':
    pass              