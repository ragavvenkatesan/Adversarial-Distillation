import tensorflow as tf
import numpy as np
import sys 
sys.path.append('../')

from tools.layers import *
from tools.support import visualize_images, log
from tools.optimizer import process_params, apply_gradient_descent
from globals import *

class expert(object):
    """
    Definition of the expert class of networks.

    Notes:
        *   Produces the lenet model. A typical lenet has 
            two convolutional layers with filters sizes ``5X5`` and ``3X3``. These
            are followed by two fully-connected layers and a softmax layer. This 
            network model, reproduces this network to be trained on MNIST images
            of size ``28X28``.     
        *   Most of the important parameters are stored in :mod:`global_definitions` 
            in the file ``global_definitions.py``.

    Args:
        images: Placeholder for images
        name: Name of the network.

    Class Properties:
        These are variables of the class that are available outside. 
        
        *   ``images``: This is the placeholder for images. This needs to be fed in.
        *   ``dropout_prob``: This is also a placeholder for dropout probability. This needs to be fed in.    
        *   ``logits``: Output node of the softmax layer, before softmax
        *   ``inference``: Output node of the softmax layer.
        *   ``predictions``: argmax of the softmax layer. 
        *   ``back_prop``: Backprop is an optimizer. 
        *   ``obj``: Is a cumulative objective tensor.
        *   ``cost``: Cost of the back prop error alone.
        *   ``labels``: Placeholder for labels, needs to be fed in.
        *   ``accuracy``: Tensor for accuracy. 

    """
    def __init__ (  self,
                    images,
                    name = 'expert' ):
        """
        Class constructor. Creates the model and allthe connections. 
        """
        with tf.variable_scope(name) as scope:
            self.images = images
            self.name = name

            self.dropout_prob = tf.placeholder_with_default(
                                                input = tf.constant(1.0, dtype = tf.float32),
                                                shape = None,
                                                name = 'dropout_probability')  

            fc1_out, params = dot_product_layer  (  input = self.images,
                                                    neurons = EXPERT,
                                                    name = 'dot')
            process_params(params, name = self.name)

            # Logits layer
            self.logits, params = dot_product_layer  (  input = fc1_out,
                                                        neurons = C,
                                                        activation = 'identity',
                                                        name = 'logits_layer')
            process_params(params, name = self.name)

            # Softmax layer
            self.inference, self.predictions = softmax_layer (  input = self.logits,
                                                                name = 'softmax_layer' )                                                    

            # Temperature Softmax layer
            self.temperature_softmax, _ = softmax_layer ( input = self.logits, 
                                                          temperature = TEMPERATURE,
                                                          name = 'temperature_softmax_layer' )

    def cook(self, labels, name ='train'):
        """
        Prepares the network for training

        Args:
            labels: placeholder for labels
            name: Training block name scope 
        """    
        with tf.variable_scope( self.name + '_objective') as scope:
            self.labels = labels
            with tf.variable_scope( self.name + '_cross-entropy') as scope:
                self.cost = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits ( 
                                                     labels = self.labels,
                                                     logits = self.logits)
                                                )
                tf.add_to_collection( self.name + '_objectives', self.cost ) 
                tf.summary.scalar('cost', self.cost)  

            self.obj = tf.add_n(tf.get_collection( self.name + '_objectives'), name='objective')
    
        with tf.variable_scope(self.name + '_train') as scope:                                                                  
            self.back_prop = apply_gradient_descent(var_list = tf.get_collection(self.name + '_trainable_params'),
                                        obj = self.obj )

        with tf.variable_scope(self.name + '_test') as scope:                                                
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1), \
                                                        name = 'correct_predictions')
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32) , name ='accuracy')                                     
            tf.summary.scalar('accuracy', self.accuracy) 
     
            with tf.variable_scope(self.name + "_confusion"):
                confusion = tf.confusion_matrix(tf.argmax(self.labels,1), self.predictions,
                                                num_classes= C,
                                                name='confusion')
                confusion_image = tf.reshape( tf.cast( confusion, tf.float32),[1, C, C, 1])
                tf.summary.image('confusion',confusion_image)   

class novice(object):
    """
    Definition of the Novice class of networks.

    Notes:
        *   Produces a MLP model.
        *   Similar to expert.

    Args:
        images: Placeholder for images
        name: Name of the network.

    Class Properties:
        These are variables of the class that are available outside. 
        
        *   ``images``: This is the placeholder for images. This needs to be fed in.
        *   ``dropout_prob``: This is also a placeholder for dropout probability. This needs to be fed in.    
        *   ``logits``: Output node of the softmax layer, before softmax
        *   ``inference``: Output node of the softmax layer.
        *   ``predictions``: argmax of the softmax layer. 
        *   ``back_prop``: Backprop is an optimizer. 
        *   ``obj``: Is a cumulative objective tensor.
        *   ``cost``: Cost of the back prop error alone.
        *   ``labels``: Placeholder for labels, needs to be fed in.
        *   ``accuracy``: Tensor for accuracy. 

    """
    def __init__ (  self,
                    images,
                    name = 'novice' ):
        """
        Class constructor. Creates the model and allthe connections. 
        """
        with tf.variable_scope(name) as scope:
            self.images = images
            self.name = name

            self.dropout_prob = tf.placeholder_with_default(
                                                input = tf.constant(1.0, dtype = tf.float32),
                                                shape = None,
                                                name = 'dropout_probability')   
                                    

            # Dot Product Layer 1
            fc1_out, params = dot_product_layer  (  input = self.images,
                                                    neurons = NOVICE,
                                                    name = 'dot')
            process_params(params, name = self.name)
            params = params
            self.params = params

            # Logits layer
            self.logits, params = dot_product_layer  (  input = fc1_out,
                                                        neurons = C,
                                                        activation = 'identity',
                                                        name = 'logits_layer')
            process_params(params, name = self.name)

            # Softmax layer
            self.inference, self.predictions = softmax_layer (  input = self.logits,
                                                                name = 'softmax_layer' )                                                    

            self.temperature_softmax, _ = softmax_layer ( input = self.logits, 
                                                          temperature = TEMPERATURE,
                                                          name = 'temperature_softmax_layer' )

    def cook(self, judgement = None, labels = None, test_labels = None, name ='train'):
        """
        Prepares the network for training

        Args:
            labels: placeholder for labels, if ``None`` Doesn't do labels training.
            judgement: node that the judge outputs. if ``None`` simply trains labels.
            test_labels: Labels used for testing. If labels is provided, this input is 
                         ignored.
            name: Training block name scope 
        """    
        with tf.variable_scope( self.name + '_objective') as scope:
            if labels is not None:
                self.labels = labels                
                with tf.variable_scope( self.name + '_cross-entropy') as scope:
                    ce_loss = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits ( 
                                                        labels = self.labels,
                                                        logits = self.logits)
                                                    )                                                
                tf.add_to_collection( self.name + '_objectives', ce_loss )                                                    
                tf.summary.scalar('cross_entropy_cost', ce_loss)  
            
            if judgement is not None:
                self.judgement = judgement
                with tf.variable_scope( self.name + '_fooler') as scope:
                    # Use the LS GAN technique. 
                    # j_loss = tf.reduce_mean((1 - judgement)**2)
                    j_loss = -0.5 * tf.reduce_mean(log(judgement))
                tf.add_to_collection( self.name + '_objectives', j_loss )                                                    
                tf.summary.scalar('judge_cost', j_loss)  
            
            self.obj = tf.add_n(tf.get_collection( self.name + '_objectives'), name='objective')
            tf.summary.scalar('total_objective', self.obj)

        with tf.variable_scope(self.name + '_train') as scope:
            self.back_prop = apply_gradient_descent(var_list = tf.get_collection(self.name + '_trainable_params'),
                                    obj = self.obj )

        is_test = False
        if not labels is None:
            is_test = True
            self.labels = labels
        
        elif not test_labels is None:
            is_test = True
            self.labels = test_labels            
 
        else:
            raise Error("Provide either one label nodes.")

        with tf.variable_scope(self.name + '_test') as scope:                                                
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1), \
                                                        name = 'correct_predictions')
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32) , name ='accuracy')                                     
            tf.summary.scalar('accuracy', self.accuracy) 
     
            with tf.variable_scope(self.name + "_confusion"):
                confusion = tf.confusion_matrix(tf.argmax(self.labels,1), self.predictions,
                                                num_classes= C,
                                                name='confusion')
                confusion_image = tf.reshape( tf.cast( confusion, tf.float32),[1, C, C, 1])
                tf.summary.image('confusion',confusion_image) 

class judge(object):
    """
    Definition of the Judge class of networks.

    Notes:
        *   Produces a the judge.
        *   This is the decoder of the GAN network.
        *   Extracts a few layers of image features.
        *   Embeds the layer feed from expert and novice into a channel.
        *   Detemines if the embedding comes from expert or novice.

    Args:
        images: Placeholder for images
        expert: Tensor that is expert outputs - ``expert.inference``
        novice: Tensor that is novice outputs - ``novice.inference``
        labels: Supply this for feed_dicts. This is useless.
        input_params: If supplied will use them for the judge image layers. 
                otherwise default to ``None``.
        name: Name of the network.

    Class Properties:
        These are variables of the class that are available outside. 
        
        *   ``images``: This is the placeholder for images. This needs to be fed in.
        *   ``dropout_prob``: This is also a placeholder for dropout probability. This needs to be fed in.    
        *   ``back_prop``: Backprop is an optimizer. 
        *   ``obj``: Is a cumulative objective tensor.
        *   ``judgement``: Is a node that determines judgement. 
    """
    def __init__ (  self,
                    images,
                    expert,
                    novice,
                    labels,
                    name = 'judge' ):
        """
        Class constructor. Creates the model and all the connections. 
        """
        self.labels = labels
        with tf.variable_scope(name) as scope:
            self.images = images
            self.name = name

            self.dropout_prob = tf.placeholder_with_default(
                                                input = tf.constant(1.0, dtype = tf.float32),
                                                shape = None,
                                                name = 'dropout_probability')                                         
            merged_expert = tf.concat([expert, self.images], 1)
            merged_novice = tf.concat([novice, self.images], 1)   

            # Merged Expert Dot Product Layer 1
            merged_expert_fc1_out, params = dot_product_layer  (  
                                            input = merged_expert, 
                                            neurons = MERGED,
                                            name = 'merged_expert_1')
            # Merged Novice Dot Product Layer 1
            merged_novice_fc1_out, params = dot_product_layer  (  
                                            input = merged_novice, 
                                            params = params,
                                            neurons = MERGED,
                                            name = 'merged_novice_1')
            process_params(params, name = self.name)

            self.judgement_expert, params = dot_product_layer  ( 
                                                input = merged_expert_fc1_out,
                                                neurons = 1,
                                                activation = 'sigmoid',
                                                name = 'expert_probability')
            self.judgement_novice, params = dot_product_layer  ( 
                                                input = merged_novice_fc1_out,
                                                neurons = 1,
                                                params = params,
                                                activation = 'sigmoid',
                                                name = 'novice_probability')                                                
            process_params(params, name = self.name)                                                    

    def cook(self, name ='train'):
        """
        Prepares the network for training

        Args:
            name: Training block name scope 
        """    
        with tf.variable_scope( self.name + '_objective') as scope:
            # self.obj = tf.reduce_mean((1 - self.judgement_expert)**2) +\
            #                    tf.reduce_mean(self.judgement_novice**2)            
            self.obj = -0.5 * tf.reduce_mean(log(self.judgement_expert)) - \
                          0.5 * tf.reduce_mean(log(1 - self.judgement_novice))
            tf.add_to_collection( self.name + '_objectives', self.obj )                                                    

            self.obj = tf.add_n(tf.get_collection( self.name + '_objectives'), name='objective')
            tf.summary.scalar('total_objective', self.obj)

        with tf.variable_scope(self.name + '_train') as scope:
            self.back_prop = apply_gradient_descent(var_list = tf.get_collection(self.name + '_trainable_params'),
                                        obj = self.obj )

if __name__ == '__main__':
    pass                    