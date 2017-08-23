import tensorflow as tf
import numpy as np

from globals import *

import sys 
sys.path.append('../')

from tools.layers import *
from tools.support import visualize_images

def apply_gradient_descent(var_list, obj):
    """
    Sets up the gradient descent optimizer

    Args:
        var_list: List of variables to optimizer over.        
        obj: Node of the objective to minimize
    Notes:
        learning_rate: What learning rate to run with. (Default = ``0.01``) Set with ``LR``
    """
    back_prop = tf.train.GradientDescentOptimizer(
                                            learning_rate = LR,
                                            name = 'gradient_descent' ).minimize(loss = obj, \
                                                    var_list = var_list ) 
    return back_prop

def apply_adam (var_list, obj, learning_rate = 1e-4):
    """
    Sets up the ADAM optmiizer

    Args:
        var_list: List of variables to optimizer over.
        obj: Node of the objective to minimize        
    
    Notes:
        learning_rate: What learning rate to run with. (Default = ``0.01``) Set with ``LR``
    """      
    back_prop = tf.train.AdamOptimizer(
                                        learning_rate = LR,
                                        name = 'adam' ).minimize(loss = obj, \
                                            var_list = var_list) 
    return back_prop                                                               

def apply_rmsprop( var_list, obj ):
    """
    Sets up the RMS Prop optimizer

    Args:
        var_list: List of variables to optimizer over.
        obj: Node of the objective to minimize        

    Notes:
        * learning_rate: What learning rate to run with. (Default = ``0.001``). Set  ``LR``
        * momentum: What is the weight for momentum to run with. (Default = ``0.7``). Set ``MOMENTUM``
        * decay: What rate should learning rate decay. (Default = ``0.95``). Set ``DECAY``            
    """    
    back_prop = tf.train.RMSPropOptimizer(
                                        learning_rate = LR,
                                        decay = DECAY,
                                        momentum = MOMENTUM,
                                        name = 'rmsprop' ).minimize(loss = obj, \
                                        var_list = var_list) 
    return back_prop

def apply_weight_decay (var_list, name = 'weight_decay'):
    """
    This method applys L2 Regularization to all weights and adds it to the ``objectives`` 
    collection. 
    
    Args:
        name: For the tensorflow scope.
        var_list: List of variables to apply.
    
    Notes:
        What is the co-efficient of the L2 weight? Set ``WEIGHT_DECAY_COEFF``.( Default = 0.0001 )
    """                              
    for param in var_list:
        norm = WEIGHT_DECAY_COEFF * tf.nn.l2_loss(param)
        tf.summary.scalar('l2_' + param.name, norm)                  
        tf.add_to_collection(name + '_objectives', norm)

def apply_l1 ( var_list, name = 'l1'):
    """
    This method applys L1 Regularization to all weights and adds it to the ``objectives`` 
    collection. 
    
    Args:
        var_list: List of variables to apply l1
        name: For the tensorflow scope.
    
    Notes:
        What is the co-efficient of the L1 weight? Set ``L1_COEFF``.( Default = 0.0001 )
    """                              
    for param in var_list:
        norm = L1_COEFF * tf.reduce_sum(tf.abs(param, name = 'abs'), name = 'l1')
        tf.summary.scalar('l1_' + param.name, norm)                  
        tf.add_to_collection(name + '_objectives', norm)

def process_params(params, name):
    """
    This method adds the params to two collections.
    The first element is added to ``regularizer_worthy_params``.
    The first and second elements are is added to ``trainable_parmas``.

    Args:
        params: List of two.
        name: For the scope
    """
    tf.add_to_collection(name + '_trainable_params', params[0])
    tf.add_to_collection(name + '_trainable_params', params[1])         
    tf.add_to_collection(name + '_regularizer_worthy_params', params[0]) 

def apply_regularizer (name, var_list):
    """
    This method applys Regularization to all weights and adds it to the ``objectives`` 
    collection. 
    
    Args:
        var_list: List of variables to apply l1
        name: For the tensorflow scope.
    
    Notes:
        What is the co-efficient of the L1 weight? Set ``L1_COEFF``.( Default = 0.0001 )
    """       
    with tf.variable_scope(name + '_weight-decay') as scope:
        if WEIGHT_DECAY_COEFF > 0:
            apply_weight_decay(name = name + 'weight_decay', var_list = var_list )

    with tf.variable_scope(name + '_l1-regularization') as scope:
        if L1_COEFF > 0:
            apply_l1(name =name + '_weight_decay',  var_list = var_list)


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

            # Unflatten Layer
            images_square = unflatten_layer ( self.images )
            visualize_images(images_square)

            # Conv Layer 1
            conv1_out, params =  conv_2d_layer (    input = images_square,
                                                    neurons = EXPERT_C1,
                                                    filter_size = EXPERT_F1,
                                                    name = 'conv_1',
                                                    visualize = True )
            process_params(params, name = self.name)
            pool1_out = max_pool_2d_layer ( input = conv1_out, name = 'pool_1')
            lrn1_out = local_response_normalization_layer (pool1_out, name = 'lrn_1' )

            # Conv Layer 2
            conv2_out, params =  conv_2d_layer (    input = lrn1_out,
                                                    neurons = EXPERT_C2,
                                                    filter_size = EXPERT_F2,
                                                    name = 'conv_2' )
            process_params(params, name = self.name)
            
            pool2_out = max_pool_2d_layer ( input = conv2_out, name = 'pool_2')
            lrn2_out = local_response_normalization_layer (pool2_out, name = 'lrn_2' )

            flattened = flatten_layer(lrn2_out)

            # Placeholder probability for dropouts.
            #self.dropout_prob = tf.placeholder(tf.float32,            
            self.dropout_prob = tf.placeholder_with_default(
                                                input = tf.constant(1.0, dtype = tf.float32),
                                                shape = None,
                                                name = 'dropout_probability')            

            # Dropout Layer 1 
            flattened_dropout = dropout_layer ( input = flattened,
                                            prob = self.dropout_prob,
                                            name = 'dropout_1')                                          

            # Dot Product Layer 1
            fc1_out, params = dot_product_layer  (  input = flattened_dropout,
                                                    neurons = EXPERT_D1,
                                                    name = 'dot_1')
            process_params(params, name = self.name)

            # Dropout Layer 2 
            fc1_out_dropout = dropout_layer ( input = fc1_out,
                                            prob = self.dropout_prob,
                                            name = 'dropout_2')
            # Dot Product Layer 2
            fc2_out, params = dot_product_layer  (  input = fc1_out_dropout, 
                                                    neurons = EXPERT_D2,
                                                    name = 'dot_2')
            process_params(params, name = self.name)

            # Dropout Layer 3 
            fc2_out_dropout = dropout_layer ( input = fc2_out,
                                            prob = self.dropout_prob,
                                            name = 'dropout_3')

            # Logits layer
            self.logits, params = dot_product_layer  (  input = fc2_out_dropout,
                                                        neurons = C,
                                                        activation = 'identity',
                                                        name = 'logits_layer')
            process_params(params, name = self.name)

            # Softmax layer
            self.inference, self.predictions = softmax_layer (  input = self.logits,
                                                                name = 'softmax_layer' )                                                    

                
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
                loss = tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits ( 
                                                     labels = self.labels,
                                                     logits = self.logits)
                                                )
                self.cost = loss
                tf.add_to_collection( self.name + '_objectives', loss ) 
                tf.summary.scalar('cost', loss)  

            apply_regularizer (name = self.name, var_list = tf.get_collection(
                                                    self.name + '_regularizer_worthy_params') )
            self.obj = tf.add_n(tf.get_collection( self.name + '_objectives'), name='objective')
    
        with tf.variable_scope(self.name + '_train') as scope:
            # Change (supply as arguments) parameters here directly in the code.
            if OPTIMIZER == 'sgd':                                                                              
                self.back_prop = apply_gradient_descent(var_list = tf.get_collection(self.name + '_trainable_params'),
                                        obj = self.obj )
            elif OPTIMIZER == 'rmsprop':
                self.back_prop = apply_rmsprop(var_list = tf.get_collection(self.name + '_trainable_params') ,
                                        obj = self.obj)
            elif OPTIMIZER == 'adam':
                self.back_prop = apply_adam (var_list = tf.get_collection( self.name + '_trainable_params') ,
                                        obj = self.obj )
            else:
                raise Error('Invalid entry to optimizer')

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

            # Unflatten Layer
            images_square = unflatten_layer ( self.images )
            visualize_images(images_square)

            # Placeholder probability for dropouts.
            #self.dropout_prob = tf.placeholder(tf.float32,
            self.dropout_prob = tf.placeholder_with_default(
                                                input = tf.constant(1.0, dtype = tf.float32),
                                                shape = None,
                                                name = 'dropout_probability')   

            # Dropout Layer 1 
            dropped_out = dropout_layer ( input = self.images,
                                            prob = self.dropout_prob,
                                            name = 'dropout_1')                                          

            # Dot Product Layer 1
            fc1_out, params = dot_product_layer  (  input = dropped_out,
                                                    neurons = NOVICE_D1,
                                                    name = 'dot_1')
            process_params(params, name = self.name)
            d1_params = params

            # Dropout Layer 2 
            fc1_out_dropout = dropout_layer ( input = fc1_out,
                                            prob = self.dropout_prob,
                                            name = 'dropout_2')
            # Dot Product Layer 2
            fc2_out, params = dot_product_layer  (  input = fc1_out_dropout, 
                                                    neurons = NOVICE_D2,
                                                    name = 'dot_2')
            process_params(params, name = self.name)
            d2_params = params 

            # Dropout Layer 3 
            fc2_out_dropout = dropout_layer ( input = fc2_out,
                                            prob = self.dropout_prob,
                                            name = 'dropout_3')

            # Logits layer
            self.logits, params = dot_product_layer  (  input = fc2_out_dropout,
                                                        neurons = C,
                                                        activation = 'identity',
                                                        name = 'logits_layer')
            process_params(params, name = self.name)

            # Softmax layer
            self.inference, self.predictions = softmax_layer (  input = self.logits,
                                                                name = 'softmax_layer' )                                                    

            self.params = [d1_params, d2_params]

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
            self.cost = tf.constant(0., dtype = tf.float32)
            if labels is not None:
                self.labels = labels                
                with tf.variable_scope( self.name + '_cross-entropy') as scope:
                    ce_loss = tf.reduce_mean(
                                tf.nn.softmax_cross_entropy_with_logits ( 
                                                        labels = self.labels,
                                                        logits = self.logits)
                                                    )                                                
                tf.add_to_collection( self.name + '_objectives', ce_loss )                                                    
                self.cost = self.cost + ce_loss
                tf.summary.scalar('cross_entropy_cost', ce_loss)  
            
            if judgement is not None:
                self.judgement = judgement
                with tf.variable_scope( self.name + '_fooler') as scope:
                    # Use the LS GAN technique. 
                    j_loss = tf.reduce_mean((1 - judgement)**2)
                    # j_loss = tf.reduce_mean(tf.log(1-judgement))
                tf.add_to_collection( self.name + '_objectives', j_loss )                                                    
                self.cost = self.cost + j_loss
                tf.summary.scalar('judge_cost', j_loss)  
            
            tf.summary.scalar('combined_cost', self.cost)
            apply_regularizer (name = self.name, var_list = tf.get_collection(
                                                    self.name + '_regularizer_worthy_params') )
            self.obj = tf.add_n(tf.get_collection( self.name + '_objectives'), name='objective')
            tf.summary.scalar('total_objective', self.obj)

        with tf.variable_scope(self.name + '_train') as scope:
            # Change (supply as arguments) parameters here directly in the code.
            if OPTIMIZER == 'sgd':                                                                              
                self.back_prop = apply_gradient_descent(var_list = tf.get_collection(self.name + '_trainable_params'),
                                        obj = self.obj )
            elif OPTIMIZER == 'rmsprop':
                self.back_prop = apply_rmsprop(var_list = tf.get_collection(self.name + '_trainable_params') ,
                                        obj = self.obj)
            elif OPTIMIZER == 'adam':
                self.back_prop = apply_adam (var_list = tf.get_collection( self.name + '_trainable_params') ,
                                        obj = self.obj )
            else:
                raise Error('Invalid entry to optimizer')

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
                    input_params = None,
                    name = 'judge' ):
        """
        Class constructor. Creates the model and all the connections. 
        """
        self.labels = labels
        with tf.variable_scope(name) as scope:
            self.images = images
            self.name = name

            # Unflatten Layer
            images_square = unflatten_layer ( self.images )
            visualize_images(images_square)

            # Placeholder probability for dropouts.
            #self.dropout_prob = tf.placeholder(tf.float32,
            self.dropout_prob = tf.placeholder_with_default(
                                                input = tf.constant(1.0, dtype = tf.float32),
                                                shape = None,
                                                name = 'dropout_probability')   

            # Dropout Layer 1 
            dropped_out = dropout_layer (   input = self.images,
                                            prob = self.dropout_prob,
                                            name = 'dropout_1')                                          

            par = None
            # Dot Product Layer 1
            if input_params is not None:
                par = input_params[0]
            fc1_out, params = dot_product_layer  (  input = dropped_out,
                                                    neurons = JUDGE_D1,
                                                    params = par,
                                                    name = 'dot_1')
            process_params(params, name = self.name)

            # Dropout Layer 2 
            fc1_out_dropout = dropout_layer ( input = fc1_out,
                                            prob = self.dropout_prob,
                                            name = 'dropout_2')

            if input_params is not None:
                par = input_params[1] 
                                          
            # Dot Product Layer 2
            fc2_out, params = dot_product_layer  (  input = fc1_out_dropout, 
                                                    neurons = JUDGE_D2,
                                                    params = par,
                                                    name = 'dot_2')
            process_params(params, name = self.name)

            # Dropout Layer 3 
            fc2_out_dropout = dropout_layer ( input = fc2_out,
                                            prob = self.dropout_prob,
                                            name = 'dropout_3')

            # Embedding layers
            expert_embed, params = dot_product_layer (input = expert,   
                                              neurons = EMBED,
                                              name = 'expert_embed' )
            novice_embed, params = dot_product_layer (input = novice,
                                              neurons = EMBED,
                                              params = params,
                                              name = 'novice_embed' )
            process_params ( params, name = self.name )
            
            #Judgement Layers
            merged_expert = tf.concat([expert_embed, fc2_out_dropout], 1)
            merged_novice = tf.concat([novice_embed, fc2_out_dropout], 1)            

            # Merged Expert Dropout Layer 1 
            merged_expert_dropout_1 = dropout_layer ( input = merged_expert,
                                            prob = self.dropout_prob,
                                            name = 'merged_expert_dropout_1')
            # Merged Novice Dropout Layer 1 
            merged_novice_dropout_1 = dropout_layer ( input = merged_novice,
                                            prob = self.dropout_prob,
                                            name = 'merged_novice_dropout_1')

            # Merged Expert Dot Product Layer 1
            merged_expert_fc1_out, params = dot_product_layer  (  
                                            input = merged_expert_dropout_1, 
                                            neurons = MERGED_D1,
                                            name = 'merged_expert_d1')
            # Merged Novice Dot Product Layer 1
            merged_novice_fc1_out, params = dot_product_layer  (  
                                            input = merged_novice_dropout_1, 
                                            params = params,
                                            neurons = MERGED_D1,
                                            name = 'merged_novice_d1')
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
            self.cost = tf.reduce_mean((1 - self.judgement_expert)**2) +\
                                 tf.reduce_mean(self.judgement_novice**2)            
            # self.cost = tf.reduce_mean(tf.log(1 - self.judgement_expert)) + \
            #            tf.reduce_mean(tf.log(self.judgement_novice))
            tf.add_to_collection( self.name + '_objectives', self.cost )                                                    
            tf.summary.scalar('cost', self.cost)
            apply_regularizer (name = self.name, var_list = tf.get_collection(
                                                    self.name + '_regularizer_worthy_params') )
            self.obj = tf.add_n(tf.get_collection( self.name + '_objectives'), name='objective')
            tf.summary.scalar('total_objective', self.obj)

        with tf.variable_scope(self.name + '_train') as scope:
            # Change (supply as arguments) parameters here directly in the code.
            if OPTIMIZER == 'sgd':                                                                              
                self.back_prop = apply_gradient_descent(var_list = tf.get_collection(self.name + '_trainable_params'),
                                        obj = self.obj )
            elif OPTIMIZER == 'rmsprop':
                self.back_prop = apply_rmsprop(var_list = tf.get_collection(self.name + '_trainable_params') ,
                                        obj = self.obj)
            elif OPTIMIZER == 'adam':
                self.back_prop = apply_adam (var_list = tf.get_collection( self.name + '_trainable_params') ,
                                        obj = self.obj )
            else:
                raise Error('Invalid entry to optimizer')

if __name__ == '__main__':
    pass                    