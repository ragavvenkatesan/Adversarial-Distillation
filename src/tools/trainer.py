import tensorflow as tf
DROPOUT_PROBABILITY   = 1.0

config = tf.ConfigProto(
        device_count = {'GPU': 2}
    )

class trainer(object):
    """
    Trainer for networks

    Args:
        network: A network class object
        dataset: A tensorflow dataset object
        session: Supply a session to run on, if nothing is provided,
            a new session will be opened. 
        tensorboard: Name for the tensorboard.
        init_vars: if ``True`` will initialize all global variables.

    Class Properties:
        These are variables of the class that are available outside. 
        
        *   ``network``: This is the network we initialized with.
        *   ``dataset``: This is also the initializer.     
        *   ``session``: This is a session created with trainer.
        *   ``tensorboard``: Is a summary writer tool. 
        
    """
    def __init__(   self, network, dataset,
                    session = None,
                    init_vars = True,
                    tensorboard = 'tensorboard'):
        """
        Class constructor
        """
        self.network = network
        self.dataset = dataset 
        if session is None:
            self.session = tf.InteractiveSession(config=config)        
        else:
            self.session = session
        if init_vars is True:
            tf.global_variables_initializer().run()
        self.summaries(name = tensorboard)

    def bp_step(self, mini_batch_size):
        """
        Sample a minibatch of data and run one step of BP.

        Args:
            mini_batch_size: Integer
        
        Returns: 
            tuple of tensors: total objective and cost of that step
        """
        x, y = self.dataset.train.next_batch(mini_batch_size)
        _, obj = self.session.run(  
                           fetches = [self.network.back_prop, self.network.obj], \
                           feed_dict = {self.network.images:x, self.network.labels:y, \
                                        self.network.dropout_prob: DROPOUT_PROBABILITY})
        return obj

    def accuracy (self, images, labels):
        """
        Return accuracy

        Args:
            images: images
            labels: labels

        Returns:
            float: accuracy            
        """
        acc = self.session.run(self.network.accuracy,\
                               feed_dict = { self.network.images: images,
                                             self.network.labels: labels,
                                             self.network.dropout_prob: 1.0} )
        return acc

    def summaries(self, name = "tensorboard"):
        """
        Just creates a summary merge bufer

        Args:
            name: a name for the tensorboard directory
        """
        self.summary = tf.summary.merge_all()
        self.tensorboard = tf.summary.FileWriter(name)
        self.tensorboard.add_graph(self.session.graph)


    def test (self):
        """
        Run validation of the model  

        Returns:
            float: accuracy                     
        """
        x = self.dataset.test.images
        y = self.dataset.test.labels
        acc = self.accuracy (images =x, labels = y)                
        return acc

    def training_accuracy (self, mini_batch_size = 500):
        """
        Run validation of the model on training set   

        Args:
            mini_batch_size: Number of samples in a mini batch 
            
        Returns:
            float: accuracy                      
        """
        x, y = self.dataset.train.next_batch(mini_batch_size)
        acc = self.accuracy (images =x, labels = y)                
        return acc

    def write_summary (self, iter = 0, mini_batch_size = 500):
        """
        This method updates the summaries
        
        Args:
            iter: iteration number to index values with.
            mini_batch_size: Mini batch to evaluate on.
        """
        x = self.dataset.test.images
        y = self.dataset.test.labels       
        s = self.session.run(self.summary, feed_dict = {self.network.images: x,
                                                        self.network.labels: y,
                                                        self.network.dropout_prob: 1.0})
        self.tensorboard.add_summary(s, iter)

    def save (path = 'models/'):
        """ 
        Saves the models down
        """
        saver = tf.train.Saver()
        
        save_path = saver.save(self.session, path)
        print("Model saved in file: %s" % path)           

    def train ( self, 
                iter= 10000, 
                mini_batch_size = 500, 
                update_after_iter = 1000, 
                training_accuracy = False,
                summarize = True):
        """
        Run backprop for ``iter`` iterations

        Args:   
            iter: number of iterations to run
            mini_batch_size: Size of the mini batch to process with
            training_accuracy: if ``True``, will calculate accuracy on training data also.
            update_after_iter: This is the iteration for validation
            summarize: Tensorboard operation
        """
        for it in range(iter):            
            obj = self.bp_step(mini_batch_size)            
            if it % update_after_iter == 0:                                              
                train_acc = self.training_accuracy(mini_batch_size = mini_batch_size)
                acc = self.test()
                print(  " Iter " + str(it) +
                        " Objective " + str(obj) +
                        " Test Accuracy " + str(acc) +
                        " Training Accuracy " + str(train_acc) 
                        )                   
                if summarize is True:               
                    self.write_summary(iter = it, mini_batch_size = mini_batch_size)
        acc = self.test()
        print ("Final Test Accuracy: " + str(acc))    
        
        self.tensorboard.close()

    def close(self):
        """
        Closes the session with which we initialized the trainer.
        """
        self.session.close()

class poly_trainer(trainer):
    """
    Trainer for multiple networks trained simultaneously

    Args:
        nets: A List of networks.
        dataset: A tensorflow dataset object        
        session: Supply a session to run on, if nothing is provided,
            a new session will be opened. 
        init_vars: if ``True`` will initialize all global variables.            

    Notes:
        * This trainer is inherited from the :class:`tools.trainer.trainer`. 
        * This was implemeted for adversarial mentoring.
        * ``nets[0] = tools.network.novice()`` ,`` nets[1] = tools.network.judge()``

    Todo:
        Have the :math:`k` option available like in GANs.        
    """
    def __init__ (self, nets, dataset, session = None, 
                    init_vars = False, tensorboard = 'tensorboard'):
        """
        Class constructor
        """
        self.nets = nets
        self.dataset = dataset 
        if session is None:
            self.session = tf.InteractiveSession()        
        else:
            self.session = session
        if init_vars is True:
            tf.global_variables_initializer().run()
        self.summaries(name = tensorboard)    

    def bp_step(self, mini_batch_size, ind):
        """
        Sample a minibatch of data and run one step of BP.

        Args:
            mini_batch_size: Integer
            ind: Supply the list index of the net to update.
        
        Returns: 
            tuple of tensors: total objective and cost of that step
        """
        x, y = self.dataset.train.next_batch(mini_batch_size)
        _, obj = self.session.run(  
                           fetches = [  self.nets[ind].back_prop, 
                                        self.nets[ind].obj], \
                           feed_dict = {self.nets[ind].images:x,
                                        self.nets[ind].labels:y, \
                                        self.nets[ind].dropout_prob: DROPOUT_PROBABILITY})
        return obj

    def accuracy (self, images, labels, ind):
        """
        Return accuracy

        Args:
            images: images
            labels: labels
            ind: Supply the list index of the net to update.            

        Returns:
            float: accuracy            
        """
        acc = self.session.run(self.nets[ind].accuracy,\
                               feed_dict = { self.nets[ind].images: images,
                                             self.nets[ind].labels: labels,
                                             self.nets[ind].dropout_prob: 1.0} )
        return acc

    def training_accuracy (self, ind, mini_batch_size = 500):
        """
        Run validation of the model on training set   

        Args:
            mini_batch_size: Number of samples in a mini batch 
            
        Returns:
            float: accuracy                      
        """
        x, y = self.dataset.train.next_batch(mini_batch_size)
        acc = self.accuracy (images =x, labels = y, ind = ind)                
        return acc

    def test (self, ind):
        """
        Run validation of the model  

        Args:
            ind: Supply the list index of the net to update.            
        
        Returns:
            float: accuracy                     
        """
        x = self.dataset.test.images
        y = self.dataset.test.labels
        acc = self.accuracy (images =x, labels = y, ind = ind)                
        return acc

    def write_summary (self, iter = 0, mini_batch_size = 500):
        """
        This method updates the summaries
        
        Args:
            iter: iteration number to index values with.
            mini_batch_size: Mini batch to evaluate on.
        """
        ind = 0 
        x = self.dataset.test.images
        y = self.dataset.test.labels       
        s = self.session.run(self.summary, feed_dict = {self.nets[ind].images: x,
                                                        self.nets[ind].labels: y,
                                                        self.nets[ind].dropout_prob: 1.0})
        self.tensorboard.add_summary(s, iter)

    def print_judgement (self):
        """
        Prints the Probability nodes.
        """
        ind = 1         
        x = self.dataset.test.images
        y = self.dataset.test.labels         
        expert = self.session.run(tf.reduce_mean(self.nets[ind].judgement_expert),
                                            feed_dict = {self.nets[ind].images: x} )

        novice = self.session.run(tf.reduce_mean(self.nets[ind].judgement_novice),
                                            feed_dict = {self.nets[ind].images: x} )
        print (" Probability - Expert : " + str(expert) + " Novice : " + str(novice))

    def _inference_print (self):
        """
        Prints the inferences.
        """
        self.matrix = tf.concat ( [  self.nets[0].labels, \
                                self.expert, \
                                self.nets[0].inference, \
                                self.nets[1].judgement_expert, \
                                self.nets[1].judgement_novice
                                ], axis = 1 )

    def print_inference( self):
        """ 
        Prints inference
        """
        x = self.dataset.test.images[0:20]
        y = self.dataset.test.labels[0:20]        
        mat = self.session.run(self.matrix,
                                feed_dict = {self.nets[0].images: x,
                                            self.nets[0].labels: y}   )
        print ("\n\n\n")
        print (mat)
        print ("\n\n\n")

    def train ( self, 
                k = 1,
                iter= 10000, 
                mini_batch_size = 500, 
                update_after_iter = 1000, 
                training_accuracy = False,
                expert = None,
                summarize = True):
        """
        Run backprop for ``iter`` iterations

        Args:   
            iter: number of iterations to run
            mini_batch_size: Size of the mini batch to process with
            training_accuracy: if ``True``, will calculate accuracy on training data also.
            update_after_iter: This is the iteration for validation
            expert: Expert inference.
            summarize: Tensorboard operation
        """
        self.expert = expert
        self._inference_print()
        obj = [0] * len(self.nets)
        cost = [0] * len(self.nets)
    
        for it in range(iter):
            obj[1] = self.bp_step(mini_batch_size, 1)  # Update Judge                        
            if it % k == 0:                
                obj[0] = self.bp_step(mini_batch_size, 0)  # Update Novice                             
                
                
            if it % update_after_iter == 0:   
                train_acc = self.training_accuracy(mini_batch_size = mini_batch_size,
                                                    ind = 0)
                acc = self.test(ind = 0)
                print ("\n\n")
                self.print_judgement()
                print(  " Iter " + str(it) +
                        " Objective [" + str(obj[0]) +
                        " , " + str(obj[1]) + " ]"
                        " Test Accuracy " + str(acc) +
                        " Training Accuracy " + str(train_acc)
                         )            

                if summarize is True:               
                    self.write_summary(iter = it, mini_batch_size = mini_batch_size)
        acc = self.test(ind = 0)
        print ("Final Test Accuracy: " + str(acc))    
        self.tensorboard.close()
if __name__ == '__main__':
    pass