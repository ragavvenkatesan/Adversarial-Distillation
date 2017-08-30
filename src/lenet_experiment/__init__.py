import sys 
sys.path.append('../')

from tools.trainer import trainer, poly_trainer

from dataset import mnist, fashion_mnist
from network import expert, novice, judge, autoencoder     
from globals import *


if __name__ == '__main__':
    dataset = mnist()
    # dataset = fashion_mnist()
    ################ Expert ################
    print (" \n\n Expert Assembly \n\n")
    expert_net = expert(images = dataset.images)     
    expert_net.cook(    labels = dataset.labels)
    expert_bp = trainer(expert_net, dataset.feed,
                        init_vars = False,
                        tensorboard = 'expert')

    ################ Indpendent Novice ################
    """
    print (" \n\n Independent Novice Assembly \n\n")
    indep_net = novice(images = dataset.images, 
                    name = 'novice_independent')  
    indep_net.cook( labels = dataset.labels )
    indep_bp = trainer( indep_net, 
                        dataset = dataset.feed,
                        session = expert_bp.session,
                        tensorboard = 'independent',
                        init_vars = False )
    """

    
    ################ Adversarial Distillation ###########
    
    print (" \n\n Judged Novice Without Labels Assembly \n\n")
    ae_net = autoencoder ( images = dataset.images,
                            name = 'autoencoder')
                        
    novice_net = novice( images = dataset.images,
                         input_params = ae_net.params,
                         name = 'novice_judged' )  

    judge_net = judge( images = dataset.images,
                       expert = expert_net.temperature_softmax,
                       novice = novice_net.temperature_softmax,
                       input_params = ae_net.params,
                       labels = dataset.labels )
    
    ae_net.cook( labels = dataset.labels )
    ae_net_bp = trainer(ae_net, dataset.feed, init_vars = False,
                        session = expert_bp.session,
                        tensorboard = 'autoencoder')
    
    novice_net.cook( judgement = judge_net.judgement_novice,
                     # labels = dataset.labels,
                     test_labels = dataset.labels )
    judge_net.cook ()
    judged_mentoring = poly_trainer ([novice_net, judge_net], 
                                        dataset.feed,
                                        tensorboard = 'judged', 
                                        init_vars = True,
                                        session = expert_bp.session               
                                        )

    
    ####### Training ################
    print (" \n\n Expert Training \n\n")  
    expert_bp.train(    iter = EXPERT_ITER, 
                        update_after_iter = UPDATE_AFTER_ITER,    
                        mini_batch_size = MINI_BATCH_SIZE, 
                        summarize = True   )

    """
    print (" \n\n Independent Novice Training \n\n")    
    indep_bp.train( iter= EXPERT_ITER, 
                    mini_batch_size = MINI_BATCH_SIZE, 
                    update_after_iter = UPDATE_AFTER_ITER,
                    summarize = True)   
    

    print (" \n\n Autoencoder Training \n\n")  
    ae_net_bp.train(    iter = EXPERT_ITER, 
                        update_after_iter = UPDATE_AFTER_ITER ,    
                        mini_batch_size = MINI_BATCH_SIZE, 
                        summarize = True   )
    """
    print (" \n\nJudged Novice Without Labels Mentored Training \n\n")    
    judged_mentoring.train( iter= JUDGED_ITER ,
                    k = K, 
                    r = R,
                    update_after_iter = UPDATE_AFTER_ITER,
                    mini_batch_size = MINI_BATCH_SIZE,  
                    expert = expert_net.temperature_softmax,   
                    summarize = True   )