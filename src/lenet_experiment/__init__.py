import sys 
sys.path.append('../')

from tools.trainer import trainer, poly_trainer
from tools.dataset import mnist

from network import expert, novice, judge     
from globals import MINI_BATCH_SIZE

def run_expert(dataset):
    """
    This method creates and runs the expert network.

    Args:
        dataset: Supply a dataset object

    Returns: 
        network.expert: Expert network.
    """
    print (" \n\n Expert Assembly \n\n")
    expert_net = expert(images = dataset.images)  
    expert_net.cook(    labels = dataset.labels)
    expert_bp = trainer(expert_net, dataset.feed,
                        tensorboard = 'expert')
    print (" \n\n Expert Training \n\n")    
    expert_bp.train(    iter = 100 * 100, 
                        update_after_iter = 1000,    
                        mini_batch_size = MINI_BATCH_SIZE, 
                        summarize = True   )
    return expert_net

def run_independent_novice(dataset):
    """
    This method creates and runs the novice as an independent network.

    Args:
        dataset: Supply a dataset object
    
    Returns:
        network.novice: Independent network.
    """
    print (" \n\n Independent Novice Assembly \n\n")
    indep_net = novice(images = dataset.images, 
                    name = 'novice_independent')  
    indep_net.cook( labels = dataset.labels )
    indep_bp = trainer(indep_net, dataset.feed)
    print (" \n\n Independent Novice Training \n\n")    
    indep_bp.train( iter= 100 * 100, 
                    mini_batch_size = MINI_BATCH_SIZE, 
                    update_after_iter = 1000,
                    summarize = True)   
    return indep_net

def run_judged_novice(dataset, expert  ):
    """
    This method creates and runs the novice

    Args:
        dataset: Supply a dataset object.
    
    Returns:
        tuple: ``(network.novice, network.judge)``.  
    """
    print (" \n\n Judged Novice Without Labels Assembly \n\n")
    novice_net = novice( images = dataset.images,
                    name = 'novice_judged' )  
    judge_net = judge( images = dataset.images,
                       expert = expert,
                       novice = novice_net.inference,
                       input_params = novice_net.params,
                       labels = dataset.labels )

    novice_net.cook( judgement = judge_net.judgement_novice,
                     labels = dataset.labels,
                     test_labels = dataset.labels )
    judge_net.cook ( )

    judged_mentoring = poly_trainer ([novice_net, judge_net], 
                                        dataset.feed,
                                        tensorboard = 'judged',                
                                        )

    print (" \n\nJudged Novice Without Labels Mentored Training \n\n")    
    judged_mentoring.train( iter= 100 * 100,
                    k =1, 
                    update_after_iter = 1000,
                    mini_batch_size = MINI_BATCH_SIZE,     
                    summarize = True   )    

    return (novice_net, judge_net)

if __name__ == '__main__':
    dataset = mnist()   
    expert_net = run_expert(dataset)
    indep_net = run_independent_novice(dataset)
    novice_net, judge_net = run_judged_novice(
                                        dataset = dataset,
                                        expert = expert_net.inference )