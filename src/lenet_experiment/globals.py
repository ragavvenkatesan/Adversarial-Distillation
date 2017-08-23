# Some Global Defaults
# Expert network
EXPERT_C1 = 20     # Number of filters in first conv layer
EXPERT_C2 = 50     # Number of filters in second conv layer
EXPERT_F1 = (5,5)  # Size of the filters in the first conv layer
EXPERT_F2 = (3,3)  # Size of the filters in the second conv layer
EXPERT_D1 = 1200    # Number of neurons in first dot-product layer
EXPERT_D2 = 1200    # Number of neurons in second dot-product layer

C = 10      # Number of classes in the dataset to predict   

# Novice Network
NOVICE_D1 = 800    # Number of neurons in first dot-product layer
NOVICE_D2 = 800    # Number of neurons in second dot-product layer

# Judge Network
JUDGE_D1 = 800
JUDGE_D2 = 800
EMBED = 100
MERGED_D1 = 900

# Optimizer properties
LR = 0.01   # Learning rate 
WEIGHT_DECAY_COEFF = 0.0000 # Co-Efficient for weight decay
L1_COEFF = 0.0000 # Co-Efficient for L1 Norm
MOMENTUM = 0.7 # Momentum rate 
OPTIMIZER = 'sgd' # Optimizer (options include 'adam', 'rmsprop') Easy to upgrade if needed.
MINI_BATCH_SIZE = 500 # Mini batch size 


if __name__ == '__main__':
    pass