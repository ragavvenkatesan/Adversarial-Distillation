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

MERGED_D1 = 800
MERGED_D2 = 800


TEMPERATURE = 1.0 

# Optimizer properties
LR = 1e-3   # Learning rate 
WEIGHT_DECAY_COEFF = 0.00001 # Co-Efficient for weight decay
L1_COEFF = 0.00001 # Co-Efficient for L1 Norm
MOMENTUM = 0.7 # Momentum rate 
OPTIMIZER = 'rmsprop' # Optimizer (options include 'adam', 'rmsprop') Easy to upgrade if needed.
DROPOUT_PROBABILITY   = 1.0

# Dataset sizes
TRAIN_SET_SIZE = 50000
TEST_SET_SIZE = 10000

# Train options
DECAY = 0.95
MINI_BATCH_SIZE = 500 # Mini batch size 
UPDATE_AFTER_ITER = (TRAIN_SET_SIZE / MINI_BATCH_SIZE ) # Update after these many iterations.
EXPERT_ITER = (TRAIN_SET_SIZE / MINI_BATCH_SIZE )  * 30 # Total number of iterations to run
JUDGED_ITER = (TRAIN_SET_SIZE / MINI_BATCH_SIZE )  * 100 # Total number of iterations to run

# Higher the values, less frequent the updates.
K = 1 # After how many iterations of judge should you update novice.
R = 1 # After how many iterations of novice should you update judge.

if __name__ == '__main__':
    pass