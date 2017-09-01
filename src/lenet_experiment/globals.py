# Some Global Defaults
# Expert network
EXPERT_C1 = 20     # Number of filters in first conv layer
EXPERT_C2 = 50     # Number of filters in second conv layer
EXPERT_F1 = (5,5)  # Size of the filters in the first conv layer
EXPERT_F2 = (3,3)  # Size of the filters in the second conv layer
EXPERT_D1 = 800    # Number of neurons in first dot-product layer
EXPERT_D2 = 800    # Number of neurons in second dot-product layer

C = 10      # Number of classes in the dataset to predict   

# Novice Network
NOVICE_D1 = 800    # Number of neurons in first dot-product layer
NOVICE_D2 = 800    # Number of neurons in second dot-product layer

# Judge Network
JUDGE_D1 = NOVICE_D1
JUDGE_D2 = NOVICE_D2

INFERENCE_EMBED = 100
IMAGE_EMBED = 100

MERGED_D1 = 200
MERGED_D2 = 200

IMAGE_SHAPE = 784
TEMPERATURE = 1
DROPOUT_PROBABILITY   = 0.5

# Optimizer properties of expert
EXPERT_LR = .01  # Learning rate 
EXPERT_WEIGHT_DECAY_COEFF = 0.0001 # Co-Efficient for weight decay
EXPERT_L1_COEFF = 0.0001 # Co-Efficient for L1 Norm
# EXPERT_MOMENTUM = 0.7 # Momentum rate 
EXPERT_OPTIMIZER = 'adam' # Optimizer (options include 'adam', 'adagrad', 'sgd',
                          # 'rmsprop') Easy to upgrade if needed.
# EXPERT_DECAY = 0.95

# Optimizer properties of autoencoder
AUTOENCODER_LR = .0001  # Learning rate 
AUTOENCODER_WEIGHT_DECAY_COEFF = 0.0001 # Co-Efficient for weight decay
AUTOENCODER_L1_COEFF = 0.0001 # Co-Efficient for L1 Norm
# AUTOENCODER_MOMENTUM = 0.7 # Momentum rate 
AUTOENCODER_OPTIMIZER = 'adam' # Optimizer (options include 'adam', 'adagrad', 'sgd',
                      # 'rmsprop') Easy to upgrade if needed.
# AUTOENCODER_DECAY = 0.95


# Optimizer properties of distillation
JUDGED_LR = .001  # Learning rate 
JUDGED_WEIGHT_DECAY_COEFF = 0.000 # Co-Efficient for weight decay
JUDGED_L1_COEFF = 0.000 # Co-Efficient for L1 Norm
# JUDGED_MOMENTUM = 0.7 # Momentum rate 
JUDGED_OPTIMIZER = 'rmsprop' # Optimizer (options include 'adam', 'adagrad', 'sgd',
                      # 'rmsprop') Easy to upgrade if needed.
# JUDGED_DECAY = 0.95


# Dataset sizes
TRAIN_SET_SIZE = 50000
TEST_SET_SIZE = 10000
# Train options
EXPERT_MINI_BATCH_SIZE = 500 # Mini batch size 
JUDGED_MINI_BATCH_SIZE = 100 # Mini batch size 
AUTOENCODER_MINI_BATCH_SIZE = 500 # Mini batch size 

EXPERT_UPDATE_AFTER_ITER = (TRAIN_SET_SIZE / EXPERT_MINI_BATCH_SIZE ) * 10 # Update after these many iterations.
AUTOENCODER_UPDATE_AFTER_ITER = (TRAIN_SET_SIZE / AUTOENCODER_MINI_BATCH_SIZE ) * 10 # Update after these many iterations.
JUDGED_UPDATE_AFTER_ITER = (TRAIN_SET_SIZE / JUDGED_MINI_BATCH_SIZE ) * 10 # Update after these many iterations.


EXPERT_ITER = (TRAIN_SET_SIZE / EXPERT_MINI_BATCH_SIZE )  * 100 # Total number of iterations to run
JUDGED_ITER = (TRAIN_SET_SIZE / JUDGED_MINI_BATCH_SIZE )  * 500 # Total number of iterations to run
AUTOENCODER_ITER = (TRAIN_SET_SIZE / AUTOENCODER_MINI_BATCH_SIZE )  * 100 # Total number of iterations to run

# Higher the values, less frequent the updates.
# Typically, at least one of these is set to 1. 
K = 1 # After how many iterations of judge should you update novice.
R = 1 # After how many iterations of novice should you update judge.

if __name__ == '__main__':
    pass