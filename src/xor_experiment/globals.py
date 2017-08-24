# Expert network
EXPERT = 5    # Number of neurons in first dot-product layer
C = 2      # Number of classes in the dataset to predict   

# Novice Network
NOVICE = 5    # Number of neurons in first dot-product layer

# Judge Network
MERGED = 5  # Number of neurons in the judge hidden-layer

TEMPERATURE = 0.3  # Temperature of softmax

# Dataset sizes
TRAIN_SET_SIZE = 10000
TEST_SET_SIZE = 10000

# Optimizer properties
LR = 0.1   # Learning rate 
OPTIMIZER = 'sgd' # Optimizer (options include 'adam', 'rmsprop') Easy to upgrade if needed.

# Train options
MINI_BATCH_SIZE = 10 # Mini batch size 
UPDATE_AFTER_ITER = (TRAIN_SET_SIZE / MINI_BATCH_SIZE )  # Update after these many iterations.
EXPERT_ITER = (TRAIN_SET_SIZE / MINI_BATCH_SIZE )  * 15 # Total number of iterations to run
JUDGED_ITER = (TRAIN_SET_SIZE / MINI_BATCH_SIZE )  * 30 # Total number of iterations to run
K = 1 # After how many iterations of judge should you update novice.

DROPOUT_PROBABILITY   = 1.0

if __name__ == '__main__':
    pass