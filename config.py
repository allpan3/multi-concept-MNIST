###########
# Configs #
###########
VERBOSE = 2
SEED = 0
ALGO = "algo1" # "algo1", "algo2"
VSA_MODE = "HARDWARE" # "SOFTWARE", "HARDWARE"
QUANTIZE_MODEL = False 
DIM = 1024
MAX_NUM_OBJECTS = 3
SINGLE_COUNT = False # True, False
NUM_POS_X = 3
NUM_POS_Y = 3
NUM_COLOR = 7
# Hardware config
FOLD_DIM = 256
EHD_BITS = 9
SIM_BITS = 13
GEMMINI_DIM = 16
# Train
TRAIN_EPOCH = 50
TRAIN_BATCH_SIZE = 128
NUM_TRAIN_SAMPLES = 10000
# Test
TEST_BATCH_SIZE = 1
NUM_TEST_SAMPLES = 300
# Resonator
RESONATOR_TYPE = "SEQUENTIAL" # "SEQUENTIAL", "CONCURRENT"
MAX_TRIALS = MAX_NUM_OBJECTS + 10
NUM_ITERATIONS = 200
ACTIVATION = 'THRESH_AND_SCALE'      # 'IDENTITY', 'THRESHOLD', 'SCALEDOWN', "THRESH_AND_SCALE"
ACT_VALUE = 16
STOCHASTICITY = "SIMILARITY"  # apply stochasticity: "NONE", "SIMILARITY", "VECTOR"
RANDOMNESS = 0.04
# Similarity thresholds are affected by the maximum number of vectors superposed. These values need to be lowered when more vectors are superposed
SIM_EXPLAIN_THRESHOLD = 0.25
SIM_DETECT_THRESHOLD = 0.12
ENERGY_THRESHOLD = 0.25
EARLY_CONVERGE = None
EARLY_TERM_THRESHOLD = 0.2     # Compared to remaining

# In hardware mode, the activation value needs to be a power of two
if VSA_MODE == "HARDWARE" and (ACTIVATION == "SCALEDOWN" or ACTIVATION == "THRESH_AND_SCALE"):
    def biggest_power_two(n):
        """Returns the biggest power of two <= n"""
        # if n is a power of two simply return it
        if not (n & (n - 1)):
            return n
        # else set only the most significant bit
        return int("1" + (len(bin(n)) - 3) * "0", 2)
    ACT_VALUE = biggest_power_two(ACT_VALUE)

# If activation is scaledown, then the early convergence threshold needs to scale down accordingly
if EARLY_CONVERGE is not None and (ACTIVATION == "SCALEDOWN" or ACTIVATION == "THRESHOLDED_SCALEDOWN"):
    EARLY_CONVERGE = EARLY_CONVERGE / ACT_VALUE
