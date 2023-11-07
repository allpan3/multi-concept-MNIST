###########
# Configs #
###########
VERBOSE = 1
SEED = 0
ALGO = "algo1" # "algo1", "algo2"
VSA_MODE = "HARDWARE" # "SOFTWARE", "HARDWARE"
QUANTIZE_MODEL = False 
DIM = 1024
MAX_NUM_OBJECTS = 9
SINGLE_COUNT = False # True, False
NUM_POS_X = 3
NUM_POS_Y = 3
NUM_COLOR = 7
COUNT_KNOWN = True
# Hardware config
FOLD_DIM = 128
EHD_BITS = 8
SIM_BITS = 13
# Train
TRAIN_EPOCH = 100
TRAIN_BATCH_SIZE = 128
NUM_TRAIN_SAMPLES = 100000
# Test
TEST_BATCH_SIZE = 1
NUM_TEST_SAMPLES_PER_OBJ = 100
NUM_TEST_SAMPLES = NUM_TEST_SAMPLES_PER_OBJ if SINGLE_COUNT else NUM_TEST_SAMPLES_PER_OBJ * MAX_NUM_OBJECTS
# Resonator
RESONATOR_TYPE = "SEQUENTIAL" # "SEQUENTIAL", "CONCURRENT"
MAX_TRIALS = MAX_NUM_OBJECTS + 10
NUM_ITERATIONS = 200
ACTIVATION = 'THRESH_AND_SCALE'      # 'IDENTITY', 'THRESHOLD', 'SCALEDOWN', "THRESH_AND_SCALE"
ACT_VALUE = 16
STOCHASTICITY = "SIMILARITY"  # apply stochasticity: "NONE", "SIMILARITY", "VECTOR"
RANDOMNESS = 0.04
# Similarity thresholds are affected by the maximum number of vectors superposed. These values need to be lowered when more vectors are superposed
SIM_EXPLAIN_THRESHOLD = 0.20                 # When count is known, we can typically use a lower threshold. Still can't be too low otherwise the same object may get counted multiple times
SIM_DETECT_THRESHOLD = 0.15
ENERGY_THRESHOLD = 0.25
EARLY_CONVERGE = 0.6
EARLY_TERM_THRESHOLD = 0.15     # Compared to remaining

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
if EARLY_CONVERGE is not None and (ACTIVATION == "SCALEDOWN" or ACTIVATION == "THRESH_AND_SCALE"):
    EARLY_CONVERGE = EARLY_CONVERGE / ACT_VALUE
