# --- Probability calibration ---
# Options: 'none', 'temperature', 'platt', 'isotonic'
CALIBRATION_METHOD = 'temperature'
# --- Assignment method ---
# Options: 'optimal' (Hungarian), 'greedy' (legacy uniqueness)
ASSIGNMENT_METHOD = 'optimal'
# --- Custom loss weights ---
# Weight for Jaccard loss (set similarity)
JACCARD_LOSS_WEIGHT = 0.0
# Weight for duplicate penalty (uniqueness enforcement)
DUPLICATE_PENALTY_WEIGHT = 0.0
# If True, use custom loss (with overcount/entropy penalty) for MLP
MLP_USE_CUSTOM_LOSS = True
# LightGBM hyperparameters (for meta-optimization)
LGBM_NUM_LEAVES = 31
LGBM_LEARNING_RATE = 0.1
LGBM_MAX_DEPTH = 7

# --- Over-prediction penalty ---
# Penalty weight for predicted count > true count (squared penalty)
OVERCOUNT_PENALTY_WEIGHT = 0.5

# --- Overconfidence/entropy penalty ---
# Penalty weight for low-entropy (overconfident) predictions in the loss
ENTROPY_PENALTY_WEIGHT = 0.5
# Label smoothing and uniform prior for regularization
LABEL_SMOOTHING = 0.05
UNIFORM_MIX_PROB = 0.05
# Lower pseudo-label confidence threshold for more diversity
PSEUDO_CONFIDENCE_THRESHOLD = 0.7
# Minimum entropy for pseudo-label acceptance (per sample, per ball)
PSEUDO_MIN_ENTROPY = 2.5

# --- Iterative stacking ---
# If True, use previous predictions as meta-features for next run
ITERATIVE_STACKING = True
# Number of rounds for iterative stacking automation (only used if ITERATIVE_STACKING is True)
ITERATIVE_STACKING_ROUNDS = 3

# --- Ensembling strategy ---
# Options: 'average', 'weighted', 'stacking'
ENSEMBLE_STRATEGY = 'average'

# --- Cross-validation ---
# Set CV_FOLDS = 1 for no cross-validation (standard train/test split)
CV_FOLDS = 5

# --- Meta-parameter optimization method ---
# Options: 'pso', 'bayesian'
META_OPT_METHOD = 'pso'
# config.py
# Central configuration for all tunable variables in the LotteryPrediction project

# --- Data paths ---
KAGGLE_CSV_FILE = 'data_sets/powerball_usa.csv'
BASE_DATASET_FILE = 'data_sets/base_dataset.csv'

# --- Data split ---
TRAIN_SPLIT = 0.8
LOOK_BACK_WINDOW = 10  # You can now tune this manually; try 5, 20, 30, etc.

# --- Model training ---
EPOCHS_TUNER = 40
EPOCHS_FINAL = 80
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# --- KerasTuner ---
TUNER_MAX_TRIALS = 40
TUNER_EXECUTIONS_PER_TRIAL = 1
TUNER_DIRECTORY = 'hypertune_dir'
TUNER_PROJECT_NAME = 'lstm_lottery'

# --- Label smoothing and uniform mixing ---
LABEL_SMOOTHING = 0.0
UNIFORM_MIX_PROB = 0.0
# --- Debug: Only run LSTM model for diagnostics ---
ONLY_LSTM_MODEL = False


# --- Early stopping ---
EARLY_STOPPING_PATIENCE = 3  # PSO meta-param

# --- Model complexity ---
FORCE_LOW_UNITS = True
FORCE_SIMPLE = True

# --- Temperature grid search (PSO meta-params)
TEMP_MIN = 0.5
TEMP_MAX = 2.0
TEMP_STEP = 0.1

# --- PSO and KerasTuner search sizes ---
PSO_PARTICLES = 3  # For quick test, increase for final runs
PSO_ITER = 3
KERAS_TUNER_MAX_TRIALS = 10
KERAS_TUNER_EXECUTIONS_PER_TRIAL = 1

# --- Random seed (optional) ---
RANDOM_SEED = 42
