import os

# --- Paths -------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "spam.csv")
PROCESSED_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "clean_spam.csv")

MODEL_DIR = os.path.join(ROOT_DIR, "models")
PIPELINE_PATH = os.path.join(MODEL_DIR, "spam_pipeline.joblib")
RESULTS_PATH = os.path.join(MODEL_DIR, "results_summary.csv")
CV_RESULTS_PATH = os.path.join(MODEL_DIR, "cv_results.csv")

# --- Reproducibility -----------------------------------------------------
RANDOM_STATE = 42

# --- Train/test split ---------------------------------------------------
TEST_SIZE = 0.2

# --- Cross-validation -----------------------------------------------------
CV_FOLDS = 4

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
