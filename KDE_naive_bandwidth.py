import numpy as np
from sklearn.neighbors import KernelDensity
from metropt_preprocessing import prepare_failure_free_dataset, read_window

TRAINING_HORIZON = 50000
TESTING_HORIZON = 50000
SANITY_LEAVEOUT_GAP = 100
HORIZON_SHIFT = 50000 
VALIDATION_HORIZON = 10000
BASIC_BANDWIDTH = 0.005
TEST_LEAVEOUT_SIZE = 50000
CSV_PATH = "MetroPT3(AirCompressor).csv"
CACHE_PATH = "MetroPT3_features_without_failures.npy"
FEATURE_COLUMNS = [
    "TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature",
    "Motor_current", "COMP", "DV_eletric", "Towers", "MPG", "LPS",
    "Pressure_switch", "Oil_level", "Caudal_impulses",
]
SCORING_BATCH_SIZE = 256
TRAINING_BATCH_SIZE = 2_000
CSV_READ_CHUNK_SIZE = 50_000

if __name__ == "__main__":
    dataset = prepare_failure_free_dataset(CSV_PATH, CACHE_PATH, FEATURE_COLUMNS, CSV_READ_CHUNK_SIZE)
    row_count = len(dataset)
    test_start = row_count - TEST_LEAVEOUT_SIZE
    usable_size = test_start
    steps = ((usable_size - TRAINING_HORIZON - VALIDATION_HORIZON)
             // HORIZON_SHIFT) + 1
    if steps <= 0:
        raise ValueError("The dataset is too short for the configured horizons")
    for bandwidth in np.geomspace(0.00005, 2, 20): # subsequently substituted with np.linspace()
        print(f"Bandwidth: {bandwidth}")
        print(f"Training the model in {steps} steps...")
        _surprise_scores = list()
        _within_variances = list()
        for step in range(steps):
            start = step * HORIZON_SHIFT
            training_data = read_window(
                dataset, start, TRAINING_HORIZON - SANITY_LEAVEOUT_GAP
            )
            validation_data = read_window(
                dataset, start + TRAINING_HORIZON, VALIDATION_HORIZON
            )
            estimator = KernelDensity(bandwidth=bandwidth).fit(training_data)
            surprise_scores = estimator.score_samples(validation_data)
            _surprise_scores.append(surprise_scores)
            print(f"\rTrained step {step}/{steps}           ", end="")
        chosen_threshold = np.quantile(-np.concatenate(_surprise_scores), 0.95) 
        print(f"The bandwidth induces a mean log-likelihood of: {np.mean(np.concatenate(_surprise_scores))}")
        print("Chosen surprise score threshold:", chosen_threshold)
