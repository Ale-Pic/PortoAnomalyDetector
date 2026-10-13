import numpy as np
from sklearn.neighbors import KernelDensity
from metropt_preprocessing import prepare_failure_free_dataset, read_window

TRAINING_HORIZON = 50000
TESTING_HORIZON = 50000
SANITY_LEAVEOUT_GAP = 100
HORIZON_SHIFT = 50000
VALIDATION_HORIZON = 10000
BASIC_BANDWIDTH = 0.107
TEST_LEAVEOUT_SIZE = 60000
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
    print(f"Training the model in {steps} steps...")
    _surprise_scores = list()
    # _within_variances = list()
    for step in range(steps):
        start = step * HORIZON_SHIFT
        training_data = read_window(
            dataset, start, TRAINING_HORIZON - SANITY_LEAVEOUT_GAP
        )
        validation_data = read_window(
            dataset, start + TRAINING_HORIZON, VALIDATION_HORIZON
        )
        estimator = KernelDensity(bandwidth=BASIC_BANDWIDTH).fit(training_data)
        surprise_scores = estimator.score_samples(validation_data)
        _surprise_scores.append(surprise_scores)
        # _within_variances.append(np.var(surprise_scores))
        print(f"Trained {step + 1}/{steps} steps")
    # print([np.quantile(-np.concatenate(_surprise_scores), 0.025 * livello) for livello in range(1, 41)])
    chosen_threshold = np.quantile(-np.concatenate(_surprise_scores), 0.95) 
    print("Chosen surprise score threshold:", chosen_threshold)
    # print("Within variances:")
    # print(_within_variances)
    # print("Total variance:")
    # print(np.var(np.concatenate(_surprise_scores)))
    # print("Between variance:")
    # print(np.var(np.concatenate(_surprise_scores))-np.mean(_within_variances))
    # print("Total within variance:")
    # print(np.mean(_within_variances))
    final_training_data = read_window(
        dataset, test_start, TESTING_HORIZON - SANITY_LEAVEOUT_GAP
    )
    test_dataset = read_window(dataset, test_start + TESTING_HORIZON, TEST_LEAVEOUT_SIZE - TESTING_HORIZON)
    final_estimator = KernelDensity(bandwidth=BASIC_BANDWIDTH).fit(final_training_data)
    final_surprises = -final_estimator.score_samples(test_dataset)
    # print([np.quantile(final_surprises, 0.025 * livello) for livello in range(1, 41)])
    higher_than_threshold = final_surprises > chosen_threshold
    raised_failures = np.sum(higher_than_threshold)
    print(f"The model as-fitted raises {raised_failures} failures on the test dataset, a failure rate of {raised_failures/len(final_surprises)}.")

    # The commented-out parts pertaining to within-fold variances and quantiles are irrelevant for the current attempt 
    # at training, and were simply added along the way for debugging and for auditing the code logic.
