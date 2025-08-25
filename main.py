# --- Ensembling support: average predictions from multiple models ---
# This script is a main entry point to download Powerball data.
# It assumes the "Powerball Data Loader" script is saved as a file
# in the same directory, for example, 'powerball_loader.py'.
# You can adjust the import statement based on your file name.
# For this example, we'll assume the file is named 'powerball_loader'.

from data.data_handler import DataHandler
import pandas as pd
import numpy as np
from models.lstm_model import LSTMModel
from models.rnn_model import RNNModel

from util.data_utils import analyze_value_ranges_per_ball
from util.model_utils import run_full_workflow
import config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show info and above
print("I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.")
DATAGOV_API_URL = 'https://data.ny.gov/resource/d6yy-54nr.json'

def main():
    # Diagnostics moved inside data-loaded block
    """
    Main function to orchestrate the data download process.
    """
    print("Starting the Powerball data download process...")
    # Call the function from the imported module to get the DataFrame.
    datagov_df = DataHandler.fetch_data_from_datagov(DATAGOV_API_URL)
    kaggle_df = DataHandler.load_data_from_kaggle(config.KAGGLE_CSV_FILE)
    # Check if the function returned a valid DataFrame.
    if not datagov_df.empty or not kaggle_df.empty:
        print("\nSuccessfully loaded the Powerball data into a DataFrame.")
        print(f"Data includes {kaggle_df.shape[0]} draws and has {kaggle_df.shape[1]} columns.")
        final_df = DataHandler.combine_and_clean_data(datagov_df, kaggle_df)
        # Save the DataFrame to a Parquet file.
        DataHandler.save_to_file(final_df)

        # --- ANALYSIS: Most Likely Value Ranges Per Ball ---
        print("\n--- Most Likely Value Ranges Per Ball (Full Dataset) ---")
        analyze_value_ranges_per_ball(final_df)

        # Split the DataFrame into training and testing sets.
        # --- Fixed test set and true labels for all rounds ---
        train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
        print(f"\nData split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")
        look_back_window = config.LOOK_BACK_WINDOW
        X_train, y_train = DataHandler.prepare_data_for_lstm(train_df, look_back=look_back_window)
        X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
        print(f"Prepared training data shape: {X_train.shape}")
        print(f"Prepared testing data shape: {X_test.shape}")
        if X_train.size == 0 or X_test.size == 0:
            print("\nNot enough data to create sequences. Exiting.")
            return
        # Store fixed test set true labels for all rounds/plots
        y_true_first_five = np.argmax(y_test[0], axis=-1) + 1  # shape (n_samples, 5)
        y_true_sixth = np.argmax(y_test[1], axis=-1) + 1       # shape (n_samples, 1)
        # Hypertuning with KerasTuner
        import keras_tuner as kt
        print("\nStarting KerasTuner hypertuning...")
        input_shape = (X_train.shape[1] // 6, 6)
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1, 6)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1, 6)
        # --- KerasTuner for model hyperparameters only ---
    if not datagov_df.empty or not kaggle_df.empty:
        final_df = DataHandler.combine_and_clean_data(datagov_df, kaggle_df)
        # Save the DataFrame to a Parquet file.
        DataHandler.save_to_file(final_df)

        # --- ANALYSIS: Most Likely Value Ranges Per Ball ---
        print("\n--- Most Likely Value Ranges Per Ball (Full Dataset) ---")
        analyze_value_ranges_per_ball(final_df)

        # --- Meta-parameter optimization for meta-hyperparameters (PSO or Bayesian) ---
        var_names = [
            "LABEL_SMOOTHING",
            "UNIFORM_MIX_PROB",
            "TEMP_MIN",
            "TEMP_MAX",
            "EARLY_STOPPING_PATIENCE",
            "OVERCOUNT_PENALTY_WEIGHT",
            "ENTROPY_PENALTY_WEIGHT",
            "LGBM_NUM_LEAVES",
            "LGBM_LEARNING_RATE",
            "LGBM_MAX_DEPTH"
        ]
        bounds = [
            (0.0, 0.3),      # LABEL_SMOOTHING
            (0.0, 0.3),      # UNIFORM_MIX_PROB
            (0.5, 1.5),      # TEMP_MIN
            (1.5, 2.5),      # TEMP_MAX
            (1, 10),         # EARLY_STOPPING_PATIENCE
            (0.0, 1.0),      # OVERCOUNT_PENALTY_WEIGHT
            (0.0, 1.0),      # ENTROPY_PENALTY_WEIGHT
            (7, 127),        # LGBM_NUM_LEAVES
            (0.01, 0.3),     # LGBM_LEARNING_RATE
            (3, 12)          # LGBM_MAX_DEPTH
        ]
        if getattr(config, 'META_OPT_METHOD', 'pso').lower() == 'bayesian':
            from bayesian_opt import bayesian_optimize
            best = bayesian_optimize(var_names, bounds, final_df, n_trials=getattr(config, 'PSO_ITER', 10))
            print("Best meta-hyperparameters (Bayesian):", dict(zip(var_names, best)))
        else:
            from particle_swarm import particle_swarm_optimize
            best = particle_swarm_optimize(var_names, bounds, final_df, n_particles=config.PSO_PARTICLES, n_iter=config.PSO_ITER)
            print("Best meta-hyperparameters (PSO):", dict(zip(var_names, best)))

        # Set config to best found by optimizer before running KerasTuner/model search
        for i, name in enumerate(var_names):
            setattr(config, name, best[i])

        # --- Multi-round iterative stacking automation ---
        num_rounds = getattr(config, 'ITERATIVE_STACKING_ROUNDS', 1) if getattr(config, 'ITERATIVE_STACKING', False) else 1
        import json
        from util.plot_utils import plot_multi_round_ball_distributions
        rounds_first_five = []
        rounds_sixth = []
        round_labels = []
        # Load previous predictions if available
        prev_pred_first_five = None
        prev_pred_sixth = None
        if os.path.exists('results_predictions.json'):
            try:
                with open('results_predictions.json', 'r') as f:
                    prev_results = json.load(f)
                prev_pred_first_five = np.array(prev_results.get('first_five_pred_numbers'))
                prev_pred_sixth = np.array(prev_results.get('sixth_pred_number'))
            except Exception as e:
                print(f"[DEBUG] Could not load previous predictions for plotting: {e}")
        # Run all rounds and collect predictions
        pseudo_train_df = None
        PSEUDO_CONFIDENCE_THRESHOLD = 0.9
        PSEUDO_MAX_SAMPLES = 100  # Limit per round
    for round_idx in range(num_rounds):
            print(f"\n[ITERATIVE STACKING] === Round {round_idx+1} of {num_rounds} ===")
            try:
                # Pseudo-labeling: for rounds > 0, add high-confidence test predictions as pseudo-labeled data
                if round_idx > 0 and pseudo_train_df is not None:
                    with open('results_predictions.json', 'r') as f:
                        results = json.load(f)
                    pseudo_test_df = pseudo_train_df['test_df']
                    first_five_pred = np.array(results['first_five_pred_numbers'])
                    sixth_pred = np.array(results['sixth_pred_number'])
                    # Softmax probabilities (if available)
                    first_five_softmax = np.array(results.get('first_five_pred_softmax')) if 'first_five_pred_softmax' in results else None
                    sixth_softmax = np.array(results.get('sixth_pred_softmax')) if 'sixth_pred_softmax' in results else None
                    # Ensure all arrays are the same length
                    n = min(len(pseudo_test_df), len(first_five_pred), len(sixth_pred))
                    if first_five_softmax is not None:
                        n = min(n, len(first_five_softmax))
                    if sixth_softmax is not None:
                        n = min(n, len(sixth_softmax))
                    pseudo_labels = []
                    pseudo_indices = []
                    entropies_first = []
                    entropies_sixth = []
                    for i in range(n):
                        # Softmax-based confidence filtering and entropy monitoring
                        accept = True
                        if first_five_softmax is not None and sixth_softmax is not None:
                            conf_first = np.max(first_five_softmax[i], axis=1)  # shape (5,)
                            conf_sixth = np.max(sixth_softmax[i], axis=1)      # shape (1,)
                            entropy_first = -np.sum(first_five_softmax[i] * np.log(first_five_softmax[i] + 1e-8), axis=1)  # (5,)
                            entropy_sixth = -np.sum(sixth_softmax[i] * np.log(sixth_softmax[i] + 1e-8), axis=1)  # (1,)
                            entropies_first.append(entropy_first)
                            entropies_sixth.append(entropy_sixth)
                            if not (np.all(conf_first > PSEUDO_CONFIDENCE_THRESHOLD) and np.all(conf_sixth > PSEUDO_CONFIDENCE_THRESHOLD)):
                                accept = False
                            # Reject if any entropy is below threshold
                            if np.any(entropy_first < getattr(config, 'PSEUDO_MIN_ENTROPY', 2.5)) or np.any(entropy_sixth < getattr(config, 'PSEUDO_MIN_ENTROPY', 2.5)):
                                accept = False
                        if accept:
                            pseudo_indices.append(i)
                    # Monitor entropy statistics
                    if entropies_first:
                        entropies_first = np.stack(entropies_first)
                        entropies_sixth = np.stack(entropies_sixth)
                        print(f"[Pseudo-Labeling] Entropy stats (first five): min {entropies_first.min():.3f}, mean {entropies_first.mean():.3f}, max {entropies_first.max():.3f}")
                        print(f"[Pseudo-Labeling] Entropy stats (sixth): min {entropies_sixth.min():.3f}, mean {entropies_sixth.mean():.3f}, max {entropies_sixth.max():.3f}")
                    # Shuffle for diversity and limit
                    np.random.shuffle(pseudo_indices)
                    pseudo_indices = pseudo_indices[:min(PSEUDO_MAX_SAMPLES, n, len(pseudo_indices))]
                    for i in pseudo_indices:
                        balls = first_five_pred[i].tolist() + sixth_pred[i].tolist()
                        pseudo_labels.append(' '.join(str(int(b)) for b in balls))
                    pseudo_df = pseudo_test_df.iloc[pseudo_indices].copy()
                    pseudo_df['Winning Numbers'] = pseudo_labels
                    # Monitor distribution
                    all_pseudo_numbers = np.concatenate([first_five_pred[pseudo_indices], sixth_pred[pseudo_indices]], axis=1).flatten()
                    unique, counts = np.unique(all_pseudo_numbers, return_counts=True)
                    print(f"[Pseudo-Labeling] Distribution of pseudo-labeled numbers: {dict(zip(unique, counts))}")
                    # Append to original training data
                    train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
                    print(f"[Pseudo-Labeling] Added {len(pseudo_df)} pseudo-labeled samples to training data.")
                # Save test_df for next round's pseudo-labeling
                pseudo_train_df = {'test_df': test_df.copy()} if 'test_df' in locals() else None
                run_full_workflow(final_df, config)
                print(f"[DEBUG] run_full_workflow completed for round {round_idx+1}.")
                # After each round, load predictions from results_predictions.json
                with open('results_predictions.json', 'r') as f:
                    results = json.load(f)
                rounds_first_five.append(np.array(results['first_five_pred_numbers']))
                rounds_sixth.append(np.array(results['sixth_pred_number']))
                round_labels.append(f'Round {round_idx+1}')
                # Always use the fixed test set true labels for plotting
                y_first_five_true_numbers = y_true_first_five
                y_sixth_true_number = y_true_sixth
            except Exception as e:
                print(f"[DEBUG] Exception in run_full_workflow (round {round_idx+1}): {e}")
    print(f"\n[ITERATIVE STACKING] Completed {num_rounds} rounds.")
    # Plot all rounds' predictions and previous predictions
    if rounds_first_five:
        plot_multi_round_ball_distributions(
            y_true=y_first_five_true_numbers,
            rounds_pred_list=rounds_first_five,
            prev_pred=prev_pred_first_five,
            num_balls=5,
            n_classes=69,
            title_prefix='Ball',
            round_labels=round_labels,
            prev_label='Previous'
        )
    if rounds_sixth:
        from util.plot_utils import plot_multi_round_powerball_distribution
        plot_multi_round_powerball_distribution(
            y_true=y_sixth_true_number,
            rounds_pred_list=rounds_sixth,
            prev_pred=prev_pred_sixth,
            n_classes=26,
            title='Powerball (6th Ball) Distribution',
            round_labels=round_labels,
            prev_label='Previous'
        )
        # --- Final compiled line plots for std and KL divergence ---
        from util.plot_utils_std import (
            plot_multi_round_true_std,
            plot_multi_round_pred_std,
            plot_multi_round_kl_divergence
        )
        # Only show after all rounds
        plot_multi_round_true_std(
            y_true=y_first_five_true_numbers,
            rounds_pred_list=rounds_first_five,
            prev_pred=prev_pred_first_five,
            num_balls=5,
            round_labels=round_labels,
            prev_label='Previous'
        )
        plot_multi_round_pred_std(
            y_true=y_first_five_true_numbers,
            rounds_pred_list=rounds_first_five,
            prev_pred=prev_pred_first_five,
            num_balls=5,
            round_labels=round_labels,
            prev_label='Previous'
        )
        plot_multi_round_kl_divergence(
            y_true=y_first_five_true_numbers,
            rounds_pred_list=rounds_first_five,
            prev_pred=prev_pred_first_five,
            num_balls=5,
            n_classes=69,
            round_labels=round_labels,
            prev_label='Previous'
        )
    else:
        print("\nFailed to download or load the data. Please check the internet connection or the source URL.")
        
if __name__ == "__main__":
    main()
