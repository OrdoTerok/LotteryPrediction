def run_keras_tuner_with_current_config(final_df, config):
    """
    Runs KerasTuner search for the current config (meta-parameters set by PSO) and returns the best validation loss.
    Expands search space for model/training hyperparameters.
    """
    train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
    look_back_window = config.LOOK_BACK_WINDOW
    X_train, y_train = DataHandler.prepare_data_for_lstm(train_df, look_back=look_back_window)
    X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
    if X_train.size == 0 or X_test.size == 0:
        return float('inf')  # Penalize invalid splits

    input_shape = (X_train.shape[1] // 6, 6)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1, 6)
    def build_lstm_hp(hp):
        units = hp.Choice('lstm_units', [32, 64, 128, 256])
        num_layers = hp.Choice('lstm_layers', [1, 2, 3])
        dropout = hp.Float('lstm_dropout', 0.0, 0.5, step=0.1)
        activation = hp.Choice('activation', ['tanh', 'relu', 'selu'])
        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'nadam'])
        learning_rate = hp.Choice('learning_rate', [1e-3, 5e-4, 1e-4, 5e-5])
        return LSTMModel.build_lstm_model(
            hp, input_shape,
            units=units, num_layers=num_layers, dropout=dropout,
            optimizer=optimizer, learning_rate=learning_rate,
            activation=activation
        )
    tuner = kt.RandomSearch(
        build_lstm_hp,
        objective='val_loss',
        max_trials=getattr(config, 'KERAS_TUNER_MAX_TRIALS', config.TUNER_MAX_TRIALS),
        executions_per_trial=getattr(config, 'KERAS_TUNER_EXECUTIONS_PER_TRIAL', config.TUNER_EXECUTIONS_PER_TRIAL),
        directory=config.TUNER_DIRECTORY,
        project_name='pso_lstm'
    )
    tuner.search(
        X_train_reshaped, y_train,
        epochs=config.EPOCHS_TUNER,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        verbose=0
    )
    best_hp = tuner.get_best_hyperparameters(1)[0]
    try:
        best_trial = tuner.get_best_trials(1)[0]
    except AttributeError:
        best_trial = tuner.oracle.get_best_trials(1)[0]
    best_val_loss = best_trial.metrics.get_best_value('val_loss')
    return best_val_loss
import numpy as np
from models.lstm_model import LSTMModel
from models.rnn_model import RNNModel
from util.metrics import smooth_labels, mix_uniform, kl_to_uniform, kl_divergence
from util.plot_utils import plot_ball_distributions, plot_powerball_distribution
from data.data_handler import DataHandler
import keras_tuner as kt
import scipy.special
from scipy.stats import entropy

def ensemble_predict(models, X):
    preds_first = []
    preds_sixth = []
    for m in models:
        pf, ps = m.predict(X, verbose=0)
        preds_first.append(pf)
        preds_sixth.append(ps)
    mean_first = np.mean(preds_first, axis=0)
    mean_sixth = np.mean(preds_sixth, axis=0)
    return mean_first, mean_sixth

def run_full_workflow(final_df, config):
    """
    Full workflow: trains, tunes, evaluates, and plots. Saves predictions and metrics for diagnostics.
    """
    # Split the DataFrame into training and testing sets.
    # ...existing code...
    # ...existing code...
    # --- Save predictions and metrics for diagnostics (at the end, after all variables are defined) ---
    try:
        first_five_pred_numbers
        sixth_pred_number
        y_first_five_true_numbers
        y_sixth_true_number
        at_least_one_first_five
        all_first_five
        powerball_match
        all_six_match
        best_entropy
        best_kl_uniform
        best_temp
    except NameError as e:
        print(f"[DEBUG] Skipping results/output because variable is missing: {e}")
        return
    import json
    results = {
        'first_five_pred_numbers': first_five_pred_numbers.tolist(),
        'sixth_pred_number': sixth_pred_number.tolist(),
        'y_first_five_true_numbers': y_first_five_true_numbers.tolist(),
        'y_sixth_true_number': y_sixth_true_number.tolist(),
        'at_least_one_first_five': float(at_least_one_first_five),
        'all_first_five': float(all_first_five),
        'powerball_match': float(powerball_match),
        'all_six_match': float(all_six_match),
        'entropy': float(best_entropy),
        'kl_uniform': float(best_kl_uniform),
        'temperature': float(best_temp)
    }
    with open('results_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    # --- Brier score for probabilistic accuracy ---
    def brier_score(y_true, y_prob):
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        return np.mean(np.sum((y_prob - y_true) ** 2, axis=-1))
    brier_first = brier_score(y_test[0], first_five_pred_temp)
    brier_sixth = brier_score(y_test[1], sixth_pred_temp)
    print(f"Brier score (first five): {brier_first:.6f}")
    print(f"Brier score (sixth): {brier_sixth:.6f}")
    # --- Calibration curve plot (for first ball as example) ---
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(
        y_test[0][:, 0, :].argmax(axis=-1) == first_five_pred_temp[:, 0, :].argmax(axis=-1),
        first_five_pred_temp[:, 0, :].max(axis=-1),
        n_bins=10
    )
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('Calibration Curve (First Ball)')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.savefig('calibration_curve_first_ball.png')
    plt.close()
    print("[DEBUG] Entered run_full_workflow")
    print("[DEBUG] After data preparation, before KerasTuner.")
    train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
    print(f"\nData split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")
    look_back_window = config.LOOK_BACK_WINDOW
    X_train, y_train = DataHandler.prepare_data_for_lstm(train_df, look_back=look_back_window)
    X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
    print(f"Prepared training data shape: {X_train.shape}")
    print(f"Prepared testing data shape: {X_test.shape}")
    if X_train.size == 0 or X_test.size == 0:
        print("\n[DEBUG] Not enough data to create sequences. Exiting run_full_workflow early.")
        return
    print("[DEBUG] Data prepared. Proceeding to KerasTuner hypertuning...")
    input_shape = (X_train.shape[1] // 6, 6)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1, 6)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1, 6)
    # LSTM Tuning
    tuner_lstm = kt.RandomSearch(
        lambda hp: LSTMModel.build_lstm_model(hp, input_shape),
        objective='val_loss',
        max_trials=config.TUNER_MAX_TRIALS,
        executions_per_trial=config.TUNER_EXECUTIONS_PER_TRIAL,
        directory=config.TUNER_DIRECTORY,
        project_name=config.TUNER_PROJECT_NAME+'_lstm'
    )
    try:
        tuner_lstm.search(
            X_train_reshaped, y_train,
            epochs=config.EPOCHS_TUNER,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=1
        )
        best_hp_lstm = tuner_lstm.get_best_hyperparameters(1)[0]
        print(f"\nBest LSTM model hyperparameters: {best_hp_lstm.values}")
        print("[DEBUG] Completed LSTM KerasTuner search.")
    except Exception as e:
        print(f"[DEBUG] Exception during LSTM KerasTuner search: {e}")
        return
    # RNN Tuning
    tuner_rnn = kt.RandomSearch(
        lambda hp: RNNModel.build_rnn_model(hp, input_shape),
        objective='val_loss',
        max_trials=config.TUNER_MAX_TRIALS,
        executions_per_trial=config.TUNER_EXECUTIONS_PER_TRIAL,
        directory=config.TUNER_DIRECTORY,
        project_name=config.TUNER_PROJECT_NAME+'_rnn'
    )
    try:
        tuner_rnn.search(
            X_train_reshaped, y_train,
            epochs=config.EPOCHS_TUNER,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=1
        )
        best_hp_rnn = tuner_rnn.get_best_hyperparameters(1)[0]
        print(f"\nBest RNN model hyperparameters: {best_hp_rnn.values}")
        print("[DEBUG] Completed RNN KerasTuner search.")
    except Exception as e:
        print(f"[DEBUG] Exception during RNN KerasTuner search: {e}")
        return
    # Label Smoothing
    y_train_smoothed = [smooth_labels(y_train[0], config.LABEL_SMOOTHING), smooth_labels(y_train[1], config.LABEL_SMOOTHING)]
    y_train_smoothed = [mix_uniform(y_train_smoothed[0], config.UNIFORM_MIX_PROB), mix_uniform(y_train_smoothed[1], config.UNIFORM_MIX_PROB)]
    # Build and train best LSTM model
    best_lstm_model = LSTMModel.build_lstm_model(
        best_hp_lstm, input_shape,
        use_custom_loss=True,
        force_low_units=config.FORCE_LOW_UNITS,
        force_simple=config.FORCE_SIMPLE
    )
    try:
        print("[DEBUG] Training best LSTM model...")
        best_lstm_model.fit(
            X_train_reshaped, y_train_smoothed,
            epochs=config.EPOCHS_FINAL,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=0
        )
        print("[DEBUG] Completed LSTM model training.")
    except Exception as e:
        print(f"[DEBUG] Exception during LSTM model training: {e}")
        return
    # Build and train best RNN model
    best_rnn_model = RNNModel.build_rnn_model(
        best_hp_rnn, input_shape
    )
    try:
        print("[DEBUG] Training best RNN model...")
        best_rnn_model.fit(
            X_train_reshaped, y_train_smoothed,
            epochs=config.EPOCHS_FINAL,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=0
        )
        print("[DEBUG] Completed RNN model training.")
    except Exception as e:
        print(f"[DEBUG] Exception during RNN model training: {e}")
        return
    # Ensembling
    try:
        print("[DEBUG] Making ensemble predictions...")
        models = [best_lstm_model, best_rnn_model]
        first_five_pred, sixth_pred = ensemble_predict(models, X_test_reshaped)
        print("[DEBUG] Completed ensemble predictions.")
    except Exception as e:
        print(f"[DEBUG] Exception during ensemble prediction: {e}")
        return
    # Temperature scaling
    print("[DEBUG] Starting temperature grid search...")
    print("[DEBUG] Finished all predictions and evaluation. Exiting run_full_workflow normally.")
    def apply_temperature_softmax(probs, temperature):
        logits = np.log(np.clip(probs, 1e-12, 1.0))
        logits /= temperature
        orig_shape = logits.shape
        logits_flat = logits.reshape(-1, logits.shape[-1])
        softmax_flat = scipy.special.softmax(logits_flat, axis=-1)
        return softmax_flat.reshape(orig_shape)
    best_temp = 1.0
    best_kl_uniform = float('inf')
    best_entropy = -float('inf')
    for temp in np.arange(config.TEMP_MIN, config.TEMP_MAX + config.TEMP_STEP, config.TEMP_STEP):
        first_five_pred_temp = apply_temperature_softmax(first_five_pred, temp)
        sixth_pred_temp = apply_temperature_softmax(sixth_pred, temp)
        kl_uniforms = []
        entropies = []
        for i in range(5):
            kl_uniforms.append(kl_to_uniform(first_five_pred_temp[:, i, :]))
            entropies.append(np.mean(entropy(first_five_pred_temp[:, i, :].T)))
        kl_uniforms.append(kl_to_uniform(sixth_pred_temp[:, 0, :]))
        entropies.append(np.mean(entropy(sixth_pred_temp[:, 0, :].T)))
        mean_kl_uniform = np.mean(kl_uniforms)
        mean_entropy = np.mean(entropies)
        if mean_kl_uniform < best_kl_uniform:
            best_kl_uniform = mean_kl_uniform
            best_entropy = mean_entropy
            best_temp = temp
    print(f"\nBest temperature found by grid search (min KL to uniform): {best_temp}")
    print(f"KL to uniform at best temperature: {best_kl_uniform:.6f}")
    print(f"Entropy at best temperature: {best_entropy:.6f}")
    # Final predictions
    first_five_pred_temp = apply_temperature_softmax(first_five_pred, best_temp)
    sixth_pred_temp = apply_temperature_softmax(sixth_pred, best_temp)
    first_five_pred_numbers = np.argmax(first_five_pred_temp, axis=-1) + 1
    sixth_pred_number = np.argmax(sixth_pred_temp, axis=-1) + 1
    # True values
    y_first_five_true_numbers = np.argmax(y_test[0], axis=-1) + 1
    y_sixth_true_number = np.argmax(y_test[1], axis=-1) + 1
    # Evaluation
    num_samples = first_five_pred_numbers.shape[0]
    first_five_matches = (first_five_pred_numbers == y_first_five_true_numbers)
    sixth_matches = (sixth_pred_number == y_sixth_true_number)
    num_first_five_matched = np.sum(first_five_matches, axis=1)
    at_least_one_first_five = np.mean(num_first_five_matched >= 1)
    all_first_five = np.mean(num_first_five_matched == 5)
    powerball_match = np.mean(sixth_matches)
    all_six_match = np.mean((num_first_five_matched == 5) & (sixth_matches[:, 0]))
    print(f"At least one of first five matched: {at_least_one_first_five:.4f}")
    print(f"All first five matched: {all_first_five:.6f}")
    print(f"Powerball matched: {powerball_match:.4f}")
    print(f"All six matched: {all_six_match:.8f}")
    # Std & KL
    for i in range(5):
        true_std = np.std(y_first_five_true_numbers[:, i])
        pred_std = np.std(first_five_pred_numbers[:, i])
        true_hist = np.bincount(y_first_five_true_numbers[:, i]-1, minlength=69)
        pred_hist = np.bincount(first_five_pred_numbers[:, i]-1, minlength=69)
        true_dist = true_hist / np.sum(true_hist)
        pred_dist = pred_hist / np.sum(pred_hist)
        kl = kl_divergence(true_dist, pred_dist)
        print(f"Ball {i+1} True Std: {true_std:.2f}, Predicted Std: {pred_std:.2f}, KL Divergence: {kl:.6f}")
    true_std_6 = np.std(y_sixth_true_number[:, 0])
    pred_std_6 = np.std(sixth_pred_number[:, 0])
    true_hist_6 = np.bincount(y_sixth_true_number[:, 0]-1, minlength=26)
    pred_hist_6 = np.bincount(sixth_pred_number[:, 0]-1, minlength=26)
    true_dist_6 = true_hist_6 / np.sum(true_hist_6)
    pred_dist_6 = pred_hist_6 / np.sum(pred_hist_6)
    kl_6 = kl_divergence(true_dist_6, pred_dist_6)
    print(f"Powerball (6th Ball) True Std: {true_std_6:.2f}, Predicted Std: {pred_std_6:.2f}, KL Divergence: {kl_6:.6f}")
    # Plots
    plot_ball_distributions(y_first_five_true_numbers, first_five_pred_numbers, num_balls=5, n_classes=69, title_prefix='Ball')
    plot_powerball_distribution(y_sixth_true_number, sixth_pred_number, n_classes=26)
