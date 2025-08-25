"""
Model utility functions for LotteryPrediction.
Handles model training, hyperparameter/meta-parameter optimization (PSO, Bayesian),
cross-validation, evaluation, and advanced ensembling (average, weighted, stacking).
"""

def run_keras_tuner_with_current_config(final_df, config):
    """
    Runs KerasTuner search for the current config (meta-parameters set by PSO/Bayesian) and returns the best validation loss.
    Supports k-fold cross-validation if config.CV_FOLDS > 1, otherwise uses standard train/test split.
    Expands search space for model/training hyperparameters.
    """
    from sklearn.model_selection import KFold
    cv_folds = getattr(config, 'CV_FOLDS', 1)
    if cv_folds <= 1:
        # Standard train/test split
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
    else:
        # K-fold cross-validation
        look_back_window = config.LOOK_BACK_WINDOW
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=getattr(config, 'RANDOM_SEED', 42))
        val_losses = []
        df_indices = np.arange(final_df.shape[0])
        for fold, (train_idx, test_idx) in enumerate(kf.split(df_indices)):
            train_df = final_df.iloc[train_idx]
            test_df = final_df.iloc[test_idx]
            X_train, y_train = DataHandler.prepare_data_for_lstm(train_df, look_back=look_back_window)
            X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
            if X_train.size == 0 or X_test.size == 0:
                continue  # Skip this fold
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
                project_name=f'pso_lstm_fold{fold}'
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
            val_losses.append(best_val_loss)
        if len(val_losses) == 0:
            return float('inf')
        return float(np.mean(val_losses))
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
    """
    Ensemble predictions from multiple models using the strategy specified in config.ENSEMBLE_STRATEGY.
    Supported strategies:
      - 'average': simple mean of model predictions
      - 'weighted': weighted average (equal weights by default)
      - 'stacking': meta-learner (logistic regression) stacking
    """
    import config
    preds_first = []
    preds_sixth = []
    for m in models:
        pf, ps = m.predict(X, verbose=0)
        preds_first.append(pf)
        preds_sixth.append(ps)
    strategy = getattr(config, 'ENSEMBLE_STRATEGY', 'average').lower()
    if strategy == 'average':
        mean_first = np.mean(preds_first, axis=0)
        mean_sixth = np.mean(preds_sixth, axis=0)
        return mean_first, mean_sixth
    elif strategy == 'weighted':
        # Example: equal weights, can be extended to learn weights
        weights = np.ones(len(models)) / len(models)
        weighted_first = np.tensordot(weights, np.array(preds_first), axes=1)
        weighted_sixth = np.tensordot(weights, np.array(preds_sixth), axes=1)
        return weighted_first, weighted_sixth
    elif strategy == 'stacking':
        # Simple stacking: train a meta-learner (logistic regression) on model outputs
        from sklearn.linear_model import LogisticRegression
        # Stack for first five balls
        n_samples, n_balls, n_classes = preds_first[0].shape
        stacked_first = np.zeros((n_samples, n_balls, n_classes))
        for b in range(n_balls):
            X_stack = np.stack([pf[:, b, :] for pf in preds_first], axis=-1).reshape(n_samples * n_classes, len(models))
            y_stack = np.argmax(np.mean(preds_first, axis=0)[:, b, :], axis=-1).repeat(n_classes)
            meta = LogisticRegression(max_iter=1000)
            try:
                meta.fit(X_stack, y_stack)
                preds = meta.predict_proba(X_stack)
                stacked_first[:, b, :] = preds.reshape(n_samples, n_classes)
            except Exception:
                stacked_first[:, b, :] = np.mean([pf[:, b, :] for pf in preds_first], axis=0)
        # Stack for sixth ball
        n_samples, n_balls, n_classes = preds_sixth[0].shape
        stacked_sixth = np.zeros((n_samples, n_balls, n_classes))
        for b in range(n_balls):
            X_stack = np.stack([ps[:, b, :] for ps in preds_sixth], axis=-1).reshape(n_samples * n_classes, len(models))
            y_stack = np.argmax(np.mean(preds_sixth, axis=0)[:, b, :], axis=-1).repeat(n_classes)
            meta = LogisticRegression(max_iter=1000)
            try:
                meta.fit(X_stack, y_stack)
                preds = meta.predict_proba(X_stack)
                stacked_sixth[:, b, :] = preds.reshape(n_samples, n_classes)
            except Exception:
                stacked_sixth[:, b, :] = np.mean([ps[:, b, :] for ps in preds_sixth], axis=0)
        return stacked_first, stacked_sixth
    else:
        # Fallback to average
        mean_first = np.mean(preds_first, axis=0)
        mean_sixth = np.mean(preds_sixth, axis=0)
        return mean_first, mean_sixth

def run_full_workflow(final_df, config):
    """
    Full workflow: trains, tunes, evaluates, and plots.
    - Meta-parameter optimization: PSO or Bayesian (config.META_OPT_METHOD)
    - Cross-validation: k-fold if config.CV_FOLDS > 1
    - Ensembling: average, weighted, or stacking (config.ENSEMBLE_STRATEGY)
    Saves predictions and metrics for diagnostics.
    """
    print("[DEBUG] Top of run_full_workflow. final_df shape:", getattr(final_df, 'shape', 'N/A'), "type:", type(final_df))
    # Split the DataFrame into training and testing sets.
    # ...existing code...
    # ...existing code...
    print("[DEBUG] Entered run_full_workflow")
    print("[DEBUG] Starting data split...")
    train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
    print(f"[DEBUG] Data split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")
    print(f"[DEBUG] train_df shape: {getattr(train_df, 'shape', 'N/A')}, test_df shape: {getattr(test_df, 'shape', 'N/A')}")
    look_back_window = config.LOOK_BACK_WINDOW
    print(f"[DEBUG] Preparing training data for LSTM with look_back={look_back_window}...")
    X_train, y_train = DataHandler.prepare_data_for_lstm(train_df, look_back=look_back_window)
    print(f"[DEBUG] Preparing testing data for LSTM with look_back={look_back_window}...")
    X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
    print(f"[DEBUG] Prepared training data shape: {X_train.shape}, type: {type(X_train)}")
    print(f"[DEBUG] Prepared testing data shape: {X_test.shape}, type: {type(X_test)}")
    if X_train.size == 0 or X_test.size == 0:
        print("[DEBUG] Not enough data to create sequences. Exiting run_full_workflow early.")
        return
    print(f"[DEBUG] y_train type: {type(y_train)}, y_test type: {type(y_test)}")
    print("[DEBUG] Data prepared. Proceeding to KerasTuner hypertuning...")
    input_shape = (X_train.shape[1] // 6, 6)
    print(f"[DEBUG] input_shape for models: {input_shape}")
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1, 6)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1, 6)
    print(f"[DEBUG] X_train_reshaped shape: {X_train_reshaped.shape}, X_test_reshaped shape: {X_test_reshaped.shape}")
    print("[DEBUG] Initializing LSTM KerasTuner...")
    tuner_lstm = kt.RandomSearch(
        lambda hp: LSTMModel.build_lstm_model(hp, input_shape),
        objective='val_loss',
        max_trials=config.TUNER_MAX_TRIALS,
        executions_per_trial=config.TUNER_EXECUTIONS_PER_TRIAL,
        directory=config.TUNER_DIRECTORY,
        project_name=config.TUNER_PROJECT_NAME+'_lstm'
    )
    try:
        print("[DEBUG] Starting LSTM KerasTuner search...")
        tuner_lstm.search(
            X_train_reshaped, y_train,
            epochs=config.EPOCHS_TUNER,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=1
        )
        best_hp_lstm = tuner_lstm.get_best_hyperparameters(1)[0]
        print(f"[DEBUG] Best LSTM model hyperparameters: {best_hp_lstm.values}")
        print("[DEBUG] Completed LSTM KerasTuner search.")
    except Exception as e:
        print(f"[DEBUG] Exception during LSTM KerasTuner search: {e}")
        return
    print("[DEBUG] Initializing RNN KerasTuner...")
    tuner_rnn = kt.RandomSearch(
        lambda hp: RNNModel.build_rnn_model(hp, input_shape),
        objective='val_loss',
        max_trials=config.TUNER_MAX_TRIALS,
        executions_per_trial=config.TUNER_EXECUTIONS_PER_TRIAL,
        directory=config.TUNER_DIRECTORY,
        project_name=config.TUNER_PROJECT_NAME+'_rnn'
    )
    try:
        print("[DEBUG] Starting RNN KerasTuner search...")
        tuner_rnn.search(
            X_train_reshaped, y_train,
            epochs=config.EPOCHS_TUNER,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=1
        )
        best_hp_rnn = tuner_rnn.get_best_hyperparameters(1)[0]
        print(f"[DEBUG] Best RNN model hyperparameters: {best_hp_rnn.values}")
        print("[DEBUG] Completed RNN KerasTuner search.")
    except Exception as e:
        print(f"[DEBUG] Exception during RNN KerasTuner search: {e}")
        return
    print("[DEBUG] Starting label smoothing and uniform mixing...")
    y_train_smoothed = [smooth_labels(y_train[0], config.LABEL_SMOOTHING), smooth_labels(y_train[1], config.LABEL_SMOOTHING)]
    y_train_smoothed = [mix_uniform(y_train_smoothed[0], config.UNIFORM_MIX_PROB), mix_uniform(y_train_smoothed[1], config.UNIFORM_MIX_PROB)]
    print(f"[DEBUG] y_train_smoothed[0] shape: {y_train_smoothed[0].shape}, y_train_smoothed[1] shape: {y_train_smoothed[1].shape}")
    print("[DEBUG] Building best LSTM model...")
    best_lstm_model = LSTMModel.build_lstm_model(
        best_hp_lstm, input_shape,
        use_custom_loss=True,
        force_low_units=config.FORCE_LOW_UNITS,
        force_simple=config.FORCE_SIMPLE
    )
    try:
        print("[DEBUG] Training best LSTM model...")
        print(f"[DEBUG] X_train_reshaped shape: {X_train_reshaped.shape}, y_train_smoothed shapes: {[y.shape for y in y_train_smoothed]}")
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
    print("[DEBUG] Building best RNN model...")
    best_rnn_model = RNNModel.build_rnn_model(
        best_hp_rnn, input_shape
    )
    try:
        print("[DEBUG] Training best RNN model...")
        print(f"[DEBUG] X_train_reshaped shape: {X_train_reshaped.shape}, y_train_smoothed shapes: {[y.shape for y in y_train_smoothed]}")
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
    print("[DEBUG] Starting ensembling...")
    try:
        print("[DEBUG] Making ensemble predictions...")
        print(f"[DEBUG] X_test_reshaped shape: {X_test_reshaped.shape}")
        models = [best_lstm_model, best_rnn_model]
        first_five_pred, sixth_pred = ensemble_predict(models, X_test_reshaped)
        print(f"[DEBUG] first_five_pred shape: {first_five_pred.shape}, sixth_pred shape: {sixth_pred.shape}")
        print("[DEBUG] Completed ensemble predictions.")
    except Exception as e:
        print(f"[DEBUG] Exception during ensemble prediction: {e}")
        return
    print("[DEBUG] Starting temperature grid search...")
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
        if temp == config.TEMP_MIN or temp == config.TEMP_MAX:
            print(f"[DEBUG] Temp {temp}: first_five_pred_temp shape: {first_five_pred_temp.shape}, sixth_pred_temp shape: {sixth_pred_temp.shape}")
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
    print(f"[DEBUG] Finished temperature grid search. Best temp: {best_temp}")
    print("[DEBUG] Finished all predictions and evaluation. Exiting run_full_workflow normally.")
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
