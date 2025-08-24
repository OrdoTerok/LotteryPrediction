# --- Ensembling support: average predictions from multiple models ---
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

# To use ensembling:
# 1. Train several models with different random seeds or KerasTuner results.
# 2. Use ensemble_predict([model1, model2, ...], X_test_reshaped) for final predictions.

# --- Suggestions for further features ---
# - Add attention layers (e.g., self-attention or transformer blocks)
# - Use external features (e.g., date, jackpot size, day of week)
# - Try convolutional layers before LSTM (CNN-LSTM)
# - Use Bayesian optimization for hyperparameters
# - Add custom callbacks for advanced early stopping or learning rate schedules
# - Try different data encodings (e.g., ordinal, embeddings)
# - Use Monte Carlo dropout for uncertainty estimation
def run_for_pso():
    # Minimal run: load data, train, predict, return predicted std and KL for fitness
    datagov_df = DataHandler.fetch_data_from_datagov(DATAGOV_API_URL)
    kaggle_df = DataHandler.load_data_from_kaggle(config.KAGGLE_CSV_FILE)
    if not datagov_df.empty or not kaggle_df.empty:
        final_df = DataHandler.combine_and_clean_data(datagov_df, kaggle_df)
        train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
        look_back_window = config.LOOK_BACK_WINDOW
        X_train, y_train = DataHandler.prepare_data_for_lstm(train_df, look_back=look_back_window)
        X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
        if X_train.size == 0 or X_test.size == 0:
            return [0], [float('inf')]
        import keras_tuner as kt
        input_shape = (X_train.shape[1] // 6, 6)
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1, 6)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1, 6)
        tuner = kt.RandomSearch(
            lambda hp: LSTMModel.build_lstm_model(hp, input_shape),
            objective='val_loss',
            max_trials=1,
            executions_per_trial=1,
            directory='hypertune_dir',
            project_name='lstm_lottery_pso'
        )
        tuner.search(
            X_train_reshaped, y_train,
            epochs=2,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=0
        )
        best_hp = tuner.get_best_hyperparameters(1)[0]
        def smooth_labels(y, smoothing=config.LABEL_SMOOTHING):
            y = np.asarray(y, dtype=np.float32)
            n_classes = y.shape[-1]
            return y * (1 - smoothing) + smoothing / n_classes
        def mix_uniform(y, mix_prob=config.UNIFORM_MIX_PROB):
            y = np.asarray(y, dtype=np.float32)
            n_classes = y.shape[-1]
            mask = np.random.rand(*y.shape[:-1]) < mix_prob
            uniform = np.ones_like(y) / n_classes
            y[mask] = uniform[mask]
            return y
        y_train_smoothed = [smooth_labels(y_train[0], config.LABEL_SMOOTHING), smooth_labels(y_train[1], config.LABEL_SMOOTHING)]
        y_train_smoothed = [mix_uniform(y_train_smoothed[0], config.UNIFORM_MIX_PROB), mix_uniform(y_train_smoothed[1], config.UNIFORM_MIX_PROB)]
        from tensorflow.keras.callbacks import EarlyStopping
        best_model = LSTMModel.build_lstm_model(
            best_hp, input_shape,
            use_custom_loss=True,
            force_low_units=config.FORCE_LOW_UNITS,
            force_simple=config.FORCE_SIMPLE
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        best_model.fit(
            X_train_reshaped, y_train_smoothed,
            epochs=4,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=0,
            callbacks=[early_stop]
        )
        first_five_pred, sixth_pred = best_model.predict(X_test_reshaped, verbose=0)
        first_five_pred_numbers = np.argmax(first_five_pred, axis=-1) + 1
        sixth_pred_number = np.argmax(sixth_pred, axis=-1) + 1
        y_first_five_true_numbers = np.argmax(y_test[0], axis=-1) + 1
        y_sixth_true_number = np.argmax(y_test[1], axis=-1) + 1
        from scipy.stats import entropy
        def kl_divergence(p, q):
            p = np.asarray(p, dtype=np.float64)
            q = np.asarray(q, dtype=np.float64)
            p = np.clip(p, 1e-12, 1)
            q = np.clip(q, 1e-12, 1)
            return entropy(p, q)
        stds = []
        kls = []
        accs = []
        entropies = []
        log_losses = []
        # Balls 1-5
        for i in range(5):
            pred_std = np.std(first_five_pred_numbers[:, i])
            pred_hist = np.bincount(first_five_pred_numbers[:, i]-1, minlength=69)
            true_hist = np.bincount(y_first_five_true_numbers[:, i]-1, minlength=69)
            pred_dist = pred_hist / np.sum(pred_hist)
            true_dist = true_hist / np.sum(true_hist)
            kl = kl_divergence(true_dist, pred_dist)
            acc = np.mean(first_five_pred_numbers[:, i] == y_first_five_true_numbers[:, i])
            ent = np.mean(entropy(first_five_pred[:, i, :].T))
            # Log loss (cross-entropy)
            true_onehot = y_test[0][:, i, :]
            pred_probs = first_five_pred[:, i, :]
            log_loss = -np.mean(np.sum(true_onehot * np.log(np.clip(pred_probs, 1e-12, 1)), axis=-1))
            stds.append(pred_std)
            kls.append(kl)
            accs.append(acc)
            entropies.append(ent)
            log_losses.append(log_loss)
        # Ball 6
        pred_std_6 = np.std(sixth_pred_number[:, 0])
        pred_hist_6 = np.bincount(sixth_pred_number[:, 0]-1, minlength=26)
        true_hist_6 = np.bincount(y_sixth_true_number[:, 0]-1, minlength=26)
        pred_dist_6 = pred_hist_6 / np.sum(pred_hist_6)
        true_dist_6 = true_hist_6 / np.sum(true_hist_6)
        kl_6 = kl_divergence(true_dist_6, pred_dist_6)
        acc_6 = np.mean(sixth_pred_number[:, 0] == y_sixth_true_number[:, 0])
        ent_6 = np.mean(entropy(sixth_pred[:, 0, :].T))
        true_onehot_6 = y_test[1][:, 0, :]
        pred_probs_6 = sixth_pred[:, 0, :]
        log_loss_6 = -np.mean(np.sum(true_onehot_6 * np.log(np.clip(pred_probs_6, 1e-12, 1)), axis=-1))
        stds.append(pred_std_6)
        kls.append(kl_6)
        accs.append(acc_6)
        entropies.append(ent_6)
        log_losses.append(log_loss_6)
        # Return all metrics as dict
        return {
            'stds': stds,
            'kls': kls,
            'accs': accs,
            'entropies': entropies,
            'log_losses': log_losses
        }
    else:
        return [0], [float('inf')]
# This script is a main entry point to download Powerball data.
# It assumes the "Powerball Data Loader" script is saved as a file
# in the same directory, for example, 'powerball_loader.py'.
# You can adjust the import statement based on your file name.
# For this example, we'll assume the file is named 'powerball_loader'.

from data.data_handler import DataHandler
import numpy as np
from models.lstm_model import LSTMModel
from typing import Tuple
import config
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
        winning_numbers = final_df['Winning Numbers'].str.split().apply(lambda x: [int(i) for i in x]).values
        balls = list(zip(*winning_numbers))  # balls[0] = all Ball 1, balls[5] = all Powerball
        import numpy as np
        import pandas as pd
        for i in range(5):
            ball_vals = np.array(balls[i])
            print(f"\nBall {i+1}:")
            print(f"  Min: {ball_vals.min()}, Max: {ball_vals.max()}")
            print(f"  25th percentile: {np.percentile(ball_vals, 25):.1f}, 75th percentile: {np.percentile(ball_vals, 75):.1f}")
            vc = pd.Series(ball_vals).value_counts().sort_index()
            top = vc.sort_values(ascending=False).head(10)
            print(f"  Top 10 most frequent numbers: {list(top.index)} (counts: {list(top.values)})")
        # Powerball (6th ball)
        ball_vals = np.array(balls[5])
        print(f"\nPowerball (6th Ball):")
        print(f"  Min: {ball_vals.min()}, Max: {ball_vals.max()}")
        print(f"  25th percentile: {np.percentile(ball_vals, 25):.1f}, 75th percentile: {np.percentile(ball_vals, 75):.1f}")
        vc = pd.Series(ball_vals).value_counts().sort_index()
        top = vc.sort_values(ascending=False).head(10)
        print(f"  Top 10 most frequent numbers: {list(top.index)} (counts: {list(top.values)})")

        # Split the DataFrame into training and testing sets.
        train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
        print(f"\nData split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")
        # The 'look_back' value determines how many previous draws the model
        # will use to predict the next one.
        look_back_window = config.LOOK_BACK_WINDOW
        X_train, y_train = DataHandler.prepare_data_for_lstm(train_df, look_back=look_back_window)
        X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
        print(f"Prepared training data shape: {X_train.shape}")
        print(f"Prepared testing data shape: {X_test.shape}")
        if X_train.size == 0 or X_test.size == 0:
            print("\nNot enough data to create sequences. Exiting.")
            return
        # Hypertuning with KerasTuner
        import keras_tuner as kt
        print("\nStarting KerasTuner hypertuning...")
        input_shape = (X_train.shape[1] // 6, 6)
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1, 6)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1, 6)
        # --- KerasTuner for model hyperparameters only ---
        tuner = kt.RandomSearch(
            lambda hp: LSTMModel.build_lstm_model(hp, input_shape),
            objective='val_loss',
            max_trials=config.TUNER_MAX_TRIALS,
            executions_per_trial=config.TUNER_EXECUTIONS_PER_TRIAL,
            directory=config.TUNER_DIRECTORY,
            project_name=config.TUNER_PROJECT_NAME
        )
        tuner.search(
            X_train_reshaped, y_train,
            epochs=config.EPOCHS_TUNER,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=1
        )
        best_hp = tuner.get_best_hyperparameters(1)[0]
        print(f"\nBest model hyperparameters found by KerasTuner: {best_hp.values}")
        # --- Label Smoothing ---
        def smooth_labels(y, smoothing=config.LABEL_SMOOTHING):
            y = np.asarray(y, dtype=np.float32)
            n_classes = y.shape[-1]
            return y * (1 - smoothing) + smoothing / n_classes

        def mix_uniform(y, mix_prob=config.UNIFORM_MIX_PROB):
            # With probability mix_prob, replace label with uniform
            y = np.asarray(y, dtype=np.float32)
            n_classes = y.shape[-1]
            mask = np.random.rand(*y.shape[:-1]) < mix_prob
            uniform = np.ones_like(y) / n_classes
            y[mask] = uniform[mask]
            return y

        y_train_smoothed = [smooth_labels(y_train[0], config.LABEL_SMOOTHING), smooth_labels(y_train[1], config.LABEL_SMOOTHING)]
        # Mix in uniform targets
        y_train_smoothed = [mix_uniform(y_train_smoothed[0], config.UNIFORM_MIX_PROB), mix_uniform(y_train_smoothed[1], config.UNIFORM_MIX_PROB)]

        # Rebuild and retrain best model on all training data with custom loss
        from tensorflow.keras.callbacks import EarlyStopping, Callback # type: ignore
        # Further reduce model complexity
        best_model = LSTMModel.build_lstm_model(
            best_hp, input_shape,
            use_custom_loss=True,
            force_low_units=config.FORCE_LOW_UNITS,
            force_simple=config.FORCE_SIMPLE
        )

        # Early stopping on validation KL-to-uniform
        class EarlyStoppingKLUniform(Callback):
            def __init__(self, patience=config.EARLY_STOPPING_PATIENCE):
                super().__init__()
                self.patience = patience
                self.best_kl = float('inf')
                self.wait = 0
                self.best_weights = None
            def on_train_begin(self, logs=None):
                # Try to get validation data from model
                self.val_data = None
                if hasattr(self.model, 'validation_data') and self.model.validation_data is not None:
                    self.val_data = self.model.validation_data
                elif hasattr(self.model, '_fit_kwargs') and 'validation_data' in self.model._fit_kwargs:
                    self.val_data = self.model._fit_kwargs['validation_data']
                # fallback: will try to get from logs in on_epoch_end

            def on_epoch_end(self, epoch, logs=None):
                # Try to get validation data from logs if not already set
                if hasattr(self, 'val_data') and self.val_data is not None:
                    val_x = self.val_data[0]
                elif logs is not None and 'val_loss' in logs:
                    # fallback: skip KL check if no val data
                    return
                else:
                    return
                val_pred = self.model.predict(val_x, verbose=0)
                first_five_pred, sixth_pred = val_pred
                def kl_divergence(p, q):
                    p = np.asarray(p, dtype=np.float64)
                    q = np.asarray(q, dtype=np.float64)
                    p = np.clip(p, 1e-12, 1)
                    q = np.clip(q, 1e-12, 1)
                    from scipy.stats import entropy
                    return entropy(p, q)
                def kl_to_uniform(p):
                    n = p.shape[-1]
                    uniform = np.ones(n) / n
                    return np.mean([kl_divergence(pred, uniform) for pred in p])
                kl_uniforms = []
                for i in range(5):
                    kl_uniforms.append(kl_to_uniform(first_five_pred[:, i, :]))
                kl_uniforms.append(kl_to_uniform(sixth_pred[:, 0, :]))
                mean_kl_uniform = np.mean(kl_uniforms)
                if mean_kl_uniform < self.best_kl:
                    self.best_kl = mean_kl_uniform
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        print(f"Early stopping at epoch {epoch+1} (val KL to uniform did not improve)")
                        self.model.stop_training = True
                        if self.best_weights is not None:
                            self.model.set_weights(self.best_weights)

        early_stop_kl = EarlyStoppingKLUniform(patience=config.EARLY_STOPPING_PATIENCE)
        best_model.fit(
            X_train_reshaped, y_train_smoothed,
            epochs=config.EPOCHS_FINAL,
            batch_size=config.BATCH_SIZE,
            validation_split=config.VALIDATION_SPLIT,
            verbose=0,
            callbacks=[early_stop_kl]
        )

        # --- Prediction and Analysis Sequence ---
        import scipy.special
        from scipy.stats import entropy
        def apply_temperature_softmax(probs, temperature):
            logits = np.log(np.clip(probs, 1e-12, 1.0))
            logits /= temperature
            orig_shape = logits.shape
            logits_flat = logits.reshape(-1, logits.shape[-1])
            softmax_flat = scipy.special.softmax(logits_flat, axis=-1)
            return softmax_flat.reshape(orig_shape)

        def kl_divergence(p, q):
            p = np.asarray(p, dtype=np.float64)
            q = np.asarray(q, dtype=np.float64)
            p = np.clip(p, 1e-12, 1)
            q = np.clip(q, 1e-12, 1)
            return entropy(p, q)

        # 1. Predict on test set
        first_five_pred, sixth_pred = best_model.predict(X_test_reshaped)
        y_first_five_true_numbers = np.argmax(y_test[0], axis=-1) + 1
        y_sixth_true_number = np.argmax(y_test[1], axis=-1) + 1

        # 2. Grid search for best temperature (minimize KL to uniform, report entropy)
        def kl_to_uniform(p):
            n = p.shape[-1]
            uniform = np.ones(n) / n
            return np.mean([kl_divergence(pred, uniform) for pred in p])

        best_temp = 1.0
        best_kl_uniform = float('inf')
        best_entropy = -float('inf')
        for temp in np.arange(config.TEMP_MIN, config.TEMP_MAX + config.TEMP_STEP, config.TEMP_STEP):
            first_five_pred_temp = apply_temperature_softmax(first_five_pred, temp)
            sixth_pred_temp = apply_temperature_softmax(sixth_pred, temp)
            # KL to uniform for all balls
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

        # 3. Apply best temperature for final predictions
        first_five_pred_temp = apply_temperature_softmax(first_five_pred, best_temp)
        sixth_pred_temp = apply_temperature_softmax(sixth_pred, best_temp)
        first_five_pred_numbers = np.argmax(first_five_pred_temp, axis=-1) + 1
        sixth_pred_number = np.argmax(sixth_pred_temp, axis=-1) + 1

        # 4. Analysis: matching, std, KL, plots
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

        # --- Standard Deviation & KL Divergence Evaluation ---
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

        # --- Seaborn Plots for Each Ball ---
        import matplotlib.pyplot as plt
        import seaborn as sns
        for i in range(5):
            plt.figure(figsize=(10, 4))
            sns.histplot(y_first_five_true_numbers[:, i], color='blue', label='True', kde=False, bins=69, stat='count', alpha=0.5)
            sns.histplot(first_five_pred_numbers[:, i], color='red', label='Predicted', kde=False, bins=69, stat='count', alpha=0.5)
            plt.title(f'Ball {i+1} Distribution (1-69)')
            plt.xlabel('Number')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.show()
        plt.figure(figsize=(10, 4))
        sns.histplot(y_sixth_true_number[:, 0], color='blue', label='True', kde=False, bins=26, stat='count', alpha=0.5)
        sns.histplot(sixth_pred_number[:, 0], color='red', label='Predicted', kde=False, bins=26, stat='count', alpha=0.5)
        plt.title('Powerball (6th Ball) Distribution (1-26)')
        plt.xlabel('Number')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("\nFailed to download or load the data. Please check the internet connection or the source URL.")

if __name__ == "__main__":
    main()
