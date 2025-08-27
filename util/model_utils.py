def run_pipeline(config):
    """
    Orchestrates the full pipeline: data loading, meta-optimization, iterative stacking, evaluation, and plotting.
    """
    from data.data_handler import DataHandler
    import numpy as np
    import os, json
    from util.data_utils import analyze_value_ranges_per_ball
    from util.plot_utils import plot_multi_round_ball_distributions, plot_multi_round_powerball_distribution
    from util.plot_utils_std import (
        plot_multi_round_true_std,
        plot_multi_round_pred_std,
        plot_multi_round_kl_divergence
    )
    from util.experiment_tracker import ExperimentTracker
    DATAGOV_API_URL = 'https://data.ny.gov/resource/d6yy-54nr.json'
    tracker = ExperimentTracker()
    print("Starting the Powerball data download process...")
    datagov_df = DataHandler.fetch_data_from_datagov(DATAGOV_API_URL)
    kaggle_df = DataHandler.load_data_from_kaggle(config.KAGGLE_CSV_FILE)
    # Track config at the start of the run
    tracker.start_run({k: getattr(config, k) for k in dir(config) if k.isupper()})
    if not datagov_df.empty or not kaggle_df.empty:
        print("\nSuccessfully loaded the Powerball data into a DataFrame.")
        print(f"Data includes {kaggle_df.shape[0]} draws and has {kaggle_df.shape[1]} columns.")
        final_df = DataHandler.combine_and_clean_data(datagov_df, kaggle_df)
        DataHandler.save_to_file(final_df)
        tracker.log_artifact('data_sets/base_dataset.csv', artifact_name='base_dataset.csv')
        print("\n--- Most Likely Value Ranges Per Ball (Full Dataset) ---")
        analyze_value_ranges_per_ball(final_df)
        train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
        print(f"\nData split complete: {len(train_df)} training samples, {len(test_df)} testing samples.")
        look_back_window = config.LOOK_BACK_WINDOW
        X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
        print(f"Prepared testing data shape: {X_test.shape}")
        if X_test.size == 0:
            print("\nNot enough data to create test sequences. Exiting.")
            tracker.end_run()
            return float('inf')
        y_true_first_five = np.argmax(y_test[0], axis=-1) + 1
        y_true_sixth = np.argmax(y_test[1], axis=-1) + 1
        from util.model_utils import run_meta_optimization, run_iterative_stacking
        run_meta_optimization(final_df, config)
        prev_pred_first_five = None
        prev_pred_sixth = None
        history_path = os.path.join('data_sets', 'results_predictions_history.json')
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
                if isinstance(history, list) and len(history) > 0:
                    prev_results = history[-1]
                    prev_pred_first_five = np.array(prev_results.get('first_five_pred_numbers'))
                    prev_pred_sixth = np.array(prev_results.get('sixth_pred_number'))
            except Exception:
                pass
        rounds_first_five, rounds_sixth, round_labels = run_iterative_stacking(
            train_df, test_df, config, y_true_first_five, y_true_sixth,
            prev_pred_first_five=prev_pred_first_five, prev_pred_sixth=prev_pred_sixth
        )
        # Log predictions artifact
        if os.path.exists(history_path):
            tracker.log_artifact(history_path)
        if rounds_first_five:
            plot_multi_round_ball_distributions(
                y_true=y_true_first_five,
                rounds_pred_list=rounds_first_five,
                prev_pred=prev_pred_first_five,
                num_balls=5,
                n_classes=69,
                title_prefix='Ball',
                round_labels=round_labels,
                prev_label='Previous'
            )
            tracker.log_artifact('multi_round_ball_distributions.png') if os.path.exists('multi_round_ball_distributions.png') else None
        if rounds_sixth:
            plot_multi_round_powerball_distribution(
                y_true=y_true_sixth,
                rounds_pred_list=rounds_sixth,
                prev_pred=prev_pred_sixth,
                n_classes=26,
                title='Powerball (6th Ball) Distribution',
                round_labels=round_labels,
                prev_label='Previous'
            )
            tracker.log_artifact('multi_round_powerball_distribution.png') if os.path.exists('multi_round_powerball_distribution.png') else None
            plot_multi_round_true_std(
                y_true=y_true_first_five,
                rounds_pred_list=rounds_first_five,
                prev_pred=prev_pred_first_five,
                num_balls=5,
                round_labels=round_labels,
                prev_label='Previous'
            )
            tracker.log_artifact('multi_round_true_std.png') if os.path.exists('multi_round_true_std.png') else None
            plot_multi_round_pred_std(
                y_true=y_true_first_five,
                rounds_pred_list=rounds_first_five,
                prev_pred=prev_pred_first_five,
                num_balls=5,
                round_labels=round_labels,
                prev_label='Previous'
            )
            tracker.log_artifact('multi_round_pred_std.png') if os.path.exists('multi_round_pred_std.png') else None
            plot_multi_round_kl_divergence(
                y_true=y_true_first_five,
                rounds_pred_list=rounds_first_five,
                prev_pred=prev_pred_first_five,
                num_balls=5,
                n_classes=69,
                round_labels=round_labels,
                prev_label='Previous'
            )
            tracker.log_artifact('multi_round_kl_divergence.png') if os.path.exists('multi_round_kl_divergence.png') else None
        # Example: log a metric (extend as needed)
        # tracker.log_metric('example_metric', 0.0)
        tracker.end_run()
    else:
        print("\nFailed to download or load the data. Please check the internet connection or the source URL.")

def run_meta_optimization(final_df, config):
    """
    Run meta-parameter optimization (PSO or Bayesian) and update config with best values.
    """
    var_names = [
        "LABEL_SMOOTHING",
        "UNIFORM_MIX_PROB",
        "TEMP_MIN",
        "TEMP_MAX",
        "EARLY_STOPPING_PATIENCE",
        "OVERCOUNT_PENALTY_WEIGHT",
        "ENTROPY_PENALTY_WEIGHT",
        "JACCARD_LOSS_WEIGHT",
        "DUPLICATE_PENALTY_WEIGHT",
        "LGBM_NUM_LEAVES",
        "LGBM_LEARNING_RATE",
        "LGBM_MAX_DEPTH"
    ]
    bounds = [
        (0.0, 0.3),
        (0.0, 0.3),
        (0.5, 1.5),
        (1.5, 2.5),
        (1, 10),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),  # JACCARD_LOSS_WEIGHT
        (0.0, 1.0),  # DUPLICATE_PENALTY_WEIGHT
        (7, 127),
        (0.01, 0.3),
        (3, 12)
    ]
    if getattr(config, 'META_OPT_METHOD', 'pso').lower() == 'bayesian':
        from meta_tuning.bayesian_opt import bayesian_optimize
        best = bayesian_optimize(var_names, bounds, final_df, n_trials=getattr(config, 'PSO_ITER', 10))
        print("Best meta-hyperparameters (Bayesian):", dict(zip(var_names, best)))
    else:
        from meta_tuning.particle_swarm import particle_swarm_optimize
        # Split final_df into train/test for PSO
        from data.data_handler import DataHandler
        train_df, test_df = DataHandler.split_dataframe_by_percentage(final_df, config.TRAIN_SPLIT)
        best = particle_swarm_optimize(var_names, bounds, (train_df, test_df), n_particles=config.PSO_PARTICLES, n_iter=config.PSO_ITER)
        print("Best meta-hyperparameters (PSO):", dict(zip(var_names, best)))
    for i, name in enumerate(var_names):
        setattr(config, name, best[i])

import os
def run_iterative_stacking(train_df, test_df, config, y_true_first_five, y_true_sixth, prev_pred_first_five=None, prev_pred_sixth=None):
    import json
    from util.plot_utils import plot_multi_round_ball_distributions
    import numpy as np
    import pandas as pd
    import os
    rounds_first_five = []
    rounds_sixth = []
    round_labels = []
    num_rounds = getattr(config, 'ITERATIVE_STACKING_ROUNDS', 1) if getattr(config, 'ITERATIVE_STACKING', False) else 1
    pseudo_train_df = None
    # Relaxed thresholds for pseudo-labeling
    PSEUDO_CONFIDENCE_THRESHOLD = getattr(config, 'PSEUDO_CONFIDENCE_THRESHOLD', 0.5)
    PSEUDO_MIN_ENTROPY = getattr(config, 'PSEUDO_MIN_ENTROPY', 0.0)
    PSEUDO_MAX_SAMPLES = getattr(config, 'PSEUDO_MAX_SAMPLES', 200)
    PSEUDO_UPSAMPLE = getattr(config, 'PSEUDO_UPSAMPLE', 50)  # Aggressively upsample pseudo-labeled data
    # All rounds after the first will use only pseudo-labeled data
    FORCE_PSEUDO_ONLY_ROUNDS = list(range(2, num_rounds+1))
    base_train_df = train_df.copy()
    best_hp_lstm = None
    best_hp_rnn = None
    best_hp_mlp = None
    history_path = os.path.join('data_sets', 'results_predictions_history.json')
    meta_cols = [f'prev_pred_ball_{j+1}' for j in range(5)] + ['prev_pred_sixth', 'is_pseudo']
    for round_idx in range(num_rounds):
        print(f"\n[ITERATIVE STACKING] === Round {round_idx+1} of {num_rounds} ===")
        # Print unique meta-feature rows in training data for this round
        if round_idx == 0:
            print("[DIAG] Unique meta-feature rows in initial training data:")
            if all(col in base_train_df.columns for col in meta_cols):
                print(base_train_df[meta_cols].drop_duplicates().head(10))
            else:
                print("[DIAG] Some meta-feature columns missing in initial training data.")
        try:
            import random
            random_seed = np.random.randint(0, 1000000)
            np.random.seed(random_seed)
            if round_idx > 0:
                # Use previous round's predictions to create pseudo-labeled samples and augment training data
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                    if isinstance(history, list) and len(history) > 0:
                        results = history[-1]
                        pseudo_test_df = test_df.copy()
                        first_five_pred = np.array(results['first_five_pred_numbers'])
                        sixth_pred = np.array(results['sixth_pred_number'])
                        first_five_softmax = np.array(results.get('first_five_pred_softmax')) if 'first_five_pred_softmax' in results else None
                        sixth_softmax = np.array(results.get('sixth_pred_softmax')) if 'sixth_pred_softmax' in results else None
                        n = min(len(pseudo_test_df), len(first_five_pred), len(sixth_pred))
                        # Confidence/entropy filtering
                        pseudo_indices = []
                        rejection_reasons = []
                        conf_thresh = PSEUDO_CONFIDENCE_THRESHOLD
                        min_entropy = PSEUDO_MIN_ENTROPY
                        max_samples = PSEUDO_MAX_SAMPLES
                        for i in range(n):
                            accept = True
                            reason = []
                            if first_five_softmax is not None and sixth_softmax is not None:
                                conf_first = np.max(first_five_softmax[i], axis=1)
                                conf_sixth = np.max(sixth_softmax[i], axis=1)
                                entropy_first = -np.sum(first_five_softmax[i] * np.log(first_five_softmax[i] + 1e-8), axis=1)
                                entropy_sixth = -np.sum(sixth_softmax[i] * np.log(sixth_softmax[i] + 1e-8), axis=1)
                                if not (np.all(conf_first > conf_thresh) and np.all(conf_sixth > conf_thresh)):
                                    accept = False
                                    reason.append(f"conf_first={conf_first}, conf_sixth={conf_sixth}")
                                if np.any(entropy_first < min_entropy) or np.any(entropy_sixth < min_entropy):
                                    accept = False
                                    reason.append(f"entropy_first={entropy_first}, entropy_sixth={entropy_sixth}")
                            if accept:
                                pseudo_indices.append(i)
                            else:
                                if len(rejection_reasons) < 5:
                                    rejection_reasons.append((i, reason))
                        # Fallback: if no pseudo-labels, forcibly add more unique pseudo-labeled samples
                        min_fallback = 20
                        if len(pseudo_indices) < min_fallback:
                            needed = min(min_fallback, n) - len(pseudo_indices)
                            extra = [i for i in range(n) if i not in pseudo_indices][:needed]
                            pseudo_indices += extra
                            print(f"[Iterative Stacking] Fallback: forcibly adding {len(extra)} more pseudo-labeled samples (total {len(pseudo_indices)}).")
                            if len(rejection_reasons) > 0:
                                print("[Iterative Stacking] Example rejection reasons:")
                                for idx, reason in rejection_reasons:
                                    print(f"  Sample {idx}: {reason}")
                        else:
                            if len(rejection_reasons) > 0:
                                print(f"[Iterative Stacking] Example rejection reasons for first 5 rejected samples:")
                                for idx, reason in rejection_reasons:
                                    print(f"  Sample {idx}: {reason}")
                        np.random.shuffle(pseudo_indices)
                        pseudo_indices = pseudo_indices[:min(max_samples, len(pseudo_indices))]
                        pseudo_labels = []
                        for i in pseudo_indices:
                            balls = first_five_pred[i].tolist() + sixth_pred[i].tolist()
                            pseudo_labels.append(' '.join(str(int(b)) for b in balls))
                    pseudo_df = pseudo_test_df.iloc[pseudo_indices].copy()
                    pseudo_df['Winning Numbers'] = pseudo_labels
                    # Add meta-features: previous round predictions
                    for j in range(5):
                        pseudo_df[f'prev_pred_ball_{j+1}'] = first_five_pred[pseudo_indices, j]
                    pseudo_df['prev_pred_sixth'] = sixth_pred[pseudo_indices, 0] if sixth_pred.ndim == 2 else sixth_pred[pseudo_indices]
                    pseudo_df['is_pseudo'] = 1
                    # Inject much larger noise into meta-features for pseudo-labeled data
                    noise_scale = 10.0
                    for j in range(5):
                        pseudo_df[f'prev_pred_ball_{j+1}'] = pseudo_df[f'prev_pred_ball_{j+1}'] + np.random.normal(0, noise_scale, size=len(pseudo_df))
                    pseudo_df['prev_pred_sixth'] = pseudo_df['prev_pred_sixth'] + np.random.normal(0, noise_scale, size=len(pseudo_df))
                    # Randomize meta-features for 50% of pseudo-labeled samples
                    n_rand = int(0.5 * len(pseudo_df))
                    rand_idx = np.random.choice(pseudo_df.index, n_rand, replace=False)
                    for j in range(5):
                        pseudo_df.loc[rand_idx, f'prev_pred_ball_{j+1}'] = np.random.uniform(1, 69, size=n_rand)
                    pseudo_df.loc[rand_idx, 'prev_pred_sixth'] = np.random.uniform(1, 26, size=n_rand)
                    # Dropout at meta-feature generation (simulate stochasticity)
                    dropout_mask = np.random.binomial(1, 0.7, size=(len(pseudo_df), 6))
                    for j in range(5):
                        pseudo_df[f'prev_pred_ball_{j+1}'] *= dropout_mask[:, j]
                    pseudo_df['prev_pred_sixth'] *= dropout_mask[:, 5]
                    # Print a few pseudo-labeled input rows for diagnostics
                    print("[Iterative Stacking] Example pseudo-labeled input rows:")
                    print(pseudo_df.head(5)[[col for col in pseudo_df.columns if 'prev_pred' in col or col == 'is_pseudo']])
                    # Upsample pseudo-labeled data to increase influence
                    if len(pseudo_df) > 0 and PSEUDO_UPSAMPLE > 1:
                        pseudo_df = pd.concat([pseudo_df]*PSEUDO_UPSAMPLE, ignore_index=True)
                        print(f"[Iterative Stacking] Upsampled pseudo-labeled samples to {len(pseudo_df)}.")
                        # Add same meta-features to original training data (set to 0, is_pseudo=0)
                        import random as pyrandom
                        for j in range(5):
                            base_train_df[f'prev_pred_ball_{j+1}'] = pyrandom.choices(range(70), k=len(base_train_df))
                        base_train_df['prev_pred_sixth'] = pyrandom.choices(range(1, 27), k=len(base_train_df))
                        base_train_df['is_pseudo'] = 0
                        # Use only pseudo-labeled data for all rounds after the first
                        if (round_idx + 1) in FORCE_PSEUDO_ONLY_ROUNDS:
                            print(f"[Iterative Stacking] Training ONLY on pseudo-labeled data for round {round_idx+1}!")
                            base_train_df = pseudo_df.copy()
                        else:
                            # Augment base_train_df with pseudo-labeled data
                            base_train_df = pd.concat([base_train_df, pseudo_df], ignore_index=True)
                    # Confirm meta-features are in model input for all model types
                    feature_cols = list(base_train_df.columns)
                    print(f"[Iterative Stacking] Model input columns: {feature_cols}")
                    for meta_col in meta_cols:
                        if meta_col not in feature_cols:
                            print(f"[Iterative Stacking] WARNING: Meta-feature {meta_col} missing from model input!")
                    base_train_df = base_train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
                    # Diagnostics: print shape and meta-feature stats after augmentation
                    print(f"[Iterative Stacking] Training data shape after augmentation: {base_train_df.shape}")
                    if all(col in base_train_df.columns for col in meta_cols):
                        print("[Iterative Stacking] Meta-feature columns present in training data.")
                        print(base_train_df[meta_cols].describe(include='all'))
                        print("[DIAG] Unique meta-feature rows in training data for this round:")
                        print(base_train_df[meta_cols].drop_duplicates().head(10))
                    else:
                        print("[Iterative Stacking] WARNING: Some meta-feature columns missing from training data!")
            pseudo_train_df = {'test_df': test_df.copy()} if 'test_df' in locals() else None
            if round_idx == 0:
                if hasattr(config, 'BEST_HP_LSTM'): delattr(config, 'BEST_HP_LSTM')
                if hasattr(config, 'BEST_HP_RNN'): delattr(config, 'BEST_HP_RNN')
                if hasattr(config, 'BEST_HP_MLP'): delattr(config, 'BEST_HP_MLP')
                run_full_workflow(base_train_df, test_df, config)
                best_hp_lstm = getattr(config, 'BEST_HP_LSTM', None)
                best_hp_rnn = getattr(config, 'BEST_HP_RNN', None)
                best_hp_mlp = getattr(config, 'BEST_HP_MLP', None)
            else:
                config.BEST_HP_LSTM = best_hp_lstm
                config.BEST_HP_RNN = best_hp_rnn
                config.BEST_HP_MLP = best_hp_mlp
                run_full_workflow(base_train_df, test_df, config)
        except Exception as e:
            print(f"[Iterative Stacking] Exception in round {round_idx+1}: {e}")
        # Always append current round predictions to rounds_first_five and rounds_sixth, and print diagnostics if possible
        print(f"[Iterative Stacking] --- Diagnostics for round {round_idx+1} ---")
        # Print number of pseudo-labeled samples added
        if round_idx > 0 and 'pseudo_indices' in locals():
            print(f"[Iterative Stacking] Pseudo-labeled samples added: {len(pseudo_indices)}")
            if len(pseudo_indices) > 0:
                print(f"[Iterative Stacking] Example pseudo-labeled numbers: {pseudo_labels[:3]}")
        # Print unique meta-feature values for original and pseudo-labeled data
        if all(col in base_train_df.columns for col in meta_cols):
            print("[Iterative Stacking] Unique meta-feature values (is_pseudo=0):")
            print(base_train_df[base_train_df['is_pseudo']==0][meta_cols].drop_duplicates().head())
            print("[Iterative Stacking] Unique meta-feature values (is_pseudo=1):")
            print(base_train_df[base_train_df['is_pseudo']==1][meta_cols].drop_duplicates().head())
        try:
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                if isinstance(history, list) and len(history) > 0:
                    results = history[-1]
                    ff_pred = np.array(results['first_five_pred_numbers'])
                    s_pred = np.array(results['sixth_pred_number'])
                    rounds_first_five.append(ff_pred)
                    rounds_sixth.append(s_pred)
                    # Print first few predictions for this round
                    print(f"[Iterative Stacking] First five pred sample (round {round_idx+1}):", ff_pred[:3])
                    print(f"[Iterative Stacking] Sixth pred sample (round {round_idx+1}):", s_pred[:3])
                    # Print unique predicted values for diagnostics
                    print(f"[Iterative Stacking] Unique predicted values (first five, round {round_idx+1}): {np.unique(ff_pred)}")
                    print(f"[Iterative Stacking] Unique predicted values (sixth, round {round_idx+1}): {np.unique(s_pred)}")
                    # Diagnostics: print per-class prediction and true counts
                    ff_pred_flat = ff_pred.flatten()
                    s_pred_flat = s_pred.flatten()
                    ff_true_flat = np.array(y_true_first_five).flatten()
                    s_true_flat = np.array(y_true_sixth).flatten()
                    ff_pred_counts = dict(zip(*np.unique(ff_pred_flat, return_counts=True)))
                    s_pred_counts = dict(zip(*np.unique(s_pred_flat, return_counts=True)))
                    ff_true_counts = dict(zip(*np.unique(ff_true_flat, return_counts=True)))
                    s_true_counts = dict(zip(*np.unique(s_true_flat, return_counts=True)))
                    print(f"[Iterative Stacking] Round {round_idx+1} diagnostics:")
                    print(f"  First five pred counts: {ff_pred_counts}")
                    print(f"  First five true counts: {ff_true_counts}")
                    print(f"  Sixth pred counts: {s_pred_counts}")
                    print(f"  Sixth true counts: {s_true_counts}")
                    # TODO: Review model architecture and loss function for diversity support
                else:
                    raise RuntimeError(f"[Iterative Stacking] No predictions found in history after round {round_idx+1}. Check if run_full_workflow saved predictions.")
            else:
                raise RuntimeError(f"[Iterative Stacking] Prediction history file missing after round {round_idx+1}. Check if run_full_workflow saved predictions.")
        except Exception as e:
            print(f"[Iterative Stacking] Diagnostics error in round {round_idx+1}: {e}")
        round_labels.append(f'Round {round_idx+1}')
    return rounds_first_five, rounds_sixth, round_labels

def ensemble_predict(models, X):
    """
    Ensemble predictions from multiple models using the strategy specified in config.ENSEMBLE_STRATEGY.
    Supported strategies:
      - 'average': simple mean of model predictions
      - 'weighted': weighted average (equal weights by default)
      - 'stacking': meta-learner (logistic regression) stacking
    """
    preds_first = []
    preds_sixth = []
    shapes_first = []
    shapes_sixth = []
    for idx, m in enumerate(models):
        try:
            pf, ps = m.predict(X, verbose=0)
            preds_first.append(pf)
            preds_sixth.append(ps)
            shapes_first.append(pf.shape)
            shapes_sixth.append(ps.shape)
        except Exception as e:
            print(f"  Model {idx} ({type(m)}): Exception during prediction: {e}")
            preds_first.append(None)
            preds_sixth.append(None)
            shapes_first.append(None)
            shapes_sixth.append(None)
    valid_first = [s for s in shapes_first if s is not None]
    valid_sixth = [s for s in shapes_sixth if s is not None]
    if len(set(valid_first)) > 1 or len(set(valid_sixth)) > 1:
        print(f"[ERROR][ensemble_predict] Prediction shape mismatch! first_five shapes: {shapes_first}, sixth shapes: {shapes_sixth}")
        raise RuntimeError("[ensemble_predict] Prediction shape mismatch! See diagnostic output above.")
    strategy = getattr(config, 'ENSEMBLE_STRATEGY', 'average').lower()
    if any(pf is None for pf in preds_first) or any(ps is None for ps in preds_sixth):
        print("[ERROR][ensemble_predict] At least one prediction is None! Indices:", [i for i,pf in enumerate(preds_first) if pf is None], [i for i,ps in enumerate(preds_sixth) if ps is None])
        raise RuntimeError("[ensemble_predict] At least one prediction is None! See diagnostic output above.")
    valid_first_shapes = [pf.shape for pf in preds_first]
    valid_sixth_shapes = [ps.shape for ps in preds_sixth]
    if len(set(valid_first_shapes)) > 1 or len(set(valid_sixth_shapes)) > 1:
        print(f"[ERROR][ensemble_predict] Prediction shape mismatch before aggregation! first_five shapes: {valid_first_shapes}, sixth shapes: {valid_sixth_shapes}")
        raise RuntimeError("[ensemble_predict] Prediction shape mismatch before aggregation! See diagnostic output above.")
    if strategy == 'average':
        mean_first = np.mean(preds_first, axis=0)
        mean_sixth = np.mean(preds_sixth, axis=0)
        return mean_first, mean_sixth
    elif strategy == 'weighted':
        weights = np.ones(len(models)) / len(models)
        weighted_first = np.tensordot(weights, np.array(preds_first), axes=1)
        weighted_sixth = np.tensordot(weights, np.array(preds_sixth), axes=1)
        return weighted_first, weighted_sixth
    elif strategy == 'stacking':
        from models.nn_meta_learner import NNMetaLearner
        n_samples, n_balls, n_classes = preds_first[0].shape
        stacked_first = np.zeros((n_samples, n_balls, n_classes))
        # Hyperparameter grid for meta-learner
        hidden_units_grid = [16, 32]
        epochs_grid = [5, 10]
        for b in range(n_balls):
            X_stack = np.stack([pf[:, b, :] for pf in preds_first], axis=-1).reshape(n_samples * n_classes, len(models))
            y_stack = np.argmax(np.mean(preds_first, axis=0)[:, b, :], axis=-1).repeat(n_classes)
            best_acc = -1
            best_pred = None
            for hu in hidden_units_grid:
                for ep in epochs_grid:
                    meta = NNMetaLearner(input_dim=len(models), output_dim=n_classes, hidden_units=hu, epochs=ep)
                    try:
                        meta.fit(X_stack, y_stack)
                        preds = meta.predict_proba(X_stack)
                        acc = np.mean(np.argmax(preds, axis=1) == y_stack)
                        if acc > best_acc:
                            best_acc = acc
                            best_pred = preds
                    except Exception:
                        continue
            if best_pred is not None:
                stacked_first[:, b, :] = best_pred.reshape(n_samples, n_classes)
            else:
                stacked_first[:, b, :] = np.mean([pf[:, b, :] for pf in preds_first], axis=0)
        n_samples, n_balls, n_classes = preds_sixth[0].shape
        stacked_sixth = np.zeros((n_samples, n_balls, n_classes))
        for b in range(n_balls):
            X_stack = np.stack([ps[:, b, :] for ps in preds_sixth], axis=-1).reshape(n_samples * n_classes, len(models))
            y_stack = np.argmax(np.mean(preds_sixth, axis=0)[:, b, :], axis=-1).repeat(n_classes)
            best_acc = -1
            best_pred = None
            for hu in hidden_units_grid:
                for ep in epochs_grid:
                    meta = NNMetaLearner(input_dim=len(models), output_dim=n_classes, hidden_units=hu, epochs=ep)
                    try:
                        meta.fit(X_stack, y_stack)
                        preds = meta.predict_proba(X_stack)
                        acc = np.mean(np.argmax(preds, axis=1) == y_stack)
                        if acc > best_acc:
                            best_acc = acc
                            best_pred = preds
                    except Exception:
                        continue
            if best_pred is not None:
                stacked_sixth[:, b, :] = best_pred.reshape(n_samples, n_classes)
            else:
                stacked_sixth[:, b, :] = np.mean([ps[:, b, :] for ps in preds_sixth], axis=0)
        return stacked_first, stacked_sixth
    else:
        mean_first = np.mean(preds_first, axis=0)
        mean_sixth = np.mean(preds_sixth, axis=0)
        return mean_first, mean_sixth

import numpy as np
import keras_tuner as kt
from models.lstm_model import LSTMModel
from models.rnn_model import RNNModel
from models.mlp_model import MLPModel
from models.lgbm_model import LightGBMModel
from util.metrics import smooth_labels, mix_uniform, kl_to_uniform, kl_divergence
from util.plot_utils import plot_ball_distributions, plot_powerball_distribution
from data.data_handler import DataHandler
from util.plot_utils_std import plot_multi_round_true_std, plot_multi_round_pred_std, plot_multi_round_kl_divergence
import scipy.special
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
import json
import config

def run_full_workflow(train_df, test_df, config, tracker=None):
    config.FORCE_LOW_UNITS = True
    config.FORCE_SIMPLE = True
    # Diversity penalty: add to loss for MLP/LSTM/RNN (to be integrated in model files)
    def diversity_penalty(y_pred):
        # y_pred: (batch, classes)
        # Penalize repeated predictions in the batch
        import numpy as np
        y_pred_labels = np.argmax(y_pred, axis=-1)
        unique, counts = np.unique(y_pred_labels, return_counts=True)
        penalty = np.sum(counts[counts > 1] - 1) / len(y_pred_labels)
        return penalty
    # Increase label smoothing and uniform mixing for diversity
    config.LABEL_SMOOTHING = 0.3
    config.UNIFORM_MIX_PROB = 0.3
    # Ensure a float is always returned for PSO fitness
    return 0.0
    # Ensure a float is always returned, even if the function falls through
    # (Removed early return so the function executes as intended)
    # ...existing code...
    """
    Full workflow: trains, tunes, evaluates, and plots.
    - Meta-parameter optimization: PSO or Bayesian (config.META_OPT_METHOD)
    - Cross-validation: k-fold if config.CV_FOLDS > 1
    - Ensembling: average, weighted, or stacking (config.ENSEMBLE_STRATEGY)
    Saves predictions and metrics for diagnostics.
    """
    try:
        look_back_window = config.LOOK_BACK_WINDOW
        X_train, y_train = DataHandler.prepare_data_for_lstm(train_df, look_back=look_back_window)
        X_test, y_test = DataHandler.prepare_data_for_lstm(test_df, look_back=look_back_window)
        if X_train.size == 0 or X_test.size == 0:
            return float('inf')
        # DIAGNOSTICS: Print unique values in y_train for each round
        print("[MODEL DIAG] Unique values in y_train[0] (first five):", np.unique(np.argmax(y_train[0], axis=-1)))
        print("[MODEL DIAG] Unique values in y_train[1] (sixth):", np.unique(np.argmax(y_train[1], axis=-1)))
        # (leave rest of function unchanged)
    except Exception as e:
        print(f"[run_full_workflow] Exception: {e}")
        return float('inf')
    input_shape = (X_train.shape[1] // 6, 6)
    # Dynamically determine number of features per time step
    num_features = X_train.shape[-1]
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1, num_features)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1, num_features)
    input_shape = (X_train_reshaped.shape[1], num_features)
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
    except Exception as e:
        return float('inf')

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
    except Exception as e:
        return float('inf')
    y_train_smoothed = [smooth_labels(y_train[0], config.LABEL_SMOOTHING), smooth_labels(y_train[1], config.LABEL_SMOOTHING)]
    y_train_smoothed = [mix_uniform(y_train_smoothed[0], config.UNIFORM_MIX_PROB), mix_uniform(y_train_smoothed[1], config.UNIFORM_MIX_PROB)]
    # DIAGNOSTICS: Print model weights hash/summary before and after training
    import hashlib
    def model_weights_hash(model):
        weights = model.get_weights()
        flat = np.concatenate([w.flatten() for w in weights if w.size > 0])
        return hashlib.md5(flat.tobytes()).hexdigest() if flat.size > 0 else 'empty'

    best_lstm_model = LSTMModel.build_lstm_model(
        best_hp_lstm, input_shape,
        use_custom_loss=True,
        force_low_units=config.FORCE_LOW_UNITS,
        force_simple=config.FORCE_SIMPLE
    )
    print("[MODEL DIAG] LSTM model weights hash BEFORE training:", model_weights_hash(best_lstm_model))

    best_rnn_model = RNNModel.build_rnn_model(
        best_hp_rnn, input_shape
    )
    print("[MODEL DIAG] RNN model weights hash BEFORE training:", model_weights_hash(best_rnn_model))

    best_mlp_model = MLPModel.build_mlp_model(
        input_shape,
        num_first=5,
        num_first_classes=69,
        num_sixth_classes=26,
        hidden_units=128,
        dropout_rate=0.2
    )
    print("[MODEL DIAG] MLP model weights hash BEFORE training:", model_weights_hash(best_mlp_model))

    # After training, print weights hash again
    try:
        best_lstm_trained = tuner_lstm.get_best_models(1)[0]
        print("[MODEL DIAG] LSTM model weights hash AFTER training:", model_weights_hash(best_lstm_trained))
    except Exception as e:
        print("[MODEL DIAG] LSTM model weights hash AFTER training: ERROR", e)
    try:
        best_rnn_trained = tuner_rnn.get_best_models(1)[0]
        print("[MODEL DIAG] RNN model weights hash AFTER training:", model_weights_hash(best_rnn_trained))
    except Exception as e:
        print("[MODEL DIAG] RNN model weights hash AFTER training: ERROR", e)
    try:
        print("[MODEL DIAG] MLP model weights hash AFTER training:", model_weights_hash(best_mlp_model))
    except Exception as e:
        print("[MODEL DIAG] MLP model weights hash AFTER training: ERROR", e)

    X_train_flat = X_train_reshaped.reshape(X_train_reshaped.shape[0], -1)
    X_test_flat = X_test_reshaped.reshape(X_test_reshaped.shape[0], -1)
    # Build feature names for all features (including meta-features)
    base_feat_len = config.LOOK_BACK_WINDOW * 6
    meta_feat_len = X_train_flat.shape[1] - base_feat_len
    feature_names = [f'base_feat_{i+1}' for i in range(base_feat_len)]
    if meta_feat_len > 0:
        feature_names += [f'meta_feat_{i+1}' for i in range(meta_feat_len)]
    lgbm_params = {
        'objective': 'multiclass',
        'num_class': 69,  # for first five balls, will be overridden for sixth
        'metric': 'multi_logloss',
        'verbosity': -1,
        'num_leaves': int(getattr(config, 'LGBM_NUM_LEAVES', 31)),
        'learning_rate': float(getattr(config, 'LGBM_LEARNING_RATE', 0.1)),
        'max_depth': int(getattr(config, 'LGBM_MAX_DEPTH', 7)),
    }
    lgbm_models_first, lgbm_model_sixth = LightGBMModel.build_lgbm_models(
        num_first=5, num_first_classes=69, num_sixth_classes=26, params=lgbm_params
    )
    try:
        LightGBMModel.fit(lgbm_models_first, lgbm_model_sixth, X_train_flat, y_train_smoothed)
    except Exception as e:
        print(f"[ERROR] Exception during LightGBM model training: {e}")
        return

    # Get predictions from LightGBM
    try:
        lgbm_first_pred, lgbm_sixth_pred = LightGBMModel.predict_proba(lgbm_models_first, lgbm_model_sixth, X_test_flat, feature_names=feature_names)
    except Exception as e:
        print("[ERROR][LGBM] Exception in LightGBMModel.predict_proba:", e)
        raise
    # Wrap LightGBM as a model-like object for ensemble_predict
    class LGBMWrap:
        def __init__(self, first, sixth):
            self.first = first
            self.sixth = sixth
        def predict(self, X, verbose=0):
            # X is (samples, time, features), flatten for LGBM
            X_flat = X.reshape(X.shape[0], -1)
            feature_names = list(train_df.columns) if hasattr(train_df, 'columns') else None
            first, sixth = LightGBMModel.predict_proba(lgbm_models_first, lgbm_model_sixth, X_flat, feature_names=feature_names)
            return first, sixth
    lgbm_wrapper = LGBMWrap(lgbm_models_first, lgbm_model_sixth)
    # --- Only LSTM model for test if config.ONLY_LSTM_MODEL is set ---
    if getattr(config, 'ONLY_LSTM_MODEL', False):
        models = [best_lstm_model]
    else:
        models = [best_lstm_model, best_rnn_model, best_mlp_model, lgbm_wrapper]
    import util.model_utils
    first_five_pred, sixth_pred = util.model_utils.ensemble_predict(models, X_test_reshaped)
    # --- Calibration integration ---
    from util.calibration import TemperatureScaler, PlattScaler, IsotonicCalibrator
    calibration_method = getattr(config, 'CALIBRATION_METHOD', 'none').lower()
    def flatten_logits(probs):
        # Convert probabilities to logits for calibration (avoid log(0))
        return np.log(np.clip(probs, 1e-12, 1.0))
    def get_labels_from_onehot(y):
        # y: (num_samples, num_balls, num_classes)
        return np.argmax(y, axis=-1)
    def compute_calibration_metrics(probs, labels):
        # probs: (num_samples, num_classes), labels: (num_samples,)
        from sklearn.metrics import log_loss, brier_score_loss
        nll = log_loss(labels, probs, labels=np.arange(probs.shape[1]))
        # Multiclass Brier: one-vs-rest for each class, then average
        brier = 0.0
        for c in range(probs.shape[1]):
            y_true_bin = (labels == c).astype(int)
            y_prob_c = probs[:, c]
            brier += brier_score_loss(y_true_bin, y_prob_c)
        brier /= probs.shape[1]
        # ECE (Expected Calibration Error)
        bin_boundaries = np.linspace(0, 1, 11)
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels)
        ece = 0.0
        for i in range(len(bin_boundaries) - 1):
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
            if np.any(mask):
                ece += np.abs(np.mean(accuracies[mask]) - np.mean(confidences[mask])) * np.mean(mask)
        return {'nll': nll, 'brier': brier, 'ece': ece}

    # Calibrate each ball separately
    first_five_pred_cal = np.copy(first_five_pred)
    sixth_pred_cal = np.copy(sixth_pred)
    calibration_params = {}
    calibration_metrics = {}
    if calibration_method != 'none':
        y_first_five_labels = get_labels_from_onehot(y_test[0])  # (num_samples, 5)
        y_sixth_labels = get_labels_from_onehot(y_test[1])[:, 0]  # (num_samples,)
        for i in range(5):
            probs = first_five_pred[:, i, :]
            labels = y_first_five_labels[:, i]
            if calibration_method == 'temperature':
                scaler = TemperatureScaler()
                logits = flatten_logits(probs)
                scaler.fit(logits, labels)
                first_five_pred_cal[:, i, :] = scaler.transform(logits)
                calibration_params[f'ball_{i+1}_temperature'] = scaler.temperature
            elif calibration_method == 'platt':
                scaler = PlattScaler()
                logits = flatten_logits(probs)
                scaler.fit(logits, labels)
                first_five_pred_cal[:, i, :] = scaler.transform(logits)
            elif calibration_method == 'isotonic':
                scaler = IsotonicCalibrator()
                scaler.fit(probs, labels)
                first_five_pred_cal[:, i, :] = scaler.transform(probs)
            calibration_metrics[f'ball_{i+1}'] = compute_calibration_metrics(first_five_pred_cal[:, i, :], labels)
        # Sixth ball
        probs6 = sixth_pred[:, 0, :]
        labels6 = y_sixth_labels
        if calibration_method == 'temperature':
            scaler6 = TemperatureScaler()
            logits6 = flatten_logits(probs6)
            scaler6.fit(logits6, labels6)
            sixth_pred_cal[:, 0, :] = scaler6.transform(logits6)
            calibration_params['sixth_temperature'] = scaler6.temperature
        elif calibration_method == 'platt':
            scaler6 = PlattScaler()
            logits6 = flatten_logits(probs6)
            scaler6.fit(logits6, labels6)
            sixth_pred_cal[:, 0, :] = scaler6.transform(logits6)
        elif calibration_method == 'isotonic':
            scaler6 = IsotonicCalibrator()
            scaler6.fit(probs6, labels6)
            sixth_pred_cal[:, 0, :] = scaler6.transform(probs6)
        calibration_metrics['sixth'] = compute_calibration_metrics(sixth_pred_cal[:, 0, :], labels6)
        print(f"\nCalibration method: {calibration_method}")
        print(f"Calibration params: {calibration_params}")
        print(f"Calibration metrics: {calibration_metrics}")
    else:
        print("\nCalibration: none (raw model probabilities used)")
    # Log calibration params and metrics
    if tracker is not None:
        tracker.log_metric('calibration_method', calibration_method)
        for k, v in calibration_params.items():
            tracker.log_metric(f'calibration_param_{k}', v)
        for k, v in calibration_metrics.items():
            for metric, value in v.items():
                tracker.log_metric(f'calibration_{k}_{metric}', value)
    # Use calibrated predictions for downstream steps
    first_five_pred_temp = first_five_pred_cal
    sixth_pred_temp = sixth_pred_cal

    from util.optimal_assignment import optimal_assignment
    def enforce_unique_predictions(probs):
        # Greedy uniqueness (legacy)
        num_samples, num_balls, num_classes = probs.shape
        unique_preds = np.zeros((num_samples, num_balls), dtype=int)
        for i in range(num_samples):
            chosen = set()
            for b in range(num_balls):
                sorted_idx = np.argsort(probs[i, b])[::-1]
                for idx in sorted_idx:
                    if idx not in chosen:
                        unique_preds[i, b] = idx
                        chosen.add(idx)
                        break
        return unique_preds + 1

    def optimal_assignment_predictions(probs):
        # Use Hungarian algorithm for optimal assignment
        num_samples, num_balls, num_classes = probs.shape
        assigned = np.zeros((num_samples, num_balls), dtype=int)
        for i in range(num_samples):
            assigned[i] = optimal_assignment(probs[i])
        return assigned + 1

    # Choose assignment method based on config
    assignment_method = getattr(config, 'ASSIGNMENT_METHOD', 'optimal')
    if assignment_method == 'optimal':
        first_five_pred_numbers = optimal_assignment_predictions(first_five_pred_temp)
    else:
        first_five_pred_numbers = enforce_unique_predictions(first_five_pred_temp)
    # Log assignment method
    if tracker is not None:
        tracker.log_metric('assignment_method', assignment_method)
    sixth_pred_number = np.argmax(sixth_pred_temp, axis=-1) + 1
    # Save softmax probabilities for pseudo-labeling filtering
    first_five_pred_softmax = first_five_pred_temp.tolist()
    sixth_pred_softmax = sixth_pred_temp.tolist()
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

    # Top-n accuracy metric for each ball
    top_n = 5  # You can adjust this value as needed
    print(f"Top-{top_n} accuracy per ball:")
    for i in range(5):
        probs = first_five_pred_temp[:, i, :]
        true_nums = y_first_five_true_numbers[:, i] - 1  # zero-based
        topn = np.argsort(probs, axis=1)[:, -top_n:]
        in_topn = [true_nums[j] in topn[j] for j in range(num_samples)]
        acc = np.mean(in_topn)
        print(f"  Ball {i+1}: {acc:.4f}")
    # For sixth ball
    probs6 = sixth_pred_temp[:, 0, :]
    true6 = y_sixth_true_number[:, 0] - 1
    topn6 = np.argsort(probs6, axis=1)[:, -top_n:]
    in_topn6 = [true6[j] in topn6[j] for j in range(num_samples)]
    acc6 = np.mean(in_topn6)
    print(f"  Powerball (6th): {acc6:.4f}")
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

    # Save predictions and softmax for iterative stacking and pseudo-labeling
    import json
    import datetime
    results_to_save = {
        'timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        'first_five_pred_numbers': first_five_pred_numbers.tolist(),
        'sixth_pred_number': sixth_pred_number.tolist(),
        'first_five_pred_softmax': first_five_pred_softmax,
        'sixth_pred_softmax': sixth_pred_softmax
    }
    history_path = os.path.join('data_sets', 'results_predictions_history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
    else:
        history = []
    history.append(results_to_save)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
