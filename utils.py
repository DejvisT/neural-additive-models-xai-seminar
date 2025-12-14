import neural_additive_models.data_utils as data_utils
import numpy as np
import os
import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from neural_additive_models.models import NAM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_col_min_max(dataset_name, correlated_n=None, rho=None, seed=None):
  """Loads the dataset according to the `dataset_name` passed."""
  if dataset_name == 'Housing':
    dataset = data_utils.load_california_housing_data()
  elif dataset_name == 'BreastCancer':
    dataset = data_utils.load_breast_data()
  elif dataset_name == 'Recidivism':
    dataset = data_utils.load_recidivism_data()
  elif dataset_name == 'Fico':
    dataset = data_utils.load_fico_score_data()
  elif dataset_name == 'Mimic2':
    dataset = load_mimic2_data()
  elif dataset_name == 'Credit':
    dataset = data_utils.load_credit_data()
  elif dataset_name == 'Correlated_linear':
    dataset = data_utils.load_correlated_linear_data(correlated_n, rho, seed)
  elif dataset_name == 'Correlated_nonlinear':
    dataset = data_utils.load_correlated_nonlinear_data(correlated_n, rho, seed)
  elif dataset_name == 'Synthetic':
    dataset = data_utils.load_synthetic_data()
  else:
    raise ValueError('{} not found!'.format(dataset_name))

  if 'full' in dataset:
    dataset = dataset['full']
  x = dataset['X']
  col_min_max = {}
  for col in x:
    unique_vals = x[col].unique()
    col_min_max[col] = (np.min(unique_vals), np.max(unique_vals))
  return col_min_max


def inverse_min_max_scaler(x, min_val, max_val):
  return (x + 1)/2 * (max_val - min_val) + min_val 



def load_nam_checkpoint(ckpt_dir: str, hyperparameters=None):
    """
    Load a NAM (Neural Additive Model) from a TensorFlow v1 checkpoint directory.

    Args:
        ckpt_dir (str): Path to the checkpoint directory containing .index and .data files.
        hyperparameters (dict, optional): Dict with keys 'dropout', 'feature_dropout', 'activation', 'shallow'.
                                         If None, uses defaults (dropout=0.0, feature_dropout=0.0, activation='relu', shallow=False).

    Returns:
        (nam, sess): A tuple containing the restored NAM model and active TensorFlow session.
    """
    # --- Locate checkpoint ---
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
    if ckpt_path is None:
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.index')]
        if not ckpt_files:
            raise FileNotFoundError(f"No valid checkpoint found in {ckpt_dir}")
        name = ckpt_files[0].split('.index')[0]
        ckpt_path = os.path.join(ckpt_dir, name)
    print(f"Using checkpoint: {ckpt_path}")

    # --- Read variable shapes to reconstruct model architecture ---
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_map = reader.get_variable_to_shape_map()

    units_by_idx = {}
    for name, shape in var_map.items():
        m = re.match(r"^model_0/activation_layer_(\d+)/beta$", name)
        if m:
            units_by_idx[int(m.group(1))] = shape[1]

    if not units_by_idx:
        raise ValueError("Could not infer unit shapes from checkpoint metadata.")

    num_units_list = [units_by_idx[i] for i in sorted(units_by_idx)]
    num_inputs = len(num_units_list)

    print("Feature widths:", num_units_list)
    print("Num input features:", num_inputs)

    # --- Build the model with hyperparameters ---
    tf.reset_default_graph()
    hp = hyperparameters or {}
    nam = NAM(
        num_inputs=num_inputs,
        num_units=num_units_list,
        dropout=hp.get('dropout', 0.0),
        feature_dropout=hp.get('feature_dropout', 0.0),
        activation=hp.get('activation', 'relu'),
        shallow=hp.get('shallow', False),
        trainable=False,
        name_scope='model_0'
    )
    _ = nam(np.zeros((1, num_inputs), np.float32), training=False)

    # --- Restore weights ---
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)
    print("✅ Restored NAM from checkpoint.")

    return nam, sess


def partition(lst, batch_size):
    lst_len = len(lst)
    index = 0
    while index < lst_len:
        yield lst[index: batch_size + index]
        index += batch_size


def generate_predictions(gen, nn_model, sess):
    """Run predictions batch-by-batch inside a TF1 session."""
    y_pred = []
    while True:
        try:
            x = next(gen)
            pred = sess.run(nn_model(x, training=False))
            y_pred.extend(pred)
        except StopIteration:
            break
    return np.array(y_pred)


def get_test_predictions(nn_model, x_test, sess, batch_size=1024):
    num_samples = x_test.shape[0]
    preds = []
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch = x_test[start:end]
        preds.append(sess.run(nn_model(batch, training=False)))
    return np.concatenate(preds, axis=0)


def get_feature_predictions(nn_model, dataset_name, sess, chunk_size=50000):
    """Compute feature predictions for all unique values safely in chunks."""
    unique_features = compute_features(dataset_name)
    feature_predictions = []

    for c, vals in enumerate(unique_features):
        preds_all = []
        n = vals.shape[0]
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch = vals[start:end]
            preds = sess.run(nn_model.feature_nns[c](batch, training=nn_model._false))
            preds_all.append(preds)
        feature_predictions.append(np.concatenate(preds_all, axis=0))
    return feature_predictions


def compute_features(dataset_name):
    x_data, _, _ = data_utils.load_dataset(dataset_name)
    n_features = x_data.shape[1]
    unique_features = []

    for i in range(n_features):
        col = np.ascontiguousarray(x_data[:, i])
        # Sort first, then unique -> less memory than np.unique on unsorted
        col.sort()
        uniq = np.unique(col)
        unique_features.append(uniq.reshape(-1, 1))
    return unique_features


def prepare_feature_arrays(data_x, column_names, col_min_max, inverse_min_max_scaler):
    """Split scaled features and inverse transform to original space."""
    num_features = data_x.shape[1]
    single_features = np.split(data_x, num_features, axis=1)
    unique_features = [np.unique(x, axis=0) for x in single_features]

    single_features_original = {}
    unique_features_original = {}

    for i, col in enumerate(column_names):
        min_val, max_val = col_min_max[col]
        unique_features_original[col] = inverse_min_max_scaler(unique_features[i][:, 0], min_val, max_val)
        single_features_original[col] = inverse_min_max_scaler(single_features[i][:, 0], min_val, max_val)

    return single_features_original, unique_features_original, unique_features


def get_dataset_config(dataset_name, column_names):
    """Return COL_NAMES, FEATURE_LABEL_MAPPING, and CATEGORICAL_NAMES for dataset."""
    FEATURE_LABEL_MAPPING = {
        'Recidivism': {
            'race': (['African\nAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Native\nAmerican', 'Other'], 90),
            'sex': (['Female', 'Male'], None)
        },
        'Mimic2': {
            'AIDS': (['No', 'Yes'], None),
            'Lymphoma': (['No', 'Yes'], None),
            'MetastaticCancer': (['No', 'Yes'], None)
        },
        'Fico': {},
        'Housing': {},
        'Correlated_linear': {},
        'Correlated_nonlinear': {},
        'Credit':{},
        'Synthetic':{}
    }

    COL_NAMES = {
        'Recidivism': {
            'age': 'Age', 'race': 'Race', 'sex': 'Gender',
            'priors_count': 'Prior Counts', 'length_of_stay': 'Length of Stay',
            'c_charge_degree': 'Charge Degree'
        },
        'Housing': {
            'MedInc': 'Median Income', 'HouseAge': 'Median House Age',
            'AveRooms': '# Avg Rooms', 'AveBedrms': '# Avg Bedrooms',
            'Population': 'Block Population', 'AveOccup': '# Avg Occupancy',
            'Latitude': 'Latitude', 'Longitude': 'Longitude'
        },
        'Fico':  {
            'MSinceOldestTradeOpen': 'Months Since Oldest Trade Open',
            'MSinceMostRecentTradeOpen':	'Months Since Most Recent Trade',
            'AverageMInFile':	'Average Months in File',
            'NumSatisfactoryTrades': '# Satisfactory Trades',	
            'NumTrades60Ever2DerogPubRec': '# Trades 60+ Ever',	
            'NumTrades90Ever2DerogPubRec':	'# Trades 90+ Ever',	
            'NumTotalTrades': '# Total Trades',
            'NumTradesOpeninLast12M': '# Trades Open in Last 12 Months',
            'PercentTradesNeverDelq':	'% Trades Never Delinquent',
            'MSinceMostRecentDelq':	'Months Since Most Recent Delinquency',	
            'MaxDelq2PublicRecLast12M':	'Max Delq/Public Records Last Year',
            'MaxDelqEver':	'Max Delinquency Ever',
            'PercentInstallTrades':	'% Installment Trades',	
            'NetFractionInstallBurden':	'Net Fraction Installment Burden',
            'NumInstallTradesWBalance': 'Number Installment Trades with Balance',	
            'MSinceMostRecentInqexcl7days':	'Months Since Most Recent Inquiry\n excluding 7 days',	
            'NumInqLast6M': '# Inquiries in Last 6 Months',
            'NumInqLast6Mexcl7days': '# Inquiries in Last 6 Months \n excluding 7 days',
            'NetFractionRevolvingBurden':	'Net Fraction Revolving Burden',
            'NumRevolvingTradesWBalance':	'# Revolving Trades with Balance',	
            'NumBank2NatlTradesWHighUtilization':	'# Bank/Natl Trades with high utilization ratio',	
            'PercentTradesWBalance': '% Trades with Balance',
            'delinquent': 'Delinquent',
            'inquiry': 'Inquiry',
        }
    }

    if dataset_name in ['Credit', 'Mimic2', 'Correlated_linear', 'Correlated_nonlinear', 'Synthetic']:
        COL_NAMES[dataset_name] = {x: x for x in column_names}

    if dataset_name in ['Housing', 'Credit', 'Correlated_linear', 'Correlated_nonlinear', 'Synthetic']:
        categorical_names = []
    elif dataset_name == 'Mimic2':
        categorical_names = ['AIDS','AdmissionType','GCS','Lymphoma','Temperature','MetastaticCancer','Renal']
    elif dataset_name == 'Recidivism':
        categorical_names = ['race','sex','c_charge_degree']
    elif dataset_name == 'Fico':
        categorical_names = ['delinquent','inquiry','MaxDelqEver','MaxDelq2PublicRecLast12M']
    else:
        raise ValueError(f"{dataset_name} not found!")

    return COL_NAMES, FEATURE_LABEL_MAPPING, categorical_names



def compute_mean_predictions(data_x, column_names, unique_features, feature_predictions):
    """Compute index alignment and mean bias per feature."""
    avg_hist_data = {col: pred for col, pred in zip(column_names, feature_predictions)}
    all_indices, mean_pred = {}, {}

    for i, col in enumerate(column_names):
        x_i = data_x[:, i]
        all_indices[col] = np.searchsorted(unique_features[i][:, 0], x_i, 'left')

    for col in column_names:
        mean_pred[col] = np.mean([avg_hist_data[col][i] for i in all_indices[col]])

    return avg_hist_data, mean_pred, all_indices


def compute_mean_feature_importance(avg_hist_data, mean_pred):
    mean_abs_score = {}
    for feature, contribs in avg_hist_data.items():
        mean_abs_score[feature] = np.mean(np.abs(contribs - mean_pred[feature]))
    
    feature_names, mean_importances = zip(*mean_abs_score.items())
    return np.array(feature_names), np.array(mean_importances)



def plot_mean_feature_importance(feature_names, mean_importances, dataset_name, width=0.4, horizontal=False):
    sorted_idx = np.argsort(mean_importances)
    sorted_names = np.array(feature_names)[sorted_idx]
    sorted_values = mean_importances[sorted_idx]

    plt.figure(figsize=(7, 5))
    
    if horizontal:
        plt.barh(sorted_names, sorted_values, height=width, edgecolor='k')
        plt.xlabel("Mean Absolute Contribution", fontsize='x-large')
        plt.ylabel("Feature", fontsize='x-large')
    else:
        ind = np.arange(len(sorted_names))
        plt.bar(ind, sorted_values, width, edgecolor='k')
        plt.xticks(ind, sorted_names, rotation=90, fontsize='large')
        plt.ylabel("Mean Absolute Contribution", fontsize='x-large')
    
    plt.title(f"Feature Importance — {dataset_name}", fontsize='x-large', pad=10)
    plt.tight_layout()
    plt.show()


def plot_all_hist(hist_data, num_rows, num_cols, color_base, mean_pred,
                  unique_features, categorical_names, col_mapping,
                  feature_mapping, dataset_label='Feature Contribution',
                  linewidth=3.0, alpha=1.0, feature_to_use=None,
                  ymin=None, ymax=None, x_limits=None, y_limits=None):

    # detect multi-model input
    if isinstance(hist_data, dict):
        hist_list = [hist_data]
        mean_list = [mean_pred]
        first_hist = hist_data
    else:
        hist_list = hist_data
        mean_list = mean_pred
        first_hist = hist_data[0]

    hist_data_pairs = sorted(first_hist.items(), key=lambda x: x[0])

    if feature_to_use:
        hist_data_pairs = [pair for pair in hist_data_pairs if pair[0] in feature_to_use]

    # plot each feature
    for i, (name, _) in enumerate(hist_data_pairs):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        x = unique_features[name]

        # plot individual model curves
        for h, m in zip(hist_list, mean_list):
            pred = h[name] - m[name]
            if name in categorical_names:
                x_round = np.round(x, 1)
                x_plot = x_round - 0.5
                x_plot[-1] += 1
                step_loc = "mid" if len(x_round) <= 2 else "post"
                ax.step(x_plot, pred, color=color_base, alpha=0.1,
                        linewidth=1, where=step_loc)
            else:
                ax.plot(x, pred, color=color_base, alpha=0.1, linewidth=1)

        # plot average curve
        avg_curve = np.mean([h[name] - m[name] for h, m in zip(hist_list, mean_list)], axis=0)

        if name in categorical_names:
            ax.step(x_plot, avg_curve, color=color_base, linewidth=3, where=step_loc)
            labels, rot = feature_mapping.get(name, (x_round, None))
            ax.set_xticks(x_round)
            ax.set_xticklabels(labels, rotation=rot, fontsize='large')
        else:
            ax.plot(x, avg_curve, color=color_base, linewidth=3)
            ax.tick_params(labelsize='large')

        if y_limits is not None and name in y_limits:
            feature_ymin, feature_ymax = y_limits[name]
            ax.set_ylim(feature_ymin, feature_ymax)
        else:
            ax.set_ylim(ymin, ymax)

        if x_limits is not None and name in x_limits:
            x_limit_val = x_limits[name]
            if isinstance(x_limit_val, (tuple, list)) and len(x_limit_val) == 2:
                min_x, max_x = x_limit_val
            else:
                min_x = np.min(x)
                max_x = x_limit_val
            if name in categorical_names:
                min_x -= 0.5
                max_x += 0.5
            ax.set_xlim(min_x, max_x)
        else:
            min_x, max_x = np.min(x), np.max(x)
            if name in categorical_names:
                min_x -= 0.5
                max_x += 0.5
            ax.set_xlim(min_x, max_x)

        if i % num_cols == 0:
            ax.set_ylabel(dataset_label, fontsize='x-large')
        ax.set_xlabel(col_mapping.get(name, name), fontsize='x-large')

    return ymin, ymax

def shade_by_density_blocks(hist_data, unique_features, single_features,
                            n_blocks=5, color=(0.9, 0.5, 0.5),
                            categorical_names=None, feature_to_use=None,
                            ymin=None, ymax=None, x_limits=None, y_limits=None):

    fig = plt.gcf()
    axes = fig.get_axes()

    hist_data_pairs = sorted(hist_data.items(), key=lambda x: x[0])
    if feature_to_use:
        hist_data_pairs = [v for v in hist_data_pairs if v[0] in feature_to_use]

    for i, (name, _) in enumerate(hist_data_pairs):
        ax = axes[i]
        x = unique_features[name]
        data = single_features[name]

        if x_limits is not None and name in x_limits:

            x_limit_val = x_limits[name]
            if isinstance(x_limit_val, (tuple, list)) and len(x_limit_val) == 2:

                min_x_orig, max_x_orig = x_limit_val
            else:
                min_x_orig = np.min(x)
                max_x_orig = x_limit_val
        else:
            # Use data limits
            min_x_orig, max_x_orig = np.min(x), np.max(x)
        
        if categorical_names and name in categorical_names:
            min_x = min_x_orig - 0.5
            max_x = max_x_orig + 0.5
        else:
            min_x = min_x_orig
            max_x = max_x_orig
        
        if y_limits is not None and name in y_limits:
            feature_ymin, feature_ymax = y_limits[name]
        else:
            feature_ymin, feature_ymax = ymin, ymax

        data_filtered = data[(data >= min_x_orig) & (data <= max_x_orig)]
        
        if len(data_filtered) == 0:
            continue

        x_visible = x[(x >= min_x_orig) & (x <= max_x_orig)]
        x_n_blocks = min(n_blocks, max(len(x_visible), 1))
        
        range_size = max_x_orig - min_x_orig
        if range_size > 0 and range_size < 10:
            x_n_blocks = min(x_n_blocks, int(range_size) + 1)
        
        density, bin_edges = np.histogram(data_filtered, bins=x_n_blocks, range=(min_x, max_x))
        if np.max(density) > 0:
            density = density / np.max(density)
        else:
            density = np.zeros(x_n_blocks)

        for p in range(x_n_blocks):
            start = bin_edges[p]
            end = bin_edges[p + 1]
            alpha = min(1.0, 0.01 + density[p])

            rect = patches.Rectangle(
                (start, feature_ymin),
                end - start,
                feature_ymax - feature_ymin,
                facecolor=color,
                edgecolor=color,
                linewidth=0,
                alpha=alpha
            )
            ax.add_patch(rect)

def plot_nam_contributions_with_density(
    hist_data,
    unique_features,
    single_features,
    categorical_names,
    col_mapping,
    feature_mapping,
    mean_pred,
    feature_to_use=None,
    colors=None,
    n_blocks=20,
    num_cols=4,
    figsize_scale=4.5,
    dataset_label="Feature Contribution",
    return_limits=False,
    x_limits=None,
    y_limits=None
):
    if colors is None:
        colors = [[0.9, 0.4, 0.5], [0.5, 0.9, 0.4], [0.4, 0.5, 0.9], [0.9, 0.5, 0.9]]

    num_features = len(hist_data) if feature_to_use is None else len(feature_to_use)
    num_rows = int(np.ceil(num_features / num_cols))

    # build figure
    fig = plt.figure(
        figsize=(num_cols * figsize_scale, num_rows * figsize_scale),
        facecolor='w',
        edgecolor='k'
    )

    # detect single or multi-model
    if isinstance(hist_data, dict):
        hist_list = [hist_data]
        mean_list = [mean_pred]
    else:
        hist_list = hist_data
        mean_list = mean_pred

    # ---- Compute unified y limits ----
    global_vals = []
    for h, m in zip(hist_list, mean_list):
        for name in h:
            global_vals.append(h[name] - m[name])
    global_vals = np.concatenate(global_vals)

    base_min = np.min(global_vals)
    base_max = np.max(global_vals)

    ymin = base_min - 1
    ymax = base_max + 1

    # ---- plot curves ----
    plot_all_hist(
        hist_data=hist_data,
        num_rows=num_rows,
        num_cols=num_cols,
        color_base=colors[2],
        mean_pred=mean_pred,
        unique_features=unique_features,
        categorical_names=categorical_names,
        col_mapping=col_mapping,
        feature_mapping=feature_mapping,
        dataset_label=dataset_label,
        feature_to_use=feature_to_use,
        ymin=ymin,
        ymax=ymax,
        x_limits=x_limits,
        y_limits=y_limits
    )

    # ---- shading ----
    shade_by_density_blocks(
        hist_data=hist_data[0] if isinstance(hist_data, list) else hist_data,
        unique_features=unique_features,
        single_features=single_features,
        n_blocks=n_blocks,
        color=colors[0],
        categorical_names=categorical_names,
        feature_to_use=feature_to_use,
        ymin=ymin,
        ymax=ymax,
        x_limits=x_limits,
        y_limits=y_limits
    )

    plt.subplots_adjust(hspace=0.25)
    plt.show()

    if return_limits:
        return fig, (ymin, ymax)
    return fig
