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


def load_col_min_max(dataset_name):
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
  elif dataset_name == 'Correlated':
    dataset = data_utils.load_correlated_data()
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



def load_nam_checkpoint(ckpt_dir: str):
    """
    Load a NAM (Neural Additive Model) from a TensorFlow v1 checkpoint directory.

    Args:
        ckpt_dir (str): Path to the checkpoint directory containing .index and .data files.

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

    # --- Build the model ---
    tf.reset_default_graph()
    nam = NAM(
        num_inputs=num_inputs,
        num_units=num_units_list,
        dropout=0.0,
        feature_dropout=0.0,
        activation='relu',
        shallow=False,
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
        'Correlated': {},
        'Credit':{}
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

    if dataset_name in ['Credit', 'Mimic2', 'Correlated']:
        COL_NAMES[dataset_name] = {x: x for x in column_names}

    if dataset_name in ['Housing', 'Credit', 'Correlated']:
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


def shade_by_density_blocks(hist_data, unique_features, single_features, 
                            num_rows, num_cols, n_blocks=5, color=(0.9, 0.5, 0.5),
                            categorical_names=None, feature_to_use=None, fig=None):

    hist_data_pairs = sorted(hist_data.items(), key=lambda x: x[0])
    min_y = np.min([np.min(a[1]) for a in hist_data_pairs])
    max_y = np.max([np.max(a[1]) for a in hist_data_pairs])
    min_y -= 0.01 * (max_y - min_y)
    max_y += 0.01 * (max_y - min_y)

    if feature_to_use:
        hist_data_pairs = [v for v in hist_data_pairs if v[0] in feature_to_use]

    # Get all axes from the figure to reuse
    if fig is None:
        fig = plt.gcf()
    axes = fig.get_axes()

    for i, (name, pred) in enumerate(hist_data_pairs):
        ax = axes[i]  # ✅ reuse existing subplot, don’t create new one
        unique_x_data = unique_features[name]
        single_feature_data = single_features[name]
        min_x, max_x = np.min(unique_x_data), np.max(unique_x_data)
        if categorical_names and name in categorical_names:
            min_x -= 0.5
            max_x += 0.5

        x_n_blocks = min(n_blocks, len(unique_x_data))
        segments = (max_x - min_x) / x_n_blocks
        density = np.histogram(single_feature_data, bins=x_n_blocks)
        normed_density = density[0] / np.max(density[0])

        for p in range(x_n_blocks):
            start_x = min_x + segments * p
            end_x = min_x + segments * (p + 1)
            alpha = min(1.0, 0.01 + normed_density[p])
            rect = patches.Rectangle((start_x, min_y - 1),
                                     end_x - start_x,
                                     max_y - min_y + 1,
                                     linewidth=0,
                                     edgecolor=color,
                                     facecolor=color,
                                     alpha=alpha)
            ax.add_patch(rect)



def plot_all_hist(hist_data, num_rows, num_cols, color_base, mean_pred,
                  unique_features, categorical_names, col_mapping,
                  feature_mapping, dataset_label='Feature Contribution',
                  linewidth=3.0, alpha=1.0, feature_to_use=None):

    hist_data_pairs = sorted(hist_data.items(), key=lambda x: x[0])
    min_y = np.min([np.min(a) for _, a in hist_data_pairs])
    max_y = np.max([np.max(a) for _, a in hist_data_pairs])
    min_y -= 0.01 * (max_y - min_y)
    max_y += 0.01 * (max_y - min_y)

    if feature_to_use:
        hist_data_pairs = [v for v in hist_data_pairs if v[0] in feature_to_use]

    for i, (name, pred) in enumerate(hist_data_pairs):
        mean_val = mean_pred.get(name, np.mean(pred))
        unique_x_data = unique_features[name]
        ax = plt.subplot(num_rows, num_cols, i + 1)

        if name in categorical_names:
            unique_x_data = np.round(unique_x_data, 1)
            step_loc = "mid" if len(unique_x_data) <= 2 else "post"
            unique_plot_data = np.array(unique_x_data) - 0.5
            unique_plot_data[-1] += 1
            ax.step(unique_plot_data, pred - mean_val, color=color_base,
                    linewidth=linewidth, where=step_loc, alpha=alpha)
            labels, rot = feature_mapping.get(name, (unique_x_data, None))
            ax.set_xticks(unique_x_data)
            ax.set_xticklabels(labels, rotation=rot, fontsize='x-large')
        else:
            ax.plot(unique_x_data, pred - mean_val, color=color_base,
                    linewidth=linewidth, alpha=alpha)
            ax.tick_params(labelsize='x-large')

        ax.set_ylim(min_y, max_y)
        min_x, max_x = np.min(unique_x_data), np.max(unique_x_data)
        if name in categorical_names:
            min_x -= 0.5
            max_x += 0.5
        ax.set_xlim(min_x, max_x)
        if i % num_cols == 0:
            ax.set_ylabel(dataset_label, fontsize='x-large')
        ax.set_xlabel(col_mapping.get(name, name), fontsize='x-large')

    return min_y, max_y


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
    return_limits=False
):

    if colors is None:
        colors = [[0.9, 0.4, 0.5], [0.5, 0.9, 0.4], [0.4, 0.5, 0.9], [0.9, 0.5, 0.9]]

    num_features = len(hist_data) if feature_to_use is None else len(feature_to_use)
    num_rows = int(np.ceil(num_features / num_cols))

    fig = plt.figure(
        figsize=(num_cols * figsize_scale, num_rows * figsize_scale),
        facecolor='w', edgecolor='k'
    )

    # Plot feature curves first
    min_y, max_y = plot_all_hist(
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
    )

    # Overlay density shading
    shade_by_density_blocks(
        hist_data=hist_data,
        unique_features=unique_features,
        single_features=single_features,
        num_rows=num_rows,
        num_cols=num_cols,
        n_blocks=n_blocks,
        color=colors[0],
        categorical_names=categorical_names,
        feature_to_use=feature_to_use,
        fig=fig
    )

    plt.subplots_adjust(hspace=0.25)
    plt.show()

    if return_limits:
        return fig, (min_y, max_y)
    return fig
