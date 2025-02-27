# plotting.py

import os
import json
import ast
import streamlit as st
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from .utils import plot


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.

    Parameters:
    d (dict): The dictionary to flatten.
    parent_key (str): The base key string for the flattened keys.
    sep (str): Separator to use between keys.

    Returns:
    dict: A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def process_algorithm(dataset_path):
    """
    Process the dataset to flatten records.

    Parameters:
    dataset_path (str): Path to the dataset.

    Returns:
    list: List of flattened records.
    """
    return [flatten_dict(d) for d in load_records(dataset_path)]


def process_dataset(result_dirs, dataset_):
    """
    Process a specific dataset by loading and flattening its records.

    Parameters:
    result_dirs (str): Directory containing the results.
    dataset_ (str): Name of the dataset.

    Returns:
    tuple: Dataset name and list of records.
    """
    dataset = dataset_.replace('_noaug', '').replace('-Balanced', '')
    dataset_path = os.path.join(result_dirs, dataset_)
    records = process_algorithm(dataset_path)
    print(f"{dataset_} done -- {len(records)} records!")
    return dataset, records


def load_records(path):
    """
    Load records from JSON files in the specified directory.

    Parameters:
    path (str): Path to the directory containing JSON files.

    Returns:
    list: List of records.
    """
    records = []
    for root, _, files in os.walk(path):
        json_files = [f for f in files if f.endswith('.json') or f.endswith('.jsonl')]
        for file_name in json_files:
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        record = json.loads(line.strip())
                        record['args']['dataset'] = record['args']['dataset'].replace('_noaug', '').replace('-Classical', '')
                        if record['args']['algorithm'] != 'ERM':
                            continue
                        records.append(record)
            except IOError:
                pass
    return records


def extract_metric(str_dict):
    """
    Extract the specified metric from a string representation of a dictionary.

    Parameters:
    str_dict (str): String representation of a dictionary.

    Returns:
    dict: Extracted dictionary or None if extraction fails.
    """
    if isinstance(str_dict, str):
        try:
            converted_dict = ast.literal_eval(str_dict)
            return converted_dict
        except (ValueError, SyntaxError) as e:
            print(f"Error evaluating '{str_dict}': {e}")
            return None
    return None


@st.cache_data
def load_data():
    """
    Load data from the results directory using parallel processing.

    Returns:
    dict: Dictionary of dataframes for each dataset.
    """
    results_dir = "./results/ood_bench_results"
    records_dict = {}

    with ProcessPoolExecutor() as executor:
        datasets = [d for d in os.listdir(results_dir) if d != ".git" and d != "ood_bench_figures" and os.path.isdir(os.path.join(results_dir, d))]
        results = executor.map(process_dataset, [results_dir] * len(datasets), datasets)
        for dataset, records in results:
            if dataset not in records_dict:
                records_dict[dataset] = records
            else:
                records_dict[dataset] += records
    
    records_df = {k: pd.DataFrame(v) for k, v in records_dict.items() if v}
    return records_df


@st.cache_data
def load_parsed_data():
    """
    Load parsed data from CSV files in the data directory.

    Returns:
    dict: Dictionary of dataframes for each dataset.
    """
    results_dir = "./data"
    records_dict = {}

    for dataset in os.listdir(results_dir):
        dataset_path = os.path.join(results_dir, dataset)
        if os.path.isdir(dataset_path):
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            if csv_files:
                df_list = [pd.read_csv(os.path.join(dataset_path, csv_file)) for csv_file in csv_files]
                records_dict[dataset] = pd.concat(df_list, ignore_index=True)

    return records_dict


def run_real_data_plotting():
    """
    Main function to run the Streamlit app for plotting real data.
    """
    st.title("Domain Generalization Benchmarks with Accuracy on the Line")

    # Load data
    records_df = load_parsed_data()

    # Sidebar parameters
    st.sidebar.header("Parameters")

    # Selected Dataset
    selected_dataset = st.sidebar.selectbox("Dataset", sorted(list(records_df.keys())))

    # Selected Axis Scaling
    scaling = st.sidebar.selectbox("Axis Scaling", ['Probit', 'Linear', 'Square Root', 'Logit'])

    # Selected Left-Out Domain
    dataset_df = records_df[selected_dataset]
    dataset_df['model_arch'] = dataset_df['model_hparams'].apply(lambda x: extract_metric(x).get('model_arch', 'None'))
    dataset_df['transfer'] = dataset_df['model_hparams'].apply(lambda x: extract_metric(x).get('transfer', 'None'))

    test_envs = dataset_df.test_env.values.tolist()
    test_env = st.sidebar.selectbox("Left Out Test Domain #", sorted(set(test_envs)))

    # Filtered View on Environments
    filtered_envs = st.sidebar.multiselect("Filtered Environment #", sorted(set(test_envs)), sorted(set(test_envs)))

    # Filter Based Model Arch
    filtered_model_arch = st.sidebar.multiselect("Filtered Model Architecture", sorted(set(dataset_df['model_arch'].values)), sorted(set(dataset_df['model_arch'].values)))

    # Filter Based Transfer Learning
    filtered_transfer = st.sidebar.toggle("Show Models with Transfer Learning", value=True)

    # Filter Based on Model Configuration
    dataset_df = dataset_df[dataset_df['test_env'] == test_env]
    dataset_df = dataset_df[dataset_df['model_arch'].isin(filtered_model_arch)]
    dataset_df = dataset_df[dataset_df['train_env'].isin(filtered_envs)]
    if not filtered_transfer:
        # Remove those with transfer learning
        dataset_df = dataset_df[dataset_df['transfer'] != True]

    # Remove linear fit lines with toggle
    linear_fit_toggle = st.sidebar.toggle("Show Linear Fit Lines", value=True)

    # Change legend to model architecture
    legend_change = st.sidebar.toggle("Change Plot Legend to Model Architecture", value=False)

    # Plotting
    if dataset_df.shape[0]:
        plotting_df = pd.DataFrame(data={
            'x': dataset_df['x'].values,
            'y': dataset_df['y'].values,
            'train_env': dataset_df['train_env'].values,
            'test_env': dataset_df['test_env'].values,
            'algorithm': dataset_df['algorithm'].values,
            'model_hparams': dataset_df['model_hparams'].values,
            'model_arch': dataset_df['model_arch'].values,
            'transfer': dataset_df['transfer'].values,
        })
        full_plot, env_df = plot(plotting_df, scaling=scaling, show_linear_fits=linear_fit_toggle, legend_change=legend_change)
        st.plotly_chart(full_plot)
        st.write(env_df)
    else:
        st.write("### No data to plot. Please reselect in filtered environment dropdown.")
