import argparse
import json
import os

from torch_geometric.datasets import Planetoid

from link_prediction import link_prediction
from data_utils import link_pred_transform


def get_configs_from_file(filename):

    with open(filename) as f:
        config_file = json.load(f)

    params = {}   # param_name : list of possible values for changing parameters 
    base_config = {}
    for key, val in config_file.items():
        if type(val) != list or key == 'layers':
            base_config[key] = val
        else:
            params[key] = val
            base_config[key] = None
    
    n_configs = 1
    for key, val in params.items():
        n_configs *= len(val)

    period = n_configs
    configs = [base_config.copy() for _ in range(n_configs)]
    for key, val_list in params.items():
        period = period // len(val_list)
        for i, config in enumerate(configs):
            config[key] = val_list[(i // period) % len(val_list)]
            
    return configs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model')
    parser.add_argument('--dataset-name', dest='dataset_name')
    parser.add_argument('--data-root', dest='data_root')
    parser.add_argument('--trials-per-conf', dest='n_trials', default=5, type=int)
    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--MLP-config-file', dest='config_MLP_file')
    parser.add_argument('--result-folder', dest='result_folder', default='RESULTS')
    parser.add_argument('--final-training-runs', dest='final_training_runs', default=3, type=int)
    return parser.parse_args()


if __name__ == '__main__':

    transform = link_pred_transform

    args = get_args()

    model = args.model
    if model != 'CGMM' and model != 'ECGMM':
        raise ValueError('You must use CGMM or ECGMM as model')

    data_root = args.data_root
    dataset_class = Planetoid
    dataset_name = args.dataset_name

    output_dir = args.result_folder + f'_{model}_{dataset_name}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    n_trials = args.n_trials

    final_training_runs = args.final_training_runs

    model_configs = get_configs_from_file(args.config_file)
    MLP_configs = get_configs_from_file(args.config_MLP_file)
    
    link_prediction(model, dataset_class, data_root, dataset_name, transform, model_configs, MLP_configs, output_dir, n_trials, final_training_runs)
    
    '''
    # Needed to avoid thread spawning, conflicts with multi-processing. You may set a number > 1 but take into account
    # the number of processes on the machine
    torch.set_num_threads(1)
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=25)
    pool.submit(link_prediction, dataset_class, data_root, dataset_name, transform, model_configs, MLP_configs, out_dir)
    '''