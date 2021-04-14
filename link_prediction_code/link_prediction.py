import os
import torch
import numpy as np

from model_selection import model_selection
from training_utils import train_unsupervised_CGMM, train_unsupervised_ECGMM, train_MLP


def link_prediction(model, dataset_class, data_root, dataset_name, transform, model_configs, MLP_configs, out_dir, n_trials, final_training_runs):
    
    dataset = dataset_class(data_root, dataset_name, transform=transform)
    graph = dataset[0]

    model_selection_results = model_selection(model, graph, model_configs, MLP_configs, out_dir, n_trials)
    best_acc, best_std = 0, 0
    best_config, best_layer, best_MLP_config = None, None, None

    if model == 'ECGMM':
        train_unsupervised = train_unsupervised_ECGMM
    else:
        train_unsupervised = train_unsupervised_CGMM

    for (i, layer, j), acc_list in model_selection_results.items():
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        if (mean_acc > best_acc) or (mean_acc == best_acc and std_acc < best_acc):
            best_acc = mean_acc
            best_std = std_acc
            best_config = model_configs[i]
            best_layer = layer 
            best_MLP_config = MLP_configs[j]
    
    best_config['max_depth'] = best_layer
    best_MLP_config['early_stopping'] = float('inf')
    
    for i in range(final_training_runs):
        print(f'Final training run {i+1} of {final_training_runs}')
        train_edge_embeddings_list, test_edge_embeddings_list = train_unsupervised(graph, best_config, out_dir, test=True)
        train_edge_embeddings = torch.cat(train_edge_embeddings_list, dim=1)
        test_edge_embeddings = torch.cat(test_edge_embeddings_list, dim=1)

        train_x = train_edge_embeddings.float()
        test_x = test_edge_embeddings.float() 

        tr_pos_attr = graph.train_pos_edge_attr[:int(graph.train_pos_edge_attr.shape[0] / 2), :]
        tr_neg_attr = graph.train_neg_edge_attr[:int(graph.train_neg_edge_attr.shape[0] / 2), :]
        vl_pos_attr = graph.val_pos_edge_attr[:int(graph.val_pos_edge_attr.shape[0] / 2), :]
        vl_neg_attr = graph.val_neg_edge_attr[:int(graph.val_neg_edge_attr.shape[0] / 2), :]
        test_pos_attr = graph.test_pos_edge_attr[:int(graph.test_pos_edge_attr.shape[0] / 2), :]
        test_neg_attr = graph.test_neg_edge_attr[:int(graph.test_neg_edge_attr.shape[0] / 2), :]

        train_y = torch.cat((tr_pos_attr, tr_neg_attr), dim=0)
        test_y = torch.cat((test_pos_attr, test_neg_attr), dim=0)

        training_results = train_MLP(train_x, train_y, test_x, test_y, best_MLP_config)
        train_y_pred, test_y_pred, train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list = training_results
        print(f'Test accuracy: {test_accuracy_list[-1]}')
        run_results_file = open(os.path.join(out_dir, f"results_{i}.txt"), "w")
        run_results_file.write(f'Test accuracy: {test_accuracy_list[-1]}')
        run_results_file.close()

    return