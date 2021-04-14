import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from training_utils import *
from data_utils import *


def model_selection(model, graph, model_configs, MLP_configs, out_dir, n_trials):

    if model == 'ECGMM':
        train_unsupervised = train_unsupervised_ECGMM
    else:
        train_unsupervised = train_unsupervised_CGMM

    model_selection_results = {}

    for n_trial in range(n_trials):
        trial_dir = os.path.join(out_dir, f'trial_{n_trial}')
        if not os.path.exists(trial_dir):
            os.mkdir(trial_dir)

        graph = shuffle_train_val(graph)

        for i, model_config in enumerate(model_configs):
            print(f'Trying model config n {i} of {len(model_configs)-1}')

            model_config_dir = os.path.join(trial_dir, f'model_config_{i}')
            if not os.path.exists(model_config_dir):
                os.mkdir(model_config_dir)
            # save config
            model_config_file = open(os.path.join(model_config_dir, "model_config.txt"), "w")
            model_config_file.write(str(model_config))
            model_config_file.close()

            config_results = {}

            # get edge_embeddings
            train_edge_embeddings_list, val_edge_embeddings_list = train_unsupervised(graph, model_config, model_config_dir)  # list of n_layers tensors n_edges x embeddings_dim

            layers = model_config['layers']
            if type(layers) != list:
                layers = [layers]
                
            for layer in layers:
                print(f'Considering layer {layer+1}')

                layer_dir = os.path.join(model_config_dir, f'layer_{layer}')
                if not os.path.exists(layer_dir):
                    os.mkdir(layer_dir)
                
                layer_results = {}
                
                train_edge_embeddings = torch.cat(train_edge_embeddings_list[:layer+1], dim=1)
                val_edge_embeddings = torch.cat(val_edge_embeddings_list[:layer+1], dim=1)
                
                # MLP model selection
                train_x = train_edge_embeddings.float()
                val_x = val_edge_embeddings.float() 
                # just one graph in the dataset
                tr_pos_attr = graph.train_pos_edge_attr[:int(graph.train_pos_edge_attr.shape[0] / 2), :]
                tr_neg_attr = graph.train_neg_edge_attr[:int(graph.train_neg_edge_attr.shape[0] / 2), :]
                vl_pos_attr = graph.val_pos_edge_attr[:int(graph.val_pos_edge_attr.shape[0] / 2), :]
                vl_neg_attr = graph.val_neg_edge_attr[:int(graph.val_neg_edge_attr.shape[0] / 2), :]

                train_y = torch.cat((tr_pos_attr, tr_neg_attr), dim=0)
                val_y = torch.cat((vl_pos_attr, vl_neg_attr), dim=0)

                for j, MLP_config in enumerate(MLP_configs):
                    print(f'Trying MLP config n {j} of {len(MLP_configs)-1}')

                    MLP_config_dir = os.path.join(layer_dir, f'MLP_config_{j}')
                    if not os.path.exists(MLP_config_dir):
                        os.mkdir(MLP_config_dir)
                    # save MLP config
                    MLP_config_file = open(os.path.join(MLP_config_dir, "MLP_config.txt"), "w")
                    MLP_config_file.write(str(MLP_config))
                    MLP_config_file.close()

                    training_results = train_MLP(train_x, train_y, val_x, val_y, MLP_config)
                    train_y_pred, val_y_pred, train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list = training_results

                    # --- plot learning curves
                    plt.plot(range(1, len(train_loss_list)+1), train_loss_list)
                    plt.plot(range(1, len(val_loss_list)+1), val_loss_list)
                    plt.title('Learning curve')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend(['train', 'val'])
                    plt.savefig(os.path.join(MLP_config_dir, f'learning_curve_loss.png'))
                    plt.close()
                    plt.plot(range(1, len(train_accuracy_list)+1), train_accuracy_list)
                    plt.plot(range(1, len(val_accuracy_list)+1), val_accuracy_list)
                    plt.title('Learning curve')
                    plt.xlabel('Epochs')
                    plt.ylabel('Accuracy')
                    plt.legend(['train', 'val'])
                    plt.savefig(os.path.join(MLP_config_dir, f'learning_curve_acc.png'))
                    plt.close()
                    # ----
                    # save results
                    val_acc = val_accuracy_list[-1]
                    train_acc = train_accuracy_list[-1]
                    f = open(os.path.join(MLP_config_dir, "results.txt"), "a")
                    f.write(f'val accuracy = {val_acc}\n')
                    f.write(f'Corresponding train accuracy = {train_acc}\n')
                    f.write(str(train_accuracy_list) + '\n')
                    f.write(str(val_accuracy_list) + '\n')
                    f.close()

                    config_results[(layer, tuple(MLP_config.items()))] = val_acc
                    layer_results[tuple(MLP_config.items())] = val_acc
                    if (i, layer, j) not in model_selection_results.keys():
                        model_selection_results[(i, layer, j)] = []
                    model_selection_results[(i, layer, j)].append(val_acc)
                
                # save layer results
                layer_results_file = open(os.path.join(layer_dir, "results.txt"), "w")
                layer_results_file.write(str(layer_results))
                layer_results_file.close()
            # save config results
            config_results_file = open(os.path.join(model_config_dir, "results.txt"), "w")
            config_results_file.write(str(config_results))
            config_results_file.close()

    return model_selection_results