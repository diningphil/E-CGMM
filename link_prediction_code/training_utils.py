from cgmm_layer import CGMMLayer
from data_utils import *

import os
import torch
from torch_scatter import scatter_max
import numpy as np
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def aggregate_edge_post(edge_posterior, pos_edge_index, neg_edge_index, num_nodes):
    
    n_pos_edge, n_neg_edge = pos_edge_index.shape[1], neg_edge_index.shape[1]
    assert edge_posterior.shape[0] == n_pos_edge + n_neg_edge

    pos_posterior = edge_posterior[:n_pos_edge]
    neg_posterior = edge_posterior[n_pos_edge:]

    pos_adj_list = [[0, []] for _ in range(num_nodes)]
    pos_upper_emb = torch.empty((int(pos_posterior.shape[0] / 2), pos_posterior.shape[1]))
    pos_lower_emb = torch.empty((int(pos_posterior.shape[0] / 2), pos_posterior.shape[1]))

    upper_idx = 0
    for i, (src, dest) in enumerate(pos_edge_index.t()):
        if dest > src:
            pos_upper_emb[upper_idx] = pos_posterior[i]
            upper_idx += 1
            pos_adj_list[src][0] += 1
            pos_adj_list[src][1].append(dest)
        else:
            idx = sum([el[0] for el in pos_adj_list[:dest]])
            idx += pos_adj_list[dest][1].index(src)
            pos_lower_emb[idx] = pos_posterior[i] 
    
    pos_aggr = (pos_upper_emb + pos_lower_emb) / 2

    neg_adj_list = [[0, []] for _ in range(num_nodes)]
    neg_upper_emb = torch.empty((int(neg_posterior.shape[0] / 2), neg_posterior.shape[1]))
    neg_lower_emb = torch.empty((int(neg_posterior.shape[0] / 2), neg_posterior.shape[1]))
    upper_idx = 0
    for i, (src, dest) in enumerate(neg_edge_index.t()):
        if dest > src:
            neg_upper_emb[upper_idx] = neg_posterior[i]
            upper_idx += 1
            neg_adj_list[src][0] += 1
            neg_adj_list[src][1].append(dest)
        else:
            idx = sum([el[0] for el in neg_adj_list[:dest]])
            idx += neg_adj_list[dest][1].index(src)
            neg_lower_emb[idx] = neg_posterior[i] 

    neg_aggr = (neg_upper_emb + neg_lower_emb) / 2

    return torch.cat((pos_aggr, neg_aggr), dim=0)


def compute_node_stat_ECGMM(node_posteriors, edge_posteriors, edge_index, C_e, C_n):

    statistics = torch.full((node_posteriors.shape[0], C_e + 2, C_n + 1), 1e-8, dtype=node_posteriors.dtype)

    srcs, dsts = edge_index

    for arc_label in range(C_e):
        sparse_label_adj_matr = torch.sparse_coo_tensor(edge_index, \
        edge_posteriors[:, arc_label], \
        torch.Size([node_posteriors.shape[0], node_posteriors.shape[0]]) ).transpose(0, 1)
        
        statistics[:, arc_label, :-1] = torch.sparse.mm(sparse_label_adj_matr, node_posteriors)

    # use self.C_e + 1 as special edge for bottom states (all in self.C_n)
    degrees = statistics[:, :, :-1].sum(dim=[1, 2]).floor()

    fake_batch = torch.zeros(node_posteriors.shape[0]).long()
    max_arieties, _ = scatter_max(degrees.int(), fake_batch)
    statistics[:, C_e + 1, C_n] += 1 - (degrees / max_arieties[fake_batch].double())
    return statistics


def compute_node_stat_CGMM(node_posteriors, edge_index, C_e, C_n):

    edge_posteriors = torch.ones((edge_index.shape[1], 1)).float() ###########
    node_posteriors = node_posteriors.float()

    # Copiato da GCGN
    statistics = torch.full((node_posteriors.shape[0], C_e + 2, C_n + 1), 1e-8, dtype=node_posteriors.dtype)

    srcs, dsts = edge_index

    for arc_label in range(C_e):
        sparse_label_adj_matr = torch.sparse_coo_tensor(edge_index, \
        edge_posteriors[:, arc_label], \
        torch.Size([node_posteriors.shape[0], node_posteriors.shape[0]]) ).transpose(0, 1)
        
        statistics[:, arc_label, :-1] = torch.sparse.mm(sparse_label_adj_matr, node_posteriors)

    # use self.C_e + 1 as special edge for bottom states (all in self.C_n)
    degrees = statistics[:, :, :-1].sum(dim=[1, 2]).floor()

    fake_batch = torch.zeros(node_posteriors.shape[0]).long()
    max_arieties, _ = scatter_max(degrees.int(), fake_batch)
    statistics[:, C_e + 1, C_n] += 1 - (degrees / max_arieties[fake_batch].double())
    return statistics


def compute_edge_stats(node_posteriors, edge_index):

    C_n = node_posteriors.shape[1]
    statistics = torch.full((edge_index.shape[1], 2, C_n), 1e-8, dtype=node_posteriors.dtype)

    srcs, dsts = edge_index
    statistics[:, 0, :] = node_posteriors[srcs]
    statistics[:, 1, :] = node_posteriors[dsts]
    return statistics


def compute_edge_posteriors(prev_node_posteriors, edge_layer, edge_index):
    n_arcs = edge_index.shape[1]
    src, dst = edge_index.long()
    transition = edge_layer.transition  # L x 2 x C_e x C_n
    layerS = edge_layer.layerS          # L
    arcS = edge_layer.arcS              # L x 2 
    # we want n_edges x L x 2 x C_e x C_n
    broad_transition = transition.unsqueeze(0)
    broad_layerS = layerS.reshape([1, -1, 1, 1, 1])
    broad_arcS = arcS.reshape([1, arcS.shape[0], arcS.shape[1], 1, 1])
    broad_node_post = torch.stack((prev_node_posteriors[src], prev_node_posteriors[dst]), dim=1).reshape(n_arcs, 1, 2, 1, prev_node_posteriors.shape[1])
    edge_posterior = torch.sum(broad_layerS * broad_arcS * broad_node_post * broad_transition, dim=[1, 2, 4])
        
    return edge_posterior


def train_unsupervised_ECGMM(data, config, out_dir, test=False):

    max_depth = config['max_depth']
    node_type = config['node_type']
    edge_type = config['edge_type']
    K_n = config['dim_node_features']
    K_e = config['dim_edge_features']
    C_n = config['C_n']
    C_e = config['C_e']
    L = config['L']
    max_epochs = config['max_epochs']
    bigram = config['bigram']
    use_cont_node = config['use_cont_node']
    use_cont_edge = config['use_cont_edge']

    node_statistics = []
    node_posteriors = []

    train_edge_posteriors = []
    train_edge_embeddings = []
    val_edge_posteriors = []
    val_edge_embeddings = []
    
    layers = []

    score_per_layer = []

    # data attributes: train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, 
    # test_pos_edge_index, and test_neg_edge_index,    and respective edge_attr

    train_edge_index = torch.cat((data.train_pos_edge_index, data.train_neg_edge_index), dim=1)
    train_edge_attr = torch.cat((data.train_pos_edge_attr, data.train_neg_edge_attr), dim=0)
    if not test:
        val_edge_index = torch.cat((data.val_pos_edge_index, data.val_neg_edge_index), dim=1)
        val_edge_attr = torch.cat((data.val_pos_edge_attr, data.val_neg_edge_attr), dim=0)
    else:
        val_edge_index = torch.cat((data.test_pos_edge_index, data.test_neg_edge_index), dim=1)
        val_edge_attr = torch.cat((data.test_pos_edge_attr, data.test_neg_edge_attr), dim=0)

    depth = 1
    while depth <= max_depth:
        print(f'Layer {depth}')

        is_first_layer = depth == 1
        node_layer = CGMMLayer(K_n, C_n, C_e + 2, C_n + 1, L, is_first_layer, node_type=node_type)

        if is_first_layer:
            prev_node_stats = None
        else:
            # just the previous layer!
            # we want ? x L x C_e x C_n  ---> unsqueeze(1)
            prev_node_stats = node_statistics[-1].unsqueeze(1)        
        
        # node layer training
        for epoch in range(1, max_epochs+1):
            node_likelihood, _ = node_layer.E_step(data.x, prev_node_stats, training=True)
            node_layer.M_step()
            if epoch == 1 or epoch == max_epochs:
                print(f'Epoch {epoch}, node likelihood = {node_likelihood}')
        
        # compute node posteriors
        _, node_posterior = node_layer.E_step(data.x, prev_node_stats, training=False)

        # infer node states, if use_cont_node is false
        if not use_cont_node:
            node_posterior = torch.where((node_posterior == torch.max(node_posterior, dim=1).values.reshape(-1, 1)), torch.tensor(1), torch.tensor(0)).double()

        node_posteriors.append(node_posterior)

        # compute "edge statistics" using node posteriors, for all edges
        # NB we want ? x L x 2 x C_n --> unsqueeze(1)
        edge_stat = compute_edge_stats(node_posterior, train_edge_index).unsqueeze(1).double()

        # NB: never in the first edge layer! 
        edge_layer = CGMMLayer(K_e, C_e, 2, C_n, L, False, node_type=edge_type)

        # edge layer training
        for epoch in range(1, max_epochs+1):
            edge_likelihood, _ = edge_layer.E_step(train_edge_attr, edge_stat, training=True)
            edge_layer.M_step()
            if epoch == 1 or epoch == max_epochs:
                print(f'Epoch {epoch}, edge likelihood = {edge_likelihood}')

        # compute edge posteriors
        # I can't use E-step because I must pretend to not know edge_attr
        train_edge_posterior = compute_edge_posteriors(node_posterior, edge_layer, train_edge_index)
        val_edge_posterior = compute_edge_posteriors(node_posterior, edge_layer, val_edge_index)

        # infer edge states, if use_cont_edge is false
        if not use_cont_edge:
            train_edge_posterior = torch.where((train_edge_posterior == torch.max(train_edge_posterior, dim=1).values.reshape(-1, 1)), torch.tensor(1), torch.tensor(0)).double()
            val_edge_posterior = torch.where((val_edge_posterior == torch.max(val_edge_posterior, dim=1).values.reshape(-1, 1)), torch.tensor(1), torch.tensor(0)).double()
        
        train_edge_posteriors.append(train_edge_posterior)
        val_edge_posteriors.append(val_edge_posterior)

        # compute node stats for next layers
        # I have to use only real train edges' posteriors, that are the first pos_train_edges
        real_train_edge_posterior = train_edge_posterior[:data.train_pos_edge_index.shape[1]]
        node_stat = compute_node_stat_ECGMM(node_posterior, real_train_edge_posterior, data.train_pos_edge_index, C_e, C_n)
        node_statistics.append(node_stat)

        # append layer's score
        score_per_layer.append((node_likelihood, edge_likelihood))
        # append layer
        layers.append((node_layer, edge_layer))    

        # embeddings
        # aggregate edge embeddings, edges are undirected!
        train_edge_embedding = aggregate_edge_post(train_edge_posterior, data.train_pos_edge_index, data.train_neg_edge_index, data.num_nodes)
        if not test:
            val_edge_embedding = aggregate_edge_post(val_edge_posterior, data.val_pos_edge_index, data.val_neg_edge_index, data.num_nodes)
        else:
            val_edge_embedding = aggregate_edge_post(val_edge_posterior, data.test_pos_edge_index, data.test_neg_edge_index, data.num_nodes)
            
        if bigram:
            train_bigram, val_bigram = None, None
            for src, dst in train_edge_index.t():
                if src > dst:
                    tr_edge_bigram = (node_posterior[src] + node_posterior[dst]) / 2
                    if train_bigram is None:
                        train_bigram = tr_edge_bigram.reshape(1, -1)
                    else:
                        train_bigram = torch.cat((train_bigram, tr_edge_bigram.reshape(1, -1)), dim=0)
            train_edge_embedding = torch.cat((train_edge_embedding, train_bigram.float()), dim=1)
            
            for src, dst in val_edge_index.t():
                if src > dst:
                    val_edge_bigram = (node_posterior[src] + node_posterior[dst]) / 2
                    if val_bigram is None:
                        val_bigram = val_edge_bigram.reshape(1, -1)
                    else:
                        val_bigram = torch.cat((val_bigram, val_edge_bigram.reshape(1, -1)), dim=0)
            val_edge_embedding = torch.cat((val_edge_embedding, val_bigram.float()), dim=1)
        

        train_edge_embeddings.append(train_edge_embedding)
        val_edge_embeddings.append(val_edge_embedding)

        depth += 1

    hs_layer_dir = os.path.join(out_dir, 'hidden_state_figures')
    if not os.path.exists(hs_layer_dir):
        os.mkdir(hs_layer_dir)
    trans_dir = os.path.join(out_dir, 'transition_figures')
    if not os.path.exists(trans_dir):
        os.mkdir(trans_dir)
    
    # ---compute and plot node and edge hidden states heatmaps ---
    node_matr = np.zeros((C_n, len(layers)))
    edge_matr = np.zeros((C_e, len(layers)))
    for idx, (_, edge_layer) in enumerate(layers):
        node_hs = torch.argmax(node_posteriors[idx], dim=1)
        edge_hs = torch.argmax(train_edge_posteriors[idx], dim=1)
        for hs_n in range(C_n):
            node_matr[hs_n, idx] = torch.sum(node_hs == hs_n).item() 
        for hs_e in range(C_e):
            edge_matr[hs_e, idx] = torch.sum(edge_hs == hs_e).item() 

    node_heat_map = sns.heatmap(node_matr, cmap="YlGnBu")
    plt.xlabel('Layers')
    plt.ylabel('Hidden state')
    plt.title('Node hidden states vs Number of layers')
    plt.savefig(os.path.join(hs_layer_dir, 'node_heatmap.png'))
    plt.close()

    edge_heat_map = sns.heatmap(edge_matr, cmap="YlGnBu")
    plt.xlabel('Layers')
    plt.ylabel('Hidden state')
    plt.title('Edge hidden states vs Number of layers')
    plt.savefig(os.path.join(hs_layer_dir, 'edge_heatmap.png'))
    plt.close()
    # ---------------------------------------------------------

    return train_edge_embeddings, val_edge_embeddings


def compute_edge_embeddings(node_posteriors, edge_index):
    
    edge_embeddings = torch.empty((int(edge_index.shape[1] / 2), node_posteriors.shape[1]))
    i = 0
    for src, dest in edge_index.t():
        if src < dest:
            edge_embeddings[i] = (node_posteriors[src] + node_posteriors[dest]) / 2
            i += 1
    return edge_embeddings


def train_unsupervised_CGMM(data, config, out_dir, test=False):

    max_depth = config['max_depth']
    node_type = config['node_type']
    K_n = config['dim_node_features']
    K_e = config['dim_edge_features']
    C_n = config['C_n']
    L = config['L']
    max_epochs = config['max_epochs']
    use_cont_node = config['use_cont_node']

    node_statistics = []
    node_posteriors = []

    train_edge_embeddings = []
    val_edge_embeddings = []
    
    layers = []

    score_per_layer = []

    # data attributes: train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, 
    # test_pos_edge_index, and test_neg_edge_index,    and respective edge_attr

    depth = 1
    while depth <= max_depth:
        print(f'Layer {depth}')

        is_first_layer = depth == 1
        node_layer = CGMMLayer(K_n, C_n, 1 + 2, C_n + 1, L, is_first_layer, node_type=node_type)

        if is_first_layer:
            prev_node_stats = None
        else:
            # just the previous layer!
            # we want ? x L x C_e x C_n (o simili) ---> unsqueeze(1)
            prev_node_stats = node_statistics[-1].unsqueeze(1)        
        
        # node layer training
        for epoch in range(1, max_epochs+1):
            node_likelihood, _ = node_layer.E_step(data.x, prev_node_stats, training=True) # se training è True aggiorna gli accumulatori
            node_layer.M_step()
            if epoch == 1 or epoch == max_epochs:
                print(f'Epoch {epoch}, node likelihood = {node_likelihood}')
        
        # compute node posteriors
        _, node_posterior = node_layer.E_step(data.x, prev_node_stats, training=False)

        # infer node states, if use_cont_node is false
        if not use_cont_node:
            node_posterior = torch.where((node_posterior == torch.max(node_posterior, dim=1).values.reshape(-1, 1)), torch.tensor(1), torch.tensor(0)).double()

        node_posteriors.append(node_posterior)

        # compute node stats for next layers
        # I have to use only real train edges' posteriors, that are the first pos_train_edges
        node_stat = compute_node_stat_CGMM(node_posterior, data.train_pos_edge_index, 1, C_n)
        node_statistics.append(node_stat)

        # compute edge embeddings
        # I can't use E-step because I must pretend to not know edge_attr
        train_pos_edge_emb = compute_edge_embeddings(node_posterior, data.train_pos_edge_index)
        train_neg_edge_emb = compute_edge_embeddings(node_posterior, data.train_neg_edge_index)
        if not test:
            val_pos_edge_emb = compute_edge_embeddings(node_posterior, data.val_pos_edge_index)
            val_neg_edge_emb = compute_edge_embeddings(node_posterior, data.val_neg_edge_index)
        else:
            val_pos_edge_emb = compute_edge_embeddings(node_posterior, data.test_pos_edge_index)
            val_neg_edge_emb = compute_edge_embeddings(node_posterior, data.test_neg_edge_index)

        train_edge_embeddings.append(torch.cat((train_pos_edge_emb, train_neg_edge_emb), dim=0))
        val_edge_embeddings.append(torch.cat((val_pos_edge_emb, val_neg_edge_emb), dim=0))

        # append layer's score
        score_per_layer.append(node_likelihood)
        # append layer
        layers.append(node_layer)    

        depth += 1

    hs_layer_dir = os.path.join(out_dir, 'hidden_state_figures')
    if not os.path.exists(hs_layer_dir):
        os.mkdir(hs_layer_dir)
    trans_dir = os.path.join(out_dir, 'transition_figures')
    if not os.path.exists(trans_dir):
        os.mkdir(trans_dir)
    
    # ---compute and plot node hidden states heatmaps ---
    node_matr = np.zeros((C_n, len(layers)))
    for idx, _ in enumerate(layers):
        node_hs = torch.argmax(node_posteriors[idx], dim=1)
        for hs_n in range(C_n):
            node_matr[hs_n, idx] = torch.sum(node_hs == hs_n).item() 
     
    node_heat_map = sns.heatmap(node_matr, cmap="YlGnBu")
    plt.xlabel('Layers')
    plt.ylabel('Hidden state')
    plt.title('Node hidden states vs Number of layers')
    plt.savefig(os.path.join(hs_layer_dir, 'node_heatmap.png'))
    plt.close()
    # ---------------------------------------------------------

    return train_edge_embeddings, val_edge_embeddings


def train_MLP(train_x, train_y, val_x, val_y, MLP_config):

    train_y = train_y.squeeze()
    val_y = val_y.squeeze()
    # already shuffled

    hidden_dim = MLP_config['hidden_dim']
    lr = MLP_config['lr']
    max_epochs_MLP = MLP_config['max_epochs_MLP']
    output_dim = MLP_config['output_dim']
    early_stopping = MLP_config['early_stopping'] 
    weight_decay = MLP_config['weight_decay'] 

    D_in, H, D_out = train_x.shape[1], hidden_dim, output_dim

    # MLP initialization
    if D_out == 2:
        # classification
        model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(inplace=False), torch.nn.Linear(H, 1), torch.nn.Sigmoid()).to('cpu')
    else:
        model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out), torch.nn.Softmax()).to('cpu')
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training MLP
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    
    best_val_acc, best_epoch = 0, 0
    patience = 0

    for epoch in range(max_epochs_MLP):
        optimizer.zero_grad()
        # compute output and loss
        train_y_pred = model(train_x).squeeze()
        val_y_pred = model(val_x).squeeze()
        train_loss = loss_fn(train_y_pred, train_y)
        train_loss_list.append(train_loss.item())
        val_loss = loss_fn(val_y_pred, val_y)
        val_loss_list.append(val_loss.item())
        # update weights
        train_loss.backward()
        optimizer.step()
        # compute predicted class
        train_y_class = torch.round(train_y_pred).detach().numpy()
        val_y_class = torch.round(val_y_pred).detach().numpy()
        # remove strange values
        train_y_class[~np.isfinite(train_y_class)] = 0
        val_y_class[~np.isfinite(val_y_class)] = 0
        # compute accuracy
        train_accuracy = accuracy_score(train_y.detach().numpy(), train_y_class)
        train_accuracy_list.append(train_accuracy)
        val_accuracy = accuracy_score(val_y.detach().numpy(), val_y_class)
        val_accuracy_list.append(val_accuracy)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        
        if patience > early_stopping:
            print('Finishing training due to patience')
            print(f'Best epoch: {best_epoch}     best accuracy: {best_val_acc}')
            return train_y_pred[:best_epoch+1], val_y_pred[:best_epoch+1], train_loss_list[:best_epoch+1], val_loss_list[:best_epoch+1], train_accuracy_list[:best_epoch+1], val_accuracy_list[:best_epoch+1]
        
        if epoch == 1 or epoch % 50 == 0:
            print(f'TRAIN: loss = {train_loss.item()}, accuracy = {train_accuracy}')
            print(f'VALID: loss = {val_loss.item()}, accuracy = {val_accuracy}')

    return train_y_pred, val_y_pred, train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list