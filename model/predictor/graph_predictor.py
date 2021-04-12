import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Dropout, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from experiment.experiment import s2c
from model.predictor.probabilistic_readout import ProbabilisticReadout


class GraphPredictor(torch.nn.Module):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()
        self.dim_node_features = dim_node_features
        self.dim_edge_features = dim_edge_features
        self.dim_target = dim_target
        self.config = config

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        raise NotImplementedError('You need to implement this method!')


class ProbabilisticGraphReadout(ProbabilisticReadout):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.emission_class = s2c(config['emission'])
        self.CN = config['C']  # number of states of a generic node
        self.CS = config['CS']  # number of states of the supersource node

        self.emission = self.emission_class(self.Y, self.CS)
        self.readout_distr = torch.nn.Parameter(torch.empty(self.CS, self.CN,
                                                            dtype=torch.float32),
                                                            requires_grad=False)
        self.readout_numerator = torch.nn.Parameter(torch.empty_like(self.readout_distr),
                                                            requires_grad=False)
        # Initialize parameters
        for i in range(0, self.CN):
            ro = torch.nn.init.uniform_(torch.empty(self.CS,
                                                    dtype=torch.float32))
            self.readout_distr[:, i] = ro / ro.sum()

    def init_accumulators(self):
        self.emission.init_accumulators()
        torch.nn.init.constant_(self.readout_numerator, self.eps)

    def e_step(self, p_Q_given_obs, x_labels, y_labels, batch):
        # ?g x CS
        emission_target = self.emission.e_step(x_labels, y_labels)
        # ?g x CS x CN
        readout_posterior = torch.mul(emission_target.unsqueeze(2),
                                      self.readout_distr.unsqueeze(0))

        # true log P(y) using the observables
        # Mean of individual node terms
        tmp = global_mean_pool(torch.mm(p_Q_given_obs,
                                        self.readout_distr.transpose(1,0)),
                               batch)  # ?g x CS
        p_y = (emission_target*tmp).sum(dim=1)

        p_y[p_y==0.] = 1.
        true_log_likelihood = p_y.log().sum(dim=0)

        return true_log_likelihood, readout_posterior, emission_target

    def infer(self, p_Q_given_obs, x_labels, batch):

        '''
        # OLD CODE
        # ?n x CS
        p_QS = torch.mm(p_Q_given_obs, self.readout_distr.transpose(1,0))

        # ?g x CS
        p_QS = global_mean_pool(p_QS, batch)
        likely_graph_labels = self.emission.infer(p_QS, x_labels)
        '''
        # TESTING: THIS WORKS FOR CLASSIFICATION ONLY?

        # ?n x CS x CN
        p_QS_QN = p_Q_given_obs.unsqueeze(1) * self.readout_distr.unsqueeze(0)

        # Get best assignment of CN for each node and each CS
        best_CN_n = torch.argmax(p_QS_QN, dim=2, keepdim=True)
        p_QS_n = torch.gather(p_QS_QN, 2, best_CN_n).squeeze()  # ?n x CS
        p_QS_g = global_mean_pool(p_QS_n, batch)  # ?g x CS

        likely_graph_labels = self.emission.infer(p_QS_g, x_labels)
        return likely_graph_labels

    def complete_log_likelihood(self, esui, emission_target, batch):
        esi = global_add_pool(esui, batch)  # ?g x CS x CN
        # assert torch.allclose(esi.sum((1,2)), torch.tensor([1.]).to(esi.get_device())), esi.sum((1,2))
        es = esi.sum(2)  # ?g x CS
        # assert torch.allclose(es.sum(1), torch.tensor([1.]).to(es.get_device())), es.sum(1)

        complete_log_likelihood_1 = (es*(emission_target.log())).sum(1).sum()


        complete_log_likelihood_2 = (esi*(self.readout_distr.unsqueeze(0).log())).sum(dim=(1,2)).sum()

        return complete_log_likelihood_1 + complete_log_likelihood_2

    def _m_step(self, x_labels, y_labels, esui, batch):
        #self.readout_numerator += global_add_pool(esui, batch).sum(0)
        #assert torch.allclose(global_add_pool(esui, batch).sum(0), esui.sum(0))
        self.readout_numerator += esui.sum(0).detach()
        self.emission._m_step(x_labels, y_labels,
                              global_add_pool(esui, batch).sum(2))

    def m_step(self):
        self.emission.m_step()
        self.readout_distr.data = torch.div(self.readout_numerator,
                                            self.readout_numerator.sum(0))
        assert torch.allclose(self.readout_distr.sum(0), torch.tensor([1.]).to(self.readout_distr.get_device()))
        self.init_accumulators()


class IdentityGraphPredictor(GraphPredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return x, x


class LinearGraphPredictor(GraphPredictor):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.W = nn.Linear(dim_node_features, dim_target, bias=True)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = global_add_pool(x, batch)
        return self.W(x), x


class CGMMGraphPredictor(nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()

        original_node_features = dim_node_features[0]
        embeddings_node_features = dim_node_features[1]
        hidden_units = config['hidden_units']

        self.fc_global = torch.nn.Linear(embeddings_node_features, hidden_units)
        self.out = torch.nn.Linear(hidden_units, dim_target)

    def forward(self, data):
        extra = data[1]
        x = torch.reshape(extra.g_outs.float(), (extra.g_outs.shape[0], -1))
        out = self.out(F.relu(self.fc_global(x)))
        return out, x


class SimpleMLP(nn.Module):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__()

        hidden_units = config['hidden_units']
        self.fc_global = torch.nn.Linear(dim_node_features, hidden_units)
        self.out = torch.nn.Linear(hidden_units, dim_target)

    def forward(self, data):
        x = data.x.float()
        o = self.fc_global(x)
        out = self.out(F.relu(o))
        return out, x
