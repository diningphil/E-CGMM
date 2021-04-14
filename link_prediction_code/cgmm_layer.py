import torch

from emissions.categorical_emission import CategoricalEmission
from emissions.gaussian_emission import GaussianEmission

class CGMMLayer:
    def __init__(self, k, c, a, c2, l, is_first_layer, node_type='discrete', device='cpu'):
        """
        utils Layer
        :param k: dimension of output's alphabet, which goes from 0 to K-1 (when discrete)
        :param c: the number of hidden states
        :param c2: the number of states of the neighbours
        :param l: number of previous layers to consider. You must pass the appropriate number of statistics at training
        :param a: dimension of edges' alphabet, which goes from 0 to A-1
        """
        super().__init__()
        self.device = device
        # For comparison w.r.t Numpy implementation
        # np.random.seed(seed=10)
        self.node_type = node_type
        self.is_layer_0 = is_first_layer

        self.eps = 1e-8  # Laplace smoothing
        self.C = c
        self.K = k
        self.orig_A = a
        #self.A = a + 2  # may consider a special case of the recurrent arc and the special case of bottom state
        self.A = a

        if not self.is_layer_0:
            self.C2 = c2
            self.L = l

        # Initialisation of the model's parameters.
        # torch.manual_seed(0)

        if self.is_layer_0:
            # For debugging w.r.t Numpy version
            # pr = torch.from_numpy(np.random.uniform(size=self.C).astype(np.float32))
            if 'cuda' in device:
                pr = torch.nn.init.uniform_(torch.empty(self.C, dtype=torch.float64)).cuda()
            else:
                pr = torch.nn.init.uniform_(torch.empty(self.C, dtype=torch.float64))
            self.prior = pr / pr.sum()

            # print(self.prior)

        if self.node_type == 'discrete':
            self.emission = CategoricalEmission(self.K, self.C)#, device)
        elif self.node_type == 'continuous':
            self.emission = GaussianEmission(self.K, self.C)#, device)

        # print(self.emission)

        if not self.is_layer_0:
            # For debugging w.r.t Numpy version
            # self.layerS = torch.from_numpy(np.random.uniform(size=self.L).astype(np.float32))  #
            if 'cuda' in device:
                self.layerS = torch.nn.init.uniform_(torch.empty(self.L, dtype=torch.float64)).cuda()
                self.arcS = torch.zeros((self.L, self.A), dtype=torch.float64).cuda()
                self.transition = torch.empty([self.L, self.A, self.C, self.C2], dtype=torch.float64).cuda()
            else:
                self.layerS = torch.nn.init.uniform_(torch.empty(self.L, dtype=torch.float64))
                self.arcS = torch.zeros((self.L, self.A), dtype=torch.float64)
                self.transition = torch.empty([self.L, self.A, self.C, self.C2], dtype=torch.float64)

            self.layerS /= self.layerS.sum()

            for layer in range(0, self.L):
                # For debugging w.r.t Numpy version
                # elf.arcS[layer, :] = torch.from_numpy(np.random.uniform(size=self.A).astype(np.float32))

                self.arcS[layer, :] = torch.nn.init.uniform_(self.arcS[layer, :])
                self.arcS[layer, :] /= self.arcS[layer, :].sum()
                for arc in range(0, self.A):
                    for j in range(0, self.C2):
                        # For debugging w.r.t Numpy version
                        # tr = torch.from_numpy(np.random.uniform(size=self.C).astype(np.float32))

                        tr = torch.nn.init.uniform_(torch.empty(self.C))
                        self.transition[layer, arc, :, j] = tr / tr.sum()

            # print(self.arcS)
            # print(self.transition)

        self.init_accumulators()

    def init_accumulators(self):

        # These are variables where I accumulate intermediate minibatches' results
        # These are needed by the M-step update equations at the end of an epoch
        self.emission.init_accumulators()

        if self.is_layer_0:
            if 'cuda' in self.device:
                self.prior_numerator = torch.full([self.C], self.eps, dtype=torch.float64).cuda()
            else:
                self.prior_numerator = torch.full([self.C], self.eps, dtype=torch.float64)
            self.prior_denominator = self.eps * self.C

        else:
            if 'cuda' in self.device:

                self.layerS_numerator = torch.full([self.L], self.eps, dtype=torch.float64).cuda()
                self.arcS_numerator = torch.full([self.L, self.A], self.eps, dtype=torch.float64).cuda()
                self.transition_numerator = torch.full([self.L, self.A, self.C, self.C2], self.eps, dtype=torch.float64).cuda()
                self.arcS_denominator = torch.full([self.L, 1], self.eps * self.A, dtype=torch.float64).cuda()
                self.transition_denominator = torch.full([self.L, self.A, 1, self.C2], self.eps * self.C,
                                                     dtype=torch.float64).cuda()
            else:
                self.layerS_numerator = torch.full([self.L], self.eps, dtype=torch.float64)
                self.arcS_numerator = torch.full([self.L, self.A], self.eps, dtype=torch.float64)
                self.transition_numerator = torch.full([self.L, self.A, self.C, self.C2], self.eps, dtype=torch.float64)
                self.arcS_denominator = torch.full([self.L, 1], self.eps * self.A, dtype=torch.float64)
                self.transition_denominator = torch.full([self.L, self.A, 1, self.C2], self.eps * self.C,
                                                         dtype=torch.float64)

            self.layerS_denominator = self.eps * self.L

    def _compute_posterior_estimate(self, emission_for_labels, stats):

        # print(stats.shape)

        batch_size = emission_for_labels.size()[0]

        # Compute the neighbourhood dimension for each vertex
        neighbDim = torch.sum(stats[:, :, :, :], dim=3).float()  # --> ? x L x A

        # Replace zeros with ones to avoid divisions by zero
        # This does not alter learning: the numerator can still be zero

        neighbDim = torch.where(neighbDim == 0., torch.tensor([1.]).to(self.device), neighbDim)
        neighbDim[:, :, -1] = 1

        broadcastable_transition = torch.unsqueeze(self.transition, 0)  # --> 1 x L x A x C x C2
        broadcastable_stats = torch.unsqueeze(stats, 3).double()  # --> ? x L x A x 1 x C2
        
        tmp = torch.sum(torch.mul(broadcastable_transition, broadcastable_stats), dim=4)  # --> ? x L x A x C2

        broadcastable_layerS = torch.unsqueeze(self.layerS, 1)  # --> L x 1

        tmp2 = torch.reshape(torch.mul(broadcastable_layerS, self.arcS), [1, self.L, self.A, 1])  # --> 1 x L x A x 1

        div_neighb = torch.reshape(neighbDim, [batch_size, self.L, self.A, 1]).double()  # --> ? x L x A x 1

        tmp_unnorm_posterior_estimate = torch.div(torch.mul(tmp, tmp2), div_neighb)  # --> ? x L x A x C2

        tmp_emission = torch.reshape(emission_for_labels,
                                     [batch_size, 1, 1, self.C])  # --> ? x 1 x 1 x C2

        unnorm_posterior_estimate = torch.mul(tmp_unnorm_posterior_estimate, tmp_emission)  # --> ? x L x A x C2

        # Normalize
        norm_constant = torch.reshape(torch.sum(unnorm_posterior_estimate, dim=[1, 2, 3]), [batch_size, 1, 1, 1])
        norm_constant = torch.where(norm_constant == 0., torch.Tensor([1.]).double().to(self.device), norm_constant)

        posterior_estimate = torch.div(unnorm_posterior_estimate, norm_constant)  # --> ? x L x A x C2

        return posterior_estimate, broadcastable_stats, broadcastable_layerS, div_neighb

    def _E_step(self, labels, stats=None):
        batch_size = labels.size()[0]

        emission_of_labels = self.emission.get_distribution_of_labels(labels)

        if self.is_layer_0:
            # Broadcasting the prior
            numerator = torch.mul(emission_of_labels, torch.reshape(self.prior, shape=[1, self.C]))  # --> ?xC

            denominator = torch.sum(numerator, dim=1, keepdim=True)

            posterior_estimate = torch.div(numerator, denominator)  # --> ?xC

            # -------------------------------- Likelihood ------------------------------- #

            likelihood = torch.sum(torch.mul(posterior_estimate, torch.log(numerator)))

            return likelihood, posterior_estimate

        else:

            posterior_estimate, broadcastable_stats, broadcastable_layerS, div_neighb \
                = self._compute_posterior_estimate(emission_of_labels, stats)

            posterior_uli = torch.sum(posterior_estimate, dim=2)  # --> ? x L x C
            posterior_ui = torch.sum(posterior_uli, dim=1)  # --> ? x C

            # -------------------------------- Likelihood -------------------------------- #

            # NOTE: these terms can become expensive in terms of memory consumption, mini-batch computation is required.

            log_trans = torch.log(self.transition)

            num = torch.div(
                torch.mul(self.transition,
                          torch.mul(torch.reshape(self.layerS, [self.L, 1, 1, 1]),
                                    torch.reshape(self.arcS, [self.L, self.A, 1, 1]))),
                torch.unsqueeze(div_neighb, 4))

            num = torch.mul(num, torch.reshape(emission_of_labels, [batch_size, 1, 1, self.C, 1]))
            num = torch.mul(num, broadcastable_stats)

            den = torch.sum(num, dim=[1, 2, 3, 4], keepdim=True)  # --> ? x 1 x 1 x 1 x 1
            den = torch.where(torch.eq(den, 0.), torch.tensor([1.]).double().to(self.device), den)

            eulaij = torch.div(num, den)  # --> ? x L x A x C x C2

            # Compute the expected complete log likelihood
            likelihood1 = torch.sum(torch.mul(posterior_ui, torch.log(emission_of_labels)))
            likelihood2 = torch.sum(torch.mul(posterior_uli, torch.log(broadcastable_layerS)))
            likelihood3 = torch.sum(torch.mul(posterior_estimate,
                                              torch.reshape(torch.log(self.arcS), [1, self.L, self.A, 1])))

            likelihood4 = torch.sum(torch.mul(torch.mul(eulaij, broadcastable_stats), log_trans))

            likelihood = likelihood1 + likelihood2 + likelihood3 + likelihood4

            return likelihood, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats

    def E_step(self, labels, stats=None, training=False):

        with torch.no_grad():
            if self.is_layer_0:
                likelihood, posterior_ui = self._E_step(labels, stats)
                if training:
                    self._M_step(labels, posterior_ui, None, None, None, None)

                return likelihood, posterior_ui

            else:
                likelihood, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats \
                    = self._E_step(labels, stats)
                if training:
                    self._M_step(labels, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats)
                    return likelihood, eulaij
                else:
                    return likelihood, posterior_ui

    def _M_step(self, labels, posterior_estimate, posterior_uli, posterior_ui, eulaij, broadcastable_stats):

        if self.is_layer_0:

            tmp = torch.sum(posterior_estimate, dim=0)
            # These are used at each minibatch
            self.prior_numerator += tmp
            self.prior_denominator += torch.sum(tmp)

            self.emission.update_accumulators(posterior_estimate, labels)

        else:

            # These are equivalent to the categorical mixture model, it just changes how the posterior is computed
            self.emission.update_accumulators(posterior_ui, labels)

            tmp_arc_num = torch.sum(posterior_estimate, dim=[0, 3])  # --> L x A
            self.arcS_numerator += tmp_arc_num
            self.arcS_denominator += torch.unsqueeze(torch.sum(tmp_arc_num, dim=1), 1)  # --> L x 1

            new_layer_num = torch.sum(posterior_uli, dim=[0, 2])  # --> [L]
            self.layerS_numerator += new_layer_num
            self.layerS_denominator += torch.sum(new_layer_num)  # --> [1]

            new_trans_num = torch.sum(torch.mul(eulaij, broadcastable_stats), dim=0)
            self.transition_numerator += new_trans_num
            self.transition_denominator += torch.unsqueeze(torch.sum(new_trans_num, dim=2), 2)  # --> L x A x 1 x C2

    def M_step(self):

        self.emission.update_parameters()
        if self.is_layer_0:
            self.prior = self.prior_numerator / self.prior_denominator

        else:

            self.layerS = self.layerS_numerator / self.layerS_denominator
            self.arcS = self.arcS_numerator / self.arcS_denominator

            self.transition = self.transition_numerator / self.transition_denominator

        # I need to re-init accumulators, otherwise they will contain statistics of the previous epochs
        self.init_accumulators()