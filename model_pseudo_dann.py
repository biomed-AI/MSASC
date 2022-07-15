import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import L2ProjFunction, GradientReversalLayer,VATLoss
import utils

import loss_utility

def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    '''
    change gradually from A to B, according to the formula (from <Importance Weighted Adversarial Nets for Partial Domain Adaptation>)
    A + (2.0 / (1 + exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)

    =code to see how it changes(almost reaches B at %40 * max_iter under default arg)::

        from matplotlib import pyplot as plt

        ys = [aToBSheduler(x, 1, 3) for x in range(10000)]
        xs = [x for x in range(10000)]

        plt.plot(xs, ys)
        plt.show()

    '''
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)


class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs

class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply
    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class scAdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(scAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=self.max_iter))

    def forward(self, x, reverse = True):
        if reverse:
            x = self.grl(x)
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]



########## Some components ##########
class MLPNet(nn.Module):

    def __init__(self, configs):
        """
        MLP network with ReLU
        """

        super().__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i+1])
                                      for i in range(self.num_hidden_layers)])
        self.final = nn.Linear(self.num_neurons[-1], configs["output_dim"])
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability
        self.process_final = configs["process_final"]

    def forward(self, x):

        for hidden in self.hiddens:
            x = F.relu(hidden(self.dropout(x)))
        if self.process_final:
            return F.relu(self.final(self.dropout(x)))
        else:
            # no dropout or transform
            return self.final(x)


class DarnMLP(nn.Module):

    def __init__(self, configs):
        """
        DARN with MLP
        """
        super(DarnMLP, self).__init__()

        self.loss_function=nn.CrossEntropyLoss()
        self.num_src_domains = configs["num_src_domains"]
        # Gradient reversal layer.
        self.grl = GradientReversalLayer.apply
        self.mode = mode = configs["mode"]
        self.mu = configs["mu"]
        self.gamma = configs["gamma"]

        if mode == "L2":
            self.proj = L2ProjFunction.apply
        else:
            self.proj = None

        fea_configs = {"input_dim": configs["input_dim"],
                       "hidden_layers": configs["hidden_layers"][:-1],
                       "output_dim": configs["hidden_layers"][-1],
                       "drop_rate": configs["drop_rate"],
                       "process_final": False}
        self.feature_net = MLPNet(fea_configs)

        self.class_net = nn.Linear(configs["hidden_layers"][-1],
                                   configs["num_classes"])

        self.ad_nets = nn.ModuleList([scAdversarialNetwork(configs["hidden_layers"][-1], 1024)
                                          for _ in range(self.num_src_domains)])



    def forward(self, sinputs, soutputs, tinputs, tipout=None, mode = 1):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:            tuple(aggregated loss, domain weights)
        """

        if mode == 1:

            # Compute features
            s_features = []
            for i in range(self.num_src_domains):
                s_features.append(self.feature_net(sinputs[i]))
            t_features = self.feature_net(tinputs)

            train_losses = torch.stack([self.loss_function(self.class_net(s_features[i]), soutputs[i])
                                    for i in range(self.num_src_domains)])
            vat=VATLoss()
            vat_losses = torch.stack([vat(self,sinputs[i])
                                    for i in range(self.num_src_domains)])

            domain_prob_discriminator_source = []
            domain_prob_discriminator_target = []
            for i in range(self.num_src_domains):
                domain_prob_discriminator_source.append(self.ad_nets[i](s_features[i]))
                domain_prob_discriminator_target.append(self.ad_nets[i](t_features))

            adv_loss = []
            for i in range(self.num_src_domains):
                adv_loss_iter = loss_utility.BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_source[i]), \
                                                     predict_prob=domain_prob_discriminator_source[i])
                adv_loss_iter += loss_utility.BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_target[i]), \
                                                      predict_prob=1 - domain_prob_discriminator_target[i])
                adv_loss.append(adv_loss_iter)

            domain_losses = torch.stack([adv_loss[i] for i in range(self.num_src_domains)])


            return self._aggregation(train_losses+0.1*vat_losses, domain_losses)
        
        if mode == 2:
            t_features = self.feature_net(tinputs)
            train_losses = self.loss_function(self.class_net(t_features), tipout)
            return train_losses

    def _aggregation(self, train_losses, domain_losses):
        """
        Aggregate the losses into a scalar
        """

        mu, alpha = self.mu, None
        if self.num_src_domains == 1:  # dann
            loss = train_losses + mu * domain_losses
        else:
            mode, gamma = self.mode, self.gamma
            if mode == "dynamic":  # mdan
                g = (train_losses + mu * domain_losses) * gamma
                loss = torch.logsumexp(g, dim=0) / gamma
            elif mode == "L2":  # darn
                g = gamma * (train_losses + mu * domain_losses)
                alpha = self.proj(g)
                loss = torch.dot(g, alpha) + torch.norm(alpha)
                alpha = alpha.cpu().detach().numpy()
            else:
                raise NotImplementedError("Unknown aggregation mode %s" % mode)

        return loss, alpha

    def inference(self, x):

        x = self.feature_net(x)
        x = self.class_net(x)
        return F.log_softmax(x, dim=1)
    def out(self, x):
        x = self.feature_net(x)
        x = self.class_net(x)
        return F.softmax(x, dim=1)
    def get_embedding(self, x):
        x = self.feature_net(x)
        # x = self.class_net(x)
        return x