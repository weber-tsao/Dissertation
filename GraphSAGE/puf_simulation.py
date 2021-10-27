import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

import pypuf.simulation
import pypuf.io

# from graphsage.encoders import Encoder
# from graphsage.aggregators import MeanAggregator

"""our Python environment or installation doesn't have the spyder‑kernels module or the right version of it installed (>= 1.9.4 and < 1.10.0). Without this module is not possible for Spyder to create a console for you.
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """

    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        #print(self.features)
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        #print(nodes) #500, 1206
        #print(to_neighs)
        a =[]
        walk = []
        v = [0, 1, 2, 3, 4, 5]
        if torch.is_tensor(nodes):
            a= nodes.tolist()
            walk = cal_PUF_path(v, a[1])
            # print(walk)
        else:
            for node in nodes:
                walk.append(cal_PUF_path(v, node))
        print(walk)
        #print(type(nodes))

        #_sample = random.sample
        samp_neighs = []

        for i in range(len(walk)):
            _set = set()
            current_node = walk[i]
            current_adj_set = to_neighs[current_node]
            iterator = iter(current_adj_set)
            a = next(iterator, None)
            b = next(iterator, None)
            for x in range(len(walk)):
                if walk[x] == a:
                    _set.update({a})
                elif walk[x] == b:
                    _set.update({b})
                else: pass
            samp_neighs.append(_set)

        #print(_set)
        #samp_neighs = to_neighs
        #print(len(samp_neighs)) #500, 1206
        samp_neighs.remove(set())
        #print(samp_neighs)
        '''if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]'''
        # make it unique
        unique_nodes_list = list(set.union(*samp_neighs))
        # print(len(unique_nodes_list)) #1206, 2097
        #print(unique_nodes_list)
        #  print ("\n unl's size=",len(unique_nodes_list))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}  # make it the format(node index: count)
        #print(len(unique_nodes)) #1206, 2097
        #print(unique_nodes)
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        #print(mask.size()) # [500, 1206] [1206, 2097]
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        #print(column_indices)
        # print(len(column_indices)) # 1811, 5431
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        #print(row_indices)
        # print(len(row_indices)) # 1811, 5431

        mask[row_indices, column_indices] = 1
        #print(mask)
        #print(mask.size()) #[500, 1206] [1206, 2097]
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        #print(num_neigh)
        # print(num_neigh.size()) # [500, 1] [1206, 1]
        #print("Previous:{}".format (mask))
        mask = mask.div(num_neigh)
        # print(mask.size()) # [500, 1206] [1206, 2097]
        #print("After:{}".format (mask))
        #print(self.features(torch.LongTensor(unique_nodes_list)))
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
            #print(self.features(torch.LongTensor(unique_nodes_list).cuda()))
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            #print(self.features(torch.LongTensor(unique_nodes_list)))
        #print("embed_matrix:{}".format(embed_matrix))
        #print(self.features(torch.LongTensor([1])))
        #print(embed_matrix.size()) # [2097, 1433] [1206, 128]
        to_feats = mask.mm(embed_matrix)
        #print("to_feat:{}".format  (to_feats))
        # print(to_feats.size()) # [1206, 1433] [500, 128]

        return to_feats


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, aggregator,
                 num_sample=10,
                 base_model=None, gcn=False, cuda=False,
                 feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        #print(nodes)
        # print(len(nodes)) #500, 1206
        v = [0, 1, 2, 3, 4, 5]
        data, crp_response = cal_puf_data()

        #walk = cal_PUF_path(v, nodes)
        #walk =  [5,3,1]
        #print(walk)
        crp_c = data
        neigh_feats = self.aggregator.forward(nodes, self.adj_lists, self.num_sample)
        #print(neigh_feats)
        #print(neigh_feats.size()) # [1206, 1433] [500, 128]

        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats

        # print(self.weight.size()) # [128, 1433] [128, 128]

        #print(combined.size()) # [1206, 1433] [500, 128]
        #print(self.weight)
        combined = F.relu(self.weight.mm(combined.t()))
        #print(combined)
        #print(combined.size()) # [128, 1206] [128, 500]
        #print(type(combined))
        return combined


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        #print(nodes)
        embeds = self.enc(nodes)
        #print(embeds)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        #print(nodes)
        scores = self.forward(nodes)
        #print(labels.squeeze())
        return self.xent(scores, labels.squeeze())

def sort_PUF_data():
    crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
    crp_to_two = []
    for i in range(500):
        c_array = crp_loaded[i][0]
        c = []
        for y in range(len(c_array)):
            c.append(c_array[y])
        r = crp_loaded[i][1]
        crp_to_two.append((c, int(r[0][0])))

        crp_r = -1
        crp_c = []
        current_challenges = crp_loaded[i][0]

        if crp_loaded[i][1] == -1:
            crp_r = 1

        for x in range(len(crp_loaded[0][0])):
            if current_challenges[x] == -1:
                crp_c.append(1)
            else:
                crp_c.append(-1)
        crp_to_two.append((crp_c, crp_r))
    return crp_to_two

def cal_PUF_path(v, loop):
    crp_to_two = sort_PUF_data()
    #print(loop)
    walk = []  # Walking list
    crp_c_list = crp_to_two[loop][0]
    crp_r_list = crp_to_two[loop][1]
    #print(crp_c_list)
    v.sort()

    if crp_r_list == 1:
        walk.append(v[-2])
    else:
        walk.append(v[-1])

    for i in range((len(crp_c_list)-1), 0, -1):
        x = walk[-1]
        if x%2 == 0:
            if crp_c_list[i] == 1:
                walk.append(v[x-1])
            else:
                walk.append(v[x-2])
        else:
            if crp_c_list[i] == -1:
                walk.append(v[x-2])
            else:
                walk.append(v[x-3])
    #print(walk)
    return walk

def cal_puf_data():
    #puf = pypuf.simulation.ArbiterPUF(n=3, seed=1)
    #crp = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=500, seed=2)
    #crp.save('crps.npz')
    crp_loaded = pypuf.io.ChallengeResponseSet.load('crps.npz')
    crp_response = []
    crp_challenge = []
    for i in range(500):
        c_array = crp_loaded[i][0]
        for y in range(len(c_array)):
            crp_challenge.append(c_array[y])
        r = crp_loaded[i][1]
        crp_response.append(int(r[0][0]))

        crp_r = -1
        crp_c = []
        current_challenges = crp_loaded[i][0]

        if crp_loaded[i][1] == -1:
            crp_r = 1

        for x in range(len(crp_loaded[0][0])):
            if current_challenges[x] == -1:
                crp_c.append(1)
            else:
                crp_c.append(-1)
        crp_response.append(crp_r)
        crp_challenge.append(crp_c)
    #print(len(crp_response))
    return crp_challenge, crp_response

def load_cora():
    num_nodes = 6
    num_feats = 6 # one hot array
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((1000, 1), dtype=np.int64)
    node_map = {}
    info = []

    crp_challenge, crp_response = cal_puf_data() # crp_response == labels for the walk

    for z in range(1000):
        if crp_response[z] == -1:
            labels[z] = 0
        else: labels[z] = crp_response[z]

    for x in range(num_nodes):
        node_feat = [0]*num_nodes
        node_feat[x] = 1
        info.append(node_feat)
    #print(info)

    for i in range(num_nodes):
        feat_data[i, :] = [x for x in info[i]]
        node_map[i] = info[i]

    #print(info[-1])
    #print(info)
    #print(feat_data)
    #print(node_map)

    adj_lists = defaultdict(set)
    for y in range(0, num_nodes, 2):
        if y == 0:
            adj_lists[y].add(None)
            adj_lists[y + 1].add(None)
        else:
            node1 = y-1
            #print(node_map[0])
            node2 = y-2
            adj_lists[y].add(y-1)
            adj_lists[y+1].add(y-1)
            adj_lists[y].add(y-2)
            adj_lists[y+1].add(y-2)
    #print(adj_lists)
    #print(len(crp_response))
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 6
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(6, 6)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    #print(features)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 6, 6, adj_lists, agg1, gcn=True, cuda=False)
    #x = lambda nodes: enc1(nodes).t()
    #print(x(4))
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    #print(enc1.embed_dim)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 6, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(2, enc2)
    #    graphsage.cuda()
    rand_indices = np.random.permutation(1000)
    #print(rand_indices)
    test = rand_indices[:200]
    val = rand_indices[200:400]
    train = list(rand_indices[400:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(1000):
        batch_nodes = train[:2]
        #print(batch_nodes)
        batch_labels = np.empty((2, 1), dtype=np.int64)
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        #print(batch_labels)
        batch_labels[batch] = labels[batch]
        loss = graphsage.loss(batch_nodes,
                Variable(torch.LongTensor(batch_labels)))

        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))


if __name__ == "__main__":
    run_cora()
