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

#from graphsage.encoders import Encoder
#from graphsage.aggregators import MeanAggregator

"""
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
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        #print(len(nodes)) #500, 1206
        #print(nodes)
        #print(type(nodes))
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        #print(len(samp_neighs)) #500, 1206
        #print(samp_neighs)
        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        #print(samp_neighs)
        # make it unique
        unique_nodes_list = list(set.union(*samp_neighs))
        #print(len(unique_nodes_list)) #1206, 2097
        #print(unique_nodes_list)
      #  print ("\n unl's size=",len(unique_nodes_list))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)} # make it the format(node index: count)
        #print(len(unique_nodes)) #1206, 2097
        #print(unique_nodes)
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        #print(mask.size()) # [500, 1206] [1206, 2097]
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        #print(column_indices)
        #print(len(column_indices)) # 1811, 5431
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # print(row_indices)
        #print(len(row_indices)) # 1811, 5431

        mask[row_indices, column_indices] = 1
        #print(mask[55])
        #print(mask.size()) #[500, 1206] [1206, 2097]
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        #print(num_neigh)
        #print(num_neigh.size()) # [500, 1] [1206, 1]

        mask = mask.div(num_neigh)
        #print(mask.size()) # [500, 1206] [1206, 2097]

        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())

        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        #print(embed_matrix)
        #print(embed_matrix.size())
        #print(self.features(torch.LongTensor([0])))
        #print(embed_matrix.size()) # [2097, 1433] [1206, 128]
        to_feats = mask.mm(embed_matrix)
        #print(to_feats)
        #print(to_feats.size()) # [1206, 1433] [500, 128]

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
        print(nodes)
        #print(len(nodes)) #500, 1206
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], 
                self.num_sample)
        #print(neigh_feats)
        #print(neigh_feats.size()) # [1206, 1433] [500, 128]

        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
                #print(self.features(torch.LongTensor(nodes).cuda()))
            else:
                self_feats = self.features(torch.LongTensor(nodes))
                #print(self.features(torch.LongTensor(nodes)))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
            print("ddd")
        else:
            combined = neigh_feats
        #print(self.features(torch.LongTensor(nodes)).size())
        #print(self.weight.size()) # [128, 1433] [128, 128]
        #print(combined.size()) # [1206, 1433] [500, 128]
        combined = F.relu(self.weight.mm(combined.t()))
        print(combined)
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
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        #print(self.xent(scores, labels.squeeze()))
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("C:/Users/weber/OneDrive/Desktop/Dissertation/GraphSAGE/cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = [float(x) for x in info[1:-1]]
            #print(feat_data)
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
            #print(info[-1])
    #print(len(labels)) # 2708
    #print(labels)

    adj_lists = defaultdict(set)
    with open("C:/Users/weber/OneDrive/Desktop/Dissertation/GraphSAGE/cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            #print(info[0])
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    #print(len(feat_data))
    #print(node_map)
    #print(adj_lists)
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    #print(features)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    #print(enc1.embed_dim)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    graphsage = SupervisedGraphSage(7, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])
    #print(train)

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        print(batch_nodes)
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        #print(Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        #print(loss)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print (batch, loss.item())

    val_output = graphsage.forward(val) 
    print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print ("Average batch time:", np.mean(times))


if __name__ == "__main__":
    run_cora()
