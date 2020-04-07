import numpy as np
import networkx as nx
import random as rand
from collections import defaultdict
import multiprocessing
from joblib import Parallel, delayed
import math
import itertools
import time
from sklearn.metrics import precision_recall_curve as pr_curve
def Initialise_graph(cascade_V_vec, cascade_t,n):
    G = nx.Graph() # initialise neihbourhood dict

    determinant_old_vec_l = []
    diag_vec_l = []
    I_c = []
    G.add_nodes_from(range(n))

    for ec, cascade_V in enumerate(cascade_V_vec):
        cascade_order = sorted(cascade_t[ec], key=cascade_t[ec].get)
        I_c.append(cascade_order[0])

        # add edges to starting graph G that can facilitate this cascade - a single line
        G.add_edges_from([(cascade_order[k - 1], cascade_order[k]) for k in range(1, len(cascade_order)) if cascade_t[ec][cascade_order[k]] < np.Inf ])

    for ec, cascade_V in enumerate(cascade_V_vec):
        # find prob of each link to add together for diagonals, note no negative as adding in the end (saves using abs)
        prob = {
        (i, j): np.exp(-(cascade_t[ec][j] - cascade_t[ec][i] if cascade_t[ec][j] > cascade_t[ec][i] else np.Inf))
        for (i, j) in nx.subgraph(G, cascade_V).edges()}
        prob.update(
            {(j, i): np.exp(-(cascade_t[ec][i] - cascade_t[ec][j] if cascade_t[ec][i] > cascade_t[ec][j] else np.Inf))
             for
             (i, j) in nx.subgraph(G, cascade_V).edges()})

        # find diag values for the nodes in the cascade.  Every node in the cascade should now have an entry
        diag = defaultdict(int)
        for (i, j) in prob.keys():
            diag[j] += prob[i, j]

        diag[I_c[ec]] = 1       # initial should be removed, so set to 1

        diag_l = {key: np.log(diag[key]) for key in diag.keys()}


        diag_vec_l.append(diag_l)


    G_adj =nx.to_dict_of_dicts(G)
    number_of_edges = len(G.edges())
    del G
    return G_adj,diag_vec_l, number_of_edges

def networkinferenceMCMC_TNT2(n,cascade_V_vec, cascade_t, p, beta, number_of_iterations, record_step=0, burn_in=0, seed=0):
    # alternative TNT method to reduce memory requirements
    if p == 1 or beta==1 or beta ==0:
        raise ValueError("p and beta must be (0,1)")


    beta_l = math.log(1-beta)
    graph_ratio_rem = math.log(1 - p) - math.log(p)
    graph_ratio_add = math.log(p) - math.log(1 - p)


    if seed !=0:
        rand.seed(seed)     # set seed

    if record_step:         # if recording
        save =[]


    tot_edges = int(n*(n-1)/2)

    # get initial graph and probabilties
    G_adj, diag_vec_l,number_of_edges = Initialise_graph(cascade_V_vec, cascade_t,n)

    # preallocate memory for edge list (good for sparse graphs, if dense just use adjacency)
    edge_list = np.zeros([number_of_edges+20*n,2],dtype=int)

    # fill edge list
    k=0
    for pair in iter((n,nbr) for n, nbrs in G_adj.items() for nbr, ddict in nbrs.items()):
        if pair[0]<pair[1]:
            edge_list[k] = pair
            k+=1
    t=time.time()
    for it in range(number_of_iterations):
        # propose a move step to get G'

        # for TNT sampler choose links or not
        if rand.random()<0.5:

            went_through_no_tie=0
            rand_ind = int(math.floor(rand.random() * number_of_edges))
            [i,j] = edge_list[rand_ind]
            had_edge = 1
            graph_ratio_l = graph_ratio_rem
            Q_l = math.log(1/tot_edges) + math.log(1/(1/number_of_edges + 1/tot_edges))


        else:
            went_through_no_tie=1
            # pick random numbers (nodes) and make sure they are different
            i = int(math.floor(rand.random() * n))
            j = int(math.floor(rand.random() * n))
            while i == j:
                 j = int(math.floor(rand.random() * n))


            if (i in G_adj) and (j in G_adj[i]):        # if already an edge
                had_edge = 1
                graph_ratio_l = graph_ratio_rem
                Q_l = math.log(1/tot_edges) +  math.log(1/(1/number_of_edges + 1/tot_edges))

            else:
                had_edge = 0

                graph_ratio_l = graph_ratio_add
                Q_l = math.log(1+ tot_edges/(number_of_edges+1))


        new_j_l = np.zeros([len(cascade_V_vec), 3])
        diag_l_tmp = 0
        beta_exponent = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            for ec, cascade_V in enumerate(cascade_V_vec):
                timei = cascade_t[ec][i]
                timej = cascade_t[ec][j]

                if timei > timej:                # make i  always infect j
                    i, j = j, i
                    timei, timej = timej, timei
                if timej < np.Inf:              #j in cascade_V:
                    new_j_l[ec,0] = ec
                    new_j_l[ec,1] = j
                    detadding = diag_vec_l[ec][j]
                    diag_l_tmp += detadding

                    new_j_l[ec,2] = detadding + np.log(1 + ((-1) ** had_edge) * math.exp(-(timej - timei)-detadding))   # logaddexp

                    if not new_j_l[ec,2] < np.Inf: #INFINITY if NAN then this will be False
                        new_j_l[ec,2] = -np.Inf #INFINITY

                if timei<np.Inf:#i in cascade_V:
                    beta_exponent += ((-1) ** had_edge)  # r changes if i or j in cascade


        alpha_l = sum(new_j_l[:,2]) - diag_l_tmp + graph_ratio_l + beta_exponent * beta_l + Q_l


        if math.log(rand.random()) < alpha_l:  # accept with prob alpha and go to the new graph
            for each in new_j_l:
                if not (each[2] ==0 and each[1]==0 and each[0]==0):
                  diag_vec_l[int(each[0])][int(each[1])] = each[2]

            if had_edge == 0:
                edge_list[number_of_edges]  = [i,j]         #add this edge to the edge_list
                number_of_edges+=1          # add one to the number of edges
                try:
                    G_adj[i][j] = {}
                except KeyError:
                    G_adj[i] = {}
                    G_adj[i][j] = {}
                try:
                    G_adj[j][i] = {}
                except KeyError:
                    G_adj[j] = {}
                    G_adj[j][i] = {}
            else:
                del G_adj[i][j]
                del G_adj[j][i]
                if went_through_no_tie==0:
                    edge_list[rand_ind] = edge_list[number_of_edges-1]        # put the last value in to the 'empty spot'
                    edge_list[number_of_edges-1] = [0,0]                      # make the last one empty
                if went_through_no_tie ==1:
                    try:
                        ind = int(np.where((edge_list[:,0] == i) & (edge_list[:,1]==j))[0])
                    except TypeError:
                        ind = int(np.where((edge_list[:,0] == j) & (edge_list[:,1]==i))[0])

                    edge_list[ind] = edge_list[number_of_edges-1]        # put the last value in to the 'empty spot'
                    edge_list[number_of_edges-1] = [0,0]                      # make the last one empty
                number_of_edges-=1                                  # reduce number of edges in graph by one

        if record_step:
            if it%record_step==0:
                if burn_in and (it>n**2):
                    save.append([(i,j) for i in G_adj.keys() for j in G_adj[i]])
                else:
                    save.append([(i,j) for i in G_adj.keys() for j in G_adj[i]])

    print(time.time()-t)
    if record_step:
        return save
    else:
        return G_adj
def networkinferenceMCMC_TNT(n,cascade_V_vec, cascade_t, p, beta, number_of_iterations, record_step=0, burn_in=0, seed=0):
    # higher memory original tnt
    if p == 1 or beta==1 or beta ==0:
        raise ValueError("p and beta must be (0,1)")


    beta_l = math.log(1-beta)
    graph_ratio_rem = math.log(1 - p) - math.log(p)
    graph_ratio_add = math.log(p) - math.log(1 - p)


    if seed !=0:
        rand.seed(seed)     # set seed

    if record_step:         # if recording
        save =[]


    tot_edges = int(n*(n-1)/2)

    # get initial graph and probabilties
    G_adj, diag_vec_l,number_of_edges = Initialise_graph(cascade_V_vec, cascade_t,n)

    # preallocate memory for edge list (good for sparse graphs, if dense just use adjacency)
    edge_list = np.zeros([tot_edges,2],dtype=int)

    # fill edge list
    k=0
    for pair in iter((n,nbr) for n, nbrs in G_adj.items() for nbr, ddict in nbrs.items()):
        if pair[0]<pair[1]:
            edge_list[k] = pair
            k+=1

    for i in range(n):
        for j in range(i+1, n):
            if not (i in G_adj.keys() and j in G_adj[i].keys() ):
                edge_list[k] = [i,j]
                k+=1

    t=time.time()
    for it in range(number_of_iterations):
        # propose a move step to get G'

        # for TNT sampler choose links or not
        if rand.random()<0.5:
            rand_ind = int(math.floor(rand.random() * number_of_edges))
            [i,j] = edge_list[rand_ind]
            had_edge = 1
            Q_l = math.log(tot_edges-number_of_edges + 1) -math.log(number_of_edges)
            graph_ratio_l = graph_ratio_rem
        else:
            rand_ind = number_of_edges + int(math.floor(rand.random() * (tot_edges-number_of_edges)))
            [i,j] = edge_list[rand_ind]
            had_edge = 0
            graph_ratio_l = graph_ratio_add
            Q_l = math.log(number_of_edges+1) - math.log(tot_edges-number_of_edges)

        new_j_l = np.zeros([len(cascade_V_vec), 3])
        diag_l_tmp = 0
        beta_exponent = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            for ec, cascade_V in enumerate(cascade_V_vec):
                timei = cascade_t[ec][i]
                timej = cascade_t[ec][j]

                if timei > timej:                # make i  always infect j
                    i, j = j, i
                    timei, timej = timej, timei
                if timej < np.Inf:              #j in cascade_V:
                    new_j_l[ec,0] = ec
                    new_j_l[ec,1] = j
                    detadding = diag_vec_l[ec][j]
                    diag_l_tmp += detadding

                    new_j_l[ec,2] = detadding + np.log(1 + ((-1) ** had_edge) * math.exp(-(timej - timei)-detadding))   # logaddexp

                    if not new_j_l[ec,2] < np.Inf: #INFINITY if NAN then this will be False
                        new_j_l[ec,2] = -np.Inf #INFINITY

                if timei<np.Inf:#i in cascade_V:
                    beta_exponent += ((-1) ** had_edge)  # r changes if i or j in cascade


        alpha_l = sum(new_j_l[:,2]) - diag_l_tmp + graph_ratio_l + beta_exponent * beta_l + Q_l


        if math.log(rand.random()) < alpha_l:  # accept with prob alpha and go to the new graph
            for each in new_j_l:
                if not (each[2] ==0 and each[1]==0 and each[0]==0):
                  diag_vec_l[int(each[0])][int(each[1])] = each[2]

            if had_edge == 0:
                edge_list[rand_ind] = edge_list[number_of_edges]
                edge_list[number_of_edges]  = [i,j]         #add this edge to the edge_list
                number_of_edges+=1          # add one to the number of edges
                try:
                    G_adj[i][j] = {}
                except KeyError:
                    G_adj[i] = {}
                    G_adj[i][j] = {}
                try:
                    G_adj[j][i] = {}
                except KeyError:
                    G_adj[j] = {}
                    G_adj[j][i] = {}
            else:
                del G_adj[i][j]
                del G_adj[j][i]
                edge_list[rand_ind] = edge_list[number_of_edges-1]        # put the last value in to the 'empty spot'
                edge_list[number_of_edges-1] = [i,j]                      # make the last one empty
                number_of_edges-=1

        if record_step:
            if it%record_step==0:
                if burn_in and (it>n**2):
                    save.append([(i,j) for i in G_adj.keys() for j in G_adj[i]])
                else:
                    save.append([(i,j) for i in G_adj.keys() for j in G_adj[i]])

    print(time.time()-t)
    if record_step:
        return save
    else:
        return G_adj


def IndependentCascade_edges(G, I, beta,edges={}):
    W = I
    if edges =={}:
        edges = defaultdict(lambda: 0)
    status = defaultdict(lambda: 0)
    time = defaultdict(lambda: float('inf'))  # [np.Inf] * len(G.nodes())
    for i in I:
        status[i] = 1
        time[i] = 0
    nodes = [I]
    node = []
    K = len(I)
    while len(W) > 0:
        W_tmp = []
        for v in W:
            for u in G.neighbors(v):
                if status[u] != 1:
                    if rand.random() < beta:
                        edges[v, u] +=1
                        edges[u, v]+=1
                        W_tmp.append(u)
                        status[u] = 1
                        time[u] = time[v] + np.random.exponential()
                        node.append(u)
                        K = K + 1
        W = W_tmp

        if len(W) > 0:
            nodes.append(node)
            node = []

    return nodes, time, edges

def SimulateCascades(RG, beta, C=None,fraction=None, filename=None):
    # simulate C cascades on this network
    # this code assumes GNP network
    number_of_edges = len(RG.edges())
    activated_nodes = []
    cascade_V_vec = []
    cascade_t = []
    n = len(RG)
    # print np.mean(RG.degree().values())
    if C and fraction:
        raise ValueError("You must choose either number of cascades or fraction of edges activated")
    if C:
        edges={}
        for I in range(C):
            [nodes, times,edges] = IndependentCascade_edges(RG, [rand.choice(list(RG.nodes()))], beta,edges)
            activated_nodes = [item for sublist in nodes for item in sublist]
            cascade_V_vec.append(activated_nodes)
            cascade_t.append(times)
    elif fraction:
        edge_frac = 0
        edges={}
        if nx.is_directed(RG):
            while len(edges.keys())/(2*number_of_edges) < fraction:
                [nodes, times,edges] = IndependentCascade_edges(RG, [rand.choice(list(RG.nodes()))], beta, edges)
                activated_nodes = [item for sublist in nodes for item in sublist]
                cascade_V_vec.append(activated_nodes)
                cascade_t.append(times)
                GG = nx.DiGraph()
                GG.add_edges_from(edges.keys())
                if len(GG.edges())/len(RG.edges()) >fraction:
                    break
        else:
                while len(edges.keys())/(2*number_of_edges) < fraction:
                    [nodes, times,edges] = IndependentCascade_edges(RG, [rand.choice(list(RG.nodes()))], beta, edges)
                    activated_nodes = [item for sublist in nodes for item in sublist]
                    cascade_V_vec.append(activated_nodes)
                    cascade_t.append(times)

    else:
        raise ValueError('You must choose one of fraction or cascades')

    if filename:
        # create a cascade file that lists the node names and their indices
        dict={}

        with open(filename, "w") as cascade_text:
            # first write the nodes
            for k,node in enumerate(sorted(list(RG.nodes()))):
                cascade_text.write(str(k) + "," + str(node) + '\n')
                dict[node] = k

            # then write blank line
            cascade_text.write('\n')
            # then write the cascades

            for each_cascade in cascade_t:
                required_format_list = []
                for key, value in each_cascade.items():
                    if value < np.inf:
                        required_format_list.append(str(dict[key]))
                        required_format_list.append(str(value))
                cascade_text.write(','.join(required_format_list) + str('\n'))

    return cascade_V_vec, cascade_t


n=20
beta = 0.4
z = 4
p = z / (n - 1)
RG = nx.fast_gnp_random_graph(n,p,seed=1)
cascade_V_vec,cascade_t = SimulateCascades(RG, beta, fraction=0.99)
number_of_iterations = n**2 * 100
num_cores = multiprocessing.cpu_count()-1
sample_size=100
samples = Parallel(n_jobs=num_cores)(delayed(networkinferenceMCMC_TNT)(n,cascade_V_vec, cascade_t, p, beta, number_of_iterations, seed=rep) for rep in range(sample_size))

save = [[(i,j) for i in G_adj.keys() for j in G_adj[i]] for G_adj in samples]

def ROC(score, y,number_of_points=100):
    # score is a vector with the ith element the probability of i existing
    # y is binary vector with 1 if the ith element exists in the true data

    roc_x = []
    roc_y = []
    # min_score = min(score)
    # max_score = max(score)
    # thr = np.linspace(min_score, max_score, 30)
    thr = np.linspace(1+1/number_of_points 0, number_of_points) 
    FP = 0
    TP = 0
    P = sum(y)
    N = len(y) - P

    if N == 0:
        roc_x = [0] * (len(thr) - 1) + [1]

    for T in thr:
        for i in range(0, len(score)):
            if (score[i] >= T):
                if (y[i] == 1):
                    TP = TP + 1
                if (y[i] == 0):
                    FP = FP + 1

        # print TP, FP
        if N != 0:
            roc_x.append(FP / float(N))

        roc_y.append(TP / float(P))
        FP = 0
        TP = 0
    return roc_x, roc_y
def edge_frequencies(save,n=None):
    # find the frequency of edges in the saved graphs
    # save is a list of [G_i.edges(),G_i.nodes()] OR [G_i.edges()]
    # edges = {(i,j):1 for i in range(n) for j in range(n) if i<j}
    if n:
        edge_freq= {(i,j):0 for (i,j) in itertools.permutations(range(n),2)}
    else:
        edge_freq = {}

    for gr in save:
        for ed in gr:
      #      if ed[0]>ed[1]:
      #          ed = (ed[1],ed[0])
            try:
                edge_freq[ed] = edge_freq[ed] + 1 / len(save)
            except KeyError:
              #  try:
              #      edge_freq[(ed[1],ed[0])] = edge_freq[(ed[1],ed[0])]+ 1 / len(save)
              #  except KeyError:
                edge_freq[ed] = 1 / len(save)

    return edge_freq
def ROC_output(edge_freq,RG):
    real_edges = {(key): 1 if key in RG.edges()  else 0 for key in edge_freq.keys()}
    score = []
    truth = []
    for key, value in edge_freq.items():
        score.append(np.round(value, 3))
        truth.append(np.round(real_edges[key], 3))

    return score,truth
score, truth = ROC_output(edge_frequencies(save,n),RG)
roc_x,roc_y = ROC(score, truth)
lr_precision, lr_recall, _ = pr_curve(truth, score)
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
