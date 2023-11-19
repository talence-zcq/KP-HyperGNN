from collections import Counter
import numpy as np
from itertools import combinations
import torch
import torch_sparse
from scipy.sparse import coo_matrix,csr_matrix
from torch import scatter_add
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data
from copy import deepcopy as c
import networkx as nx


# will delete later
from torch_geometric.utils import to_scipy_sparse_matrix


def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
#     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges
    if not ((data.n_x+data.num_hyperedges-1) == data.edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[
        0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data

def Add_Self_Loops(data):
    # update so we dont jump on some indices
    # Assume edge_index = [V;E]. If not, use ExtractV2E()
    edge_index = data.edge_index
    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges

    if not ((data.n_x + data.num_hyperedges - 1) == data.edge_index[1].max().item()):
        print('num_hyperedges does not match! 2')
        return

    hyperedge_appear_fre = Counter(edge_index[1].numpy())
    # store the nodes that already have self-loops
    skip_node_lst = []
    for edge in hyperedge_appear_fre:
        if hyperedge_appear_fre[edge] == 1:
            skip_node = edge_index[0][torch.where(
                edge_index[1] == edge)[0].item()]
            skip_node_lst.append(skip_node.item())

    new_edge_idx = edge_index[1].max() + 1
    new_edges = torch.zeros(
        (2, num_nodes - len(skip_node_lst)), dtype=edge_index.dtype)
    tmp_count = 0
    for i in range(num_nodes):
        if i not in skip_node_lst:
            new_edges[0][tmp_count] = i
            new_edges[1][tmp_count] = new_edge_idx
            new_edge_idx += 1
            tmp_count += 1

    data.totedges = num_hyperedges + num_nodes - len(skip_node_lst)
    edge_index = torch.cat((edge_index, new_edges), dim=1)
    # Sort along w.r.t. nodes
    _, sorted_idx = torch.sort(edge_index[0])
    data.edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)
    return data

def expand_edge_index(data, edge_th=0):
    '''
    args:
        num_nodes: regular nodes. i.e. x.shape[0]
        num_edges: number of hyperedges. not the star expansion edges.

    this function will expand each n2he relations, [[n_1, n_2, n_3],
                                                    [e_7, e_7, e_7]]
    to :
        [[n_1,   n_1,   n_2,   n_2,   n_3,   n_3],
         [e_7_2, e_7_3, e_7_1, e_7_3, e_7_1, e_7_2]]

    and each he2n relations:   [[e_7, e_7, e_7],
                                [n_1, n_2, n_3]]
    to :
        [[e_7_1, e_7_2, e_7_3],
         [n_1,   n_2,   n_3]]

    and repeated for every hyperedge.
    '''
    edge_index = data.edge_index
    num_nodes = data.n_x[0].item()
    if hasattr(data, 'totedges'):
        num_edges = data.totedges
    else:
        num_edges = data.num_hyperedges[0]

    expanded_n2he_index = []
#     n2he_with_same_heid = []

#     expanded_he2n_index = []
#     he2n_with_same_heid = []

    # start edge_id from the largest node_id + 1.
    cur_he_id = num_nodes
    # keep an mapping of new_edge_id to original edge_id for edge_size query.
    new_edge_id_2_original_edge_id = {}

    # do the expansion for all annotated he_id in the original edge_index
#     ipdb.set_trace()
    for he_idx in range(num_nodes, num_edges + num_nodes):
        # find all nodes within the same hyperedge.
        selected_he = edge_index[:, edge_index[1] == he_idx]
        size_of_he = selected_he.shape[1]

#         Trim a hyperedge if its size>edge_th
        if edge_th > 0:
            if size_of_he > edge_th:
                continue

        if size_of_he == 1:
            # there is only one node in this hyperedge -> self-loop node. add to graph.
            #             n2he_with_same_heid.append(selected_he)

            new_n2he = selected_he.clone()
            new_n2he[1] = cur_he_id
            expanded_n2he_index.append(new_n2he)

            # ====
#             new_he2n_same_heid = torch.flip(selected_he, dims = [0])
#             he2n_with_same_heid.append(new_he2n_same_heid)

#             new_he2n = torch.flip(selected_he, dims = [0])
#             new_he2n[0] = cur_he_id
#             expanded_he2n_index.append(new_he2n)

            cur_he_id += 1
            continue

        # -------------------------------
#         # new_n2he_same_heid uses same he id for all nodes.
#         new_n2he_same_heid = selected_he.repeat_interleave(size_of_he - 1, dim = 1)
#         n2he_with_same_heid.append(new_n2he_same_heid)

        # for new_n2he mapping. connect the nodes to all repeated he first.
        # then remove those connection that corresponding to the node itself.
        new_n2he = selected_he.repeat_interleave(size_of_he, dim=1)

        # new_edge_ids start from the he_id from previous iteration (cur_he_id).
        new_edge_ids = torch.LongTensor(
            np.arange(cur_he_id, cur_he_id + size_of_he)).repeat(size_of_he)
        new_n2he[1] = new_edge_ids

        # build a mapping between node and it's corresponding edge.
        # e.g. {n_1: e_7_1, n_2: e_7_2}
        tmp_node_id_2_he_id_dict = {}
        for idx in range(size_of_he):
            new_edge_id_2_original_edge_id[cur_he_id] = he_idx
            cur_node_id = selected_he[0][idx].item()
            tmp_node_id_2_he_id_dict[cur_node_id] = cur_he_id
            cur_he_id += 1

        # create n2he by deleting the self-product edge.
        new_he_select_mask = torch.BoolTensor([True] * new_n2he.shape[1])
        for col_idx in range(new_n2he.shape[1]):
            tmp_node_id, tmp_edge_id = new_n2he[0, col_idx].item(
            ), new_n2he[1, col_idx].item()
            if tmp_node_id_2_he_id_dict[tmp_node_id] == tmp_edge_id:
                new_he_select_mask[col_idx] = False
        new_n2he = new_n2he[:, new_he_select_mask]
        expanded_n2he_index.append(new_n2he)


#         # ---------------------------
#         # create he2n from mapping.
#         new_he2n = np.array([[he_id, node_id] for node_id, he_id in tmp_node_id_2_he_id_dict.items()])
#         new_he2n = torch.from_numpy(new_he2n.T).to(device = edge_index.device)
#         expanded_he2n_index.append(new_he2n)

#         # create he2n with same heid as input edge_index.
#         new_he2n_same_heid = torch.zeros_like(new_he2n, device = edge_index.device)
#         new_he2n_same_heid[1] = new_he2n[1]
#         new_he2n_same_heid[0] = torch.ones_like(new_he2n[0]) * he_idx
#         he2n_with_same_heid.append(new_he2n_same_heid)

    new_edge_index = torch.cat(expanded_n2he_index, dim=1)
#     new_he2n_index = torch.cat(expanded_he2n_index, dim = 1)
#     new_edge_index = torch.cat([new_n2he_index, new_he2n_index], dim = 1)
    # sort the new_edge_index by first row. (node_ids)
    new_order = new_edge_index[0].argsort()
    data.edge_index = new_edge_index[:, new_order]

    return data

def norm_contruction(data, option='all_one', TYPE='V2E'):
    if TYPE == 'V2E':
        if option == 'all_one':
            data.norm = torch.ones_like(data.edge_index[0])

        elif option == 'deg_half_sym':
            edge_weight = torch.ones_like(data.edge_index[0])
            cidx = data.edge_index[1].min()
            Vdeg = scatter_add(edge_weight, data.edge_index[0], dim=0)
            HEdeg = scatter_add(edge_weight, data.edge_index[1]-cidx, dim=0)
            V_norm = Vdeg**(-1/2)
            E_norm = HEdeg**(-1/2)
            data.norm = V_norm[data.edge_index[0]] * \
                E_norm[data.edge_index[1]-cidx]

    elif TYPE == 'V2V':
        data.edge_index, data.norm = gcn_norm(
            data.edge_index, data.norm, add_self_loops=True)
    return data

def ConstructV2V(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.edge_index)
    """
    For each he, clique-expansion. Note that we DONT allow duplicated edges.
    Instead, we record its corresponding weights.
    We default no self loops so far.
    """
# # Note that the method below for CE can be memory expensive!!!
#     new_edge_index = []
#     for he in np.unique(edge_index[1, :]):
#         nodes_in_he = edge_index[0, :][edge_index[1, :] == he]
#         if len(nodes_in_he) == 1:
#             continue #skip self loops
#         combs = combinations(nodes_in_he,2)
#         for comb in combs:
#             new_edge_index.append([comb[0],comb[1]])


#     new_edge_index, new_edge_weight = torch.tensor(new_edge_index).type(torch.LongTensor).unique(dim=0,return_counts=True)
#     data.edge_index = new_edge_index.transpose(0,1)
#     data.norm = new_edge_weight.type(torch.float)

# # Use the method below for better memory complexity
    edge_weight_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = np.sort(edge_index[0, :][edge_index[1, :] == he])
        if len(nodes_in_he) == 1:
            continue  # skip self loops
        combs = combinations(nodes_in_he, 2)
        for comb in combs:
            if not comb in edge_weight_dict.keys():
                edge_weight_dict[comb] = 1
            else:
                edge_weight_dict[comb] += 1

# # Now, translate dict to edge_index and norm
#
    new_edge_index = np.zeros((2, len(edge_weight_dict)))
    new_norm = np.zeros((len(edge_weight_dict)))
    cur_idx = 0
    for edge in edge_weight_dict:
        new_edge_index[:, cur_idx] = edge
        new_norm[cur_idx] = edge_weight_dict[edge]
        cur_idx += 1

    data.edge_index = torch.tensor(new_edge_index).type(torch.LongTensor)
    data.norm = torch.tensor(new_norm).type(torch.FloatTensor)
    return data

def generate_norm_HNHN(H, data, args):
    """
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
#     H = data.incident_mat
    alpha = args.HNHN_alpha
    beta = args.HNHN_beta
    H = np.array(H)

    # the degree of the node
    DV = np.sum(H, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    num_nodes = data.n_x[0]
    num_hyperedges = int(data.totedges)
    # alpha part
    D_e_alpha = DE ** alpha
    D_v_alpha = np.zeros(num_nodes)
    for i in range(num_nodes):
        # which edges this node is in
        he_list = np.where(H[i] == 1)[0]
        D_v_alpha[i] = np.sum(DE[he_list] ** alpha)

    # beta part
    D_v_beta = DV ** beta
    D_e_beta = np.zeros(num_hyperedges)
    for i in range(num_hyperedges):
        # which nodes are in this hyperedge
        node_list = np.where(H[:, i] == 1)[0]
        D_e_beta[i] = np.sum(DV[node_list] ** beta)

    D_v_alpha_inv = 1.0 / D_v_alpha
    D_v_alpha_inv[D_v_alpha_inv == float("inf")] = 0

    D_e_beta_inv = 1.0 / D_e_beta
    D_e_beta_inv[D_e_beta_inv == float("inf")] = 0

    data.D_e_alpha = torch.from_numpy(D_e_alpha).float()
    data.D_v_alpha_inv = torch.from_numpy(D_v_alpha_inv).float()
    data.D_v_beta = torch.from_numpy(D_v_beta).float()
    data.D_e_beta_inv = torch.from_numpy(D_e_beta_inv).float()

    return data

def ConstructH_HNHN(data):
    """
    Construct incidence matrix H of size (num_nodes, num_hyperedges) from edge_index = [V;E]
    """
    edge_index = np.array(data.edge_index)
    num_nodes = data.n_x[0]
    num_hyperedges = int(data.totedges)
    H = np.zeros((num_nodes, num_hyperedges))
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.
        cur_idx += 1

#     data.incident_mat = H
    return H

def ConstructH(data):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
#     ipdb.set_trace()
    edge_index = np.array(data.edge_index)
    cidx = edge_index[1].min()
    edge_index[1] -= cidx  # make sure we do not waste memory
    # Don't use edge_index[0].max()+1, as some nodes maybe isolated
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(edge_index[1])-np.min(edge_index[1])+1
    shape = (num_nodes, num_hyperedges)
    H = coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=shape)

    data.edge_index = H
    return data

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        #         ipdb.set_trace()
        indices = []
        for i in range(label.max()+1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop/(label.max()+1)*len(label))
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx

def cal_the_normalization_diag_matrix(A):

    out_degree = A.coalesce().values().view(-1)

    degree_matrix = torch.sparse_coo_tensor(indices=torch.stack([A.coalesce().indices()[0], A.coalesce().indices()[0]]),
                                            values=out_degree,
                                            size=(A.size(0), A.size(0))).coalesce()

    degree_values = degree_matrix.values()

    degree_values_sqrt_inv = 1.0 / torch.sqrt(degree_values)

    return torch.sparse_coo_tensor(indices=degree_matrix.indices(),
                                                     values=degree_values_sqrt_inv,
                                                     size=degree_matrix.size())

def get_HyperGCN_He_dict(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.edge_index)
    """
    For each he, clique-expansion. Note that we allow the weighted edge.
    Note that if node pair (vi,vj) is contained in both he1, he2, we will have (vi,vj) twice in edge_index. (weighted version CE)
    We default no self loops so far.
    """
# #     Construct a dictionary
#     He2V_List = []
# #     Sort edge_index according to he_id
#     _, sorted_idx = torch.sort(edge_index[1])
#     edge_index = edge_index[:,sorted_idx].type(torch.LongTensor)
#     current_heid = -1
#     for idx, he_id in enumerate(edge_index[1]):
#         if current_heid != he_id:
#             current_heid = he_id
#             if idx != 0 and len(he2v)>1: #drop original self loops
#                 He2V_List.append(he2v)
#             he2v = []
#         he2v.append(edge_index[0,idx].item())
# #     Remember to append the last he
#     if len(he2v)>1:
#         He2V_List.append(he2v)
# #     Now, turn He2V_List into a dictionary
    edge_index[1, :] = edge_index[1, :]-edge_index[1, :].min()
    He_dict = {}
    for he in np.unique(edge_index[1, :]):
        #         ipdb.set_trace()
        nodes_in_he = list(edge_index[0, :][edge_index[1, :] == he])
        He_dict[he.item()] = nodes_in_he

#     for he_id, he in enumerate(He2V_List):
#         He_dict[he_id] = he

    return He_dict

def extract_Hypergraph_multi_hop_neighbors(args, data):
    """generate multi-hop neighbors for input PyG graph using shortest path distance kernel
    Args:
        data (torch_geometric.data.Data): PyG graph data instance
        K (int): number of hop
        max_edge_attr_num (int): maximum number of encoding used for hopk edge
        max_hop_num (int): maximum number of hop to consider in computing node configuration of peripheral subgraph
        max_edge_type (int): maximum number of edge type to consider
        max_edge_count (int): maximum number of count for each type of edge
        max_distance_count (int): maximum number of count for each distance
        kernel (str): kernel used to extract neighbors
    """
    assert (isinstance(data, Data))
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes
    # graph with no edge
    if edge_index.size(1) == 0:
        edge_matrix_size = [num_nodes, args.K, args.max_edge_type, 2]
        configuration_matrix_size = [num_nodes, args.K, args.max_hop_num]
        peripheral_edge_matrix = torch.zeros(edge_matrix_size, dtype=torch.long)
        peripheral_configuration_matrix = torch.zeros(configuration_matrix_size, dtype=torch.long)
        data.peripheral_edge_attr = peripheral_edge_matrix
        data.peripheral_configuration = peripheral_configuration_matrix
        return data

    # calculate the induced matrix of hypergraph
    # consider the self-loop condition

    num_nodes = data.x.shape[0]
    num_hyperedges = torch.unique(data.edge_index[1]).size(0)
    shape = (num_nodes, num_hyperedges)
    cidx = data.edge_index[1].min()
    data.edge_index[1] -= cidx  # make sure we do not waste memory
    hypergraph_adj = coo_matrix((np.ones(edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])), shape=shape)
    line_hypergraph_adj = hypergraph_adj.transpose().dot(hypergraph_adj).astype(np.float32)
    # set all the value to 1,if > 1
    line_hypergraph_adj.data = np.minimum(line_hypergraph_adj.data, 1)
    # compute each order of adj
    line_hypergraph_adj_list = adj_K_order(args, line_hypergraph_adj, args.K)

    if "edge_attr" in data:
        hyper_edge_attr = data.edge_attr
    else:
        # skip 0, 1 as it is the mask and self-loop defined in the model
        hyper_edge_attr = (torch.ones([line_hypergraph_adj.shape[0]]) * 2)


    edge_attr_adj = torch.from_numpy(to_scipy_sparse_matrix(edge_index, edge_attr, num_nodes).toarray()).long()

    if args.kernel == "gd":
        # create K-hop edge with graph diffusion kernel
        final_adj = 0
        for adj_ in line_hypergraph_adj_list:
            final_adj += adj_
        final_adj.data = np.minimum(final_adj.data, 1)
    else:
        # process adj list to generate shortest path distance matrix with path number
        exist_adj = c(torch.from_numpy(line_hypergraph_adj_list[0].toarray()).int())
        for i in range(1, len(line_hypergraph_adj_list)):
            adj_ = c(torch.from_numpy(line_hypergraph_adj_list[i].toarray()).int())
            adj_[exist_adj > 0] = 0
            exist_adj = exist_adj + adj_
            exist_adj[exist_adj > 1] = 1
            line_hypergraph_adj_list[i] = csr_matrix(adj_)
            # mask all the edge that already exist in previous hops, too long to wait
            # nonzero_indices = exist_adj.nonzero()
            # for row, col in zip(nonzero_indices[0], nonzero_indices[1]):
            #     adj_.data[np.where((adj_.row == row) & (adj_.col == col))] = 0
            # exist_adj = exist_adj + adj_
            # exist_adj.data = np.minimum(exist_adj.data, 1)
            # line_hypergraph_adj_list[i] = adj_
        # create K-hop edge with sortest path distance kernel
        final_adj = csr_matrix(exist_adj)

        del exist_adj
        del adj_

    (row, col), value = torch_sparse.from_scipy(final_adj)
    edge_index = torch.stack([row, col], dim=0)

    hop1_edge_attr = edge_attr_hypergraph_adj[edge_index[0, :], edge_index[1, :]]
    # edge_attr_list = [hop1_edge_attr.unsqueeze(-1)]
    pe_attr_list = []
    for i in range(1, len(line_hypergraph_adj_list)):
        adj_ = c(line_hypergraph_adj_list[i])
        adj_[adj_ > args.max_pe_num] = args.max_pe_num
        # skip 1 as it is the self-loop defined in the model
        adj_[adj_ > 0] = adj_[adj_ > 0] + 1
        adj_ = adj_.long()
        hopk_edge_attr = adj_[edge_index[0, :], edge_index[1, :]].unsqueeze(-1)
        edge_attr_list.append(hopk_edge_attr)
        pe_attr_list.append(torch.diag(adj_).unsqueeze(-1))
    edge_attr = torch.cat(edge_attr_list, dim=-1)  # E * K
    if args.K > 1:
        pe_attr = torch.cat(pe_attr_list, dim=-1)  # N * K-1
    else:
        pe_attr = None

    peripheral_edge_attr, peripheral_configuration_attr = get_peripheral_attr(adj_list, edge_attr_adj, args.max_hop_num,
                                                                              args.max_edge_type, args.max_edge_count,
                                                                              args.max_distance_count)
    # update all the attributes
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.peripheral_edge_attr = peripheral_edge_attr
    data.peripheral_configuration_attr = peripheral_configuration_attr
    data.pe_attr = pe_attr
    return data

def adj_K_order(args, adj, K):
    """compute the K order of adjacency given scipy matrix
    adj (coo_matrix): adjacency matrix
    K (int): number of hop
    """
    # use gpu to speed up
    (row, col), value = torch_sparse.from_scipy(adj)
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), value, torch.Size(adj.shape)).to('cuda:'+str(args.cuda))

    adj_list = [adj.clone()]
    for i in range(K - 1):
        adj_ = torch.sparse.mm(adj_list[-1], adj)
        adj_list.append(adj_)
    for i, adj_ in enumerate(adj_list):
        adj_ = adj_.to("cpu")
        row_indices, col_indices = adj_.coalesce().indices()
        values = adj_.coalesce().values()
        adj_ = coo_matrix((values, (row_indices, col_indices)), shape=adj_.shape)
        adj_.setdiag(0)
        # set all value to 1 if > 1
        adj_.data = np.minimum(adj_.data, 1)
        adj_list[i] = adj_
    # # use cpu
    # adj_list = [c(adj)]
    # for i in range(K - 1):
    #     adj_ = adj_list[-1].dot(adj)
    #     adj_list.append(adj_)
    # for i, adj_ in enumerate(adj_list):
    #     adj_.setdiag(0)
    #     adj_list[i] = adj_

    return adj_list

def get_peripheral_attr(adj_list, edge_attr_adj, max_hop_num,
                        max_edge_type, max_edge_count, max_distance_count):
    """Compute peripheral information for each node in graph
    Args:
        adj_list (list): adjacency matrix list of data for each hop
        edge_attr_adj (torch.tensor):edge feature matrix
        max_hop_num (int): maximum number of hop to consider in computing node configuration of peripheral subgraph
        max_edge_type (int): maximum number of edge type to consider
        max_edge_count (int): maximum number of count for each type of edge
        max_distance_count (int): maximum number of count for each distance
    """
    K = len(adj_list)
    num_nodes = edge_attr_adj.size(0)
    if max_hop_num > 0 and max_edge_type > 0:
        peripheral_edge_matrix_list = []
        peripheral_configuration_matrix_list = []
        for i in range(K):
            adj_ = c(adj_list[i])
            peripheral_edge_matrix, peripheral_configuration_matrix = extract_peripheral_attr_v2(edge_attr_adj, adj_,
                                                                                                 max_hop_num,
                                                                                                 max_edge_type,
                                                                                                 max_edge_count,
                                                                                                 max_distance_count)
            peripheral_edge_matrix_list.append(peripheral_edge_matrix)
            peripheral_configuration_matrix_list.append(peripheral_configuration_matrix)

        peripheral_edge_attr = torch.cat(peripheral_edge_matrix_list, dim=0)
        peripheral_configuration_attr = torch.cat(peripheral_configuration_matrix_list, dim=0)
        peripheral_edge_attr = peripheral_edge_attr.transpose(0, 1)  # N * K * c * f
        peripheral_configuration_attr = peripheral_configuration_attr.transpose(0, 1)  # N * K * c * f
    else:
        peripheral_edge_attr = None
        peripheral_configuration_attr = None

    return peripheral_edge_attr, peripheral_configuration_attr

def extract_peripheral_attr_v2(adj, k_adj, max_hop_num, max_edge_type, max_edge_count, max_distance_count):
    """extract peripheral attr information for each node using adj at this hop and original adj
    Args:
        adj (torch.tensor): adjacency matrix of original graph N*N, different number means different edge type
        k_adj (torch.tensor): adjacency matrix at the hop we want to extract peripheral information N*N
        max_hop_num (int): maximum number of hop to consider in computing node configuration of peripheral subgraph
        max_edge_type (int): maximum number of edge type to consider
        max_edge_count (int): maximum number of count for each type of edge
        max_distance_count (int): maximum number of count for each distance
    """
    num_nodes = adj.size(0)

    # component_dim=max_component_num
    # record peripheral edge information
    edge_matrix_size = [num_nodes, max_edge_type, 2]
    peripheral_edge_matrix = torch.zeros(edge_matrix_size, dtype=torch.long)
    # record node configuration
    configuration_matrix_size = [num_nodes, max_hop_num + 1]
    peripheral_configuration_matrix = torch.zeros(configuration_matrix_size, dtype=torch.long)
    for i in range(num_nodes):
        row = torch.where(k_adj[i] > 0)[0]
        # subgrapb with less than 2 nodes, no edges, thus skip
        num_sub_nodes = row.size(-1)
        if num_sub_nodes < 2:
            continue
        peripheral_subgraph = adj[row][:, row]
        s = nx.from_numpy_array(peripheral_subgraph.numpy(), create_using=nx.DiGraph)
        s_edge_list = list(nx.get_edge_attributes(s, "weight").values())
        if len(s_edge_list) == 0:
            continue
        s_edge_list = torch.tensor(s_edge_list).long()
        edge_count = torch.bincount(s_edge_list, minlength=max_edge_type + 2)
        # remove 0 and 1
        edge_count = edge_count[2:]
        sort_count, sort_type = torch.sort(edge_count, descending=True)
        sort_count = sort_count[:max_edge_type]
        sort_type = sort_type[:max_edge_type]
        sort_count[sort_count > max_edge_count] = max_edge_count
        peripheral_edge_matrix[i, :, 0] = sort_type
        peripheral_edge_matrix[i, :, 1] = sort_count
        shortest_path_matrix = nx_compute_shortest_path_length(s, max_length=max_hop_num)
        num_sub_p_edges = 0
        for j in range(num_sub_nodes):
            for h in range(1, max_hop_num + 1):
                h_nodes = torch.where(shortest_path_matrix[j] == h)[0]
                if h_nodes.size(-1) < 2:
                    continue
                else:
                    pp_subgraph = peripheral_subgraph[h_nodes][:, h_nodes]
                    num_sub_p_edges += torch.sum(pp_subgraph)

        configuration_feature = torch.bincount(shortest_path_matrix.view(-1), minlength=max_hop_num + 1)
        # configuration_feature=configuration_feature[1:]
        configuration_feature[0] = num_sub_p_edges
        configuration_feature[configuration_feature > max_distance_count] = max_distance_count
        peripheral_configuration_matrix[i, :] = configuration_feature
    return peripheral_edge_matrix.unsqueeze(0), peripheral_configuration_matrix.unsqueeze(0)

def nx_compute_shortest_path_length(G, max_length):
    """Compute all pair shortest path length in the graph
    Args:
        G (networkx): input graph
        max_length (int): max length when computing shortest path
    """
    num_node = G.number_of_nodes()
    shortest_path_length_matrix = torch.zeros([num_node, num_node]).int()
    all_shortest_path_lengths = nx.all_pairs_shortest_path_length(G, max_length)
    for shortest_path_lengths in all_shortest_path_lengths:
        index, path_lengths = shortest_path_lengths
        for end_node, path_length in path_lengths.items():
            if end_node == index:
                continue
            else:
                shortest_path_length_matrix[index, end_node] = path_length
    return shortest_path_length_matrix