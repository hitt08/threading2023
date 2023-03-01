import pickle
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import argparse
import cupy as cp
import os
import networkx as nx
import torch

from hint.npc import sp_split_nw_comm
from hint.weight_functions import cos_time, cosine_similarity_custom, get_nxe_weight_edges
from utils.data_utils import load_config, load_hyperparams, read_parts, read_dict_dump, read, read_dict
from utils.thread_utils import get_graph_threads, create_entity_graph


def get_connected_nodes(DG):
    return [DG.subgraph(cc).nodes for cc in nx.connected_components(DG.to_undirected())]


class GraphThreadGenerator:
    def __init__(self, hyperparams=None, is_td=False, is_ent=False, method="npc", fwd_only=True, use_gpu=True):

        self.is_td = is_td
        self.alpha = hyperparams.alpha
        self.is_ent = is_ent
        self.gamma = hyperparams.gamma
        self.max_common_ent = hyperparams.max_common_ent
        self.max_ent_bet = hyperparams.max_ent_bet
        self.weight_threshold = hyperparams.weight_threshold
        self.method = method
        self.fwd_only = fwd_only  # If True: Edges are always defined foward in time (Set to FALSE for continous runs to extend already identified threads)
        self.use_gpu = use_gpu
        self.log = logging.getLogger()

    def find_inner_communities(self, DG, communities, th="auto", outdegree_limit=-1):
        res_graph = nx.DiGraph()
        comms = communities

        for c in comms:
            gres = sp_split_nw_comm(c, DG, th=th, outdegree_limit=outdegree_limit, logging=False, return_graph=True)
            res_graph = nx.compose(res_graph, gres)

        return res_graph

    def create_doc_graph(self, train_doc_ids, train_emb, train_emb_y=None, td_T=None, ent_graph=None, batch_size=128,
                         DG=None, batch_start=0, verbose=True):

        pool_iter = []
        st = 0
        en = batch_size
        b = 0
        while st < train_emb.shape[0]:
            pool_iter.append((b, train_emb[st:en]))
            st = en
            en = st + batch_size
            b += 1

        edges = []

        if verbose:
            pbar = tqdm(total=train_emb.shape[0])
            print_newline_batch = 1000
            print_newline_cnt = 0

        if self.fwd_only:
            train_emb_y = train_emb

        for idx, x in pool_iter:
            if self.is_td:
                item_weights = cos_time(x, train_emb_y, T=td_T, alpha=self.alpha, use_gpu=self.use_gpu)
            else:
                item_weights = 1 - cosine_similarity_custom(x, train_emb_y, use_gpu=self.use_gpu)

            if self.use_gpu:
                item_weights = cp.asnumpy(item_weights)
            temp = []
            candidates = np.argwhere(item_weights < self.weight_threshold)
            for s in range(item_weights.shape[0]):
                source_node_id = idx * batch_size + s + batch_start
                neighbours = candidates[candidates[:, 0] == s, 1]

                if self.fwd_only:
                    nxt_neighbours = neighbours[neighbours > source_node_id]
                    prev_neighbours = []
                else:
                    nxt_neighbours = neighbours[neighbours > source_node_id]
                    prev_neighbours = neighbours[neighbours < source_node_id]

                source = train_doc_ids[source_node_id]

                if len(nxt_neighbours) > 0:
                    targets = [train_doc_ids[n] for n in nxt_neighbours]
                    if self.is_ent:
                        new_edges = get_nxe_weight_edges(ent_graph, source, targets, item_weights[s, nxt_neighbours],
                                                         gamma=self.gamma, max_common_ent=self.max_common_ent,
                                                         max_ent_between=self.max_ent_bet)
                    else:
                        new_edges = list(zip([source] * len(targets), targets, item_weights[s, nxt_neighbours]))
                    temp.extend(new_edges)

                if not self.fwd_only and len(prev_neighbours) > 0:
                    targets = [train_doc_ids[n] for n in prev_neighbours]
                    if self.is_ent:
                        new_edges = get_nxe_weight_edges(ent_graph, source, targets, item_weights[s, prev_neighbours],
                                                         gamma=self.gamma, max_common_ent=self.max_common_ent,
                                                         max_ent_between=self.max_ent_bet, reverse=True)
                    else:
                        new_edges = list(zip(targets, [source] * len(targets), item_weights[s, prev_neighbours]))
                    temp.extend(new_edges)

                if verbose:
                    pbar.update()
                    print_newline_cnt += 1

            edges.extend(temp)
            if verbose:
                pbar.set_description(f"Number of Edges: {len(edges)}")
                if print_newline_cnt > print_newline_batch:
                    self.log.warning("")
                    print_newline_cnt = 0

        if DG is None:
            DG = nx.DiGraph()
        DG.add_weighted_edges_from(edges)
        return DG

    def get_threads(self, ofile, train_doc_ids, train_emb, passage_doc_ids, passage_emb, date_data, dt_feat=None,
                    ent_graph=None, force_graph_create=False, verbose=True):
        doc2id = dict([(doc_id, idx) for idx, doc_id in enumerate(passage_doc_ids)])

        if self.is_td:
            train_emb = np.hstack((train_emb, dt_feat[:, None]))  # term features, dt_feat
            td_T = np.max(dt_feat) - np.min(dt_feat)
        else:
            td_T = None

        if self.use_gpu:
            train_emb = cp.asarray(train_emb)

        self.log.warning(
            f"\tTrain Data: {train_emb.shape}. TD: {self.is_td}, ENT: {self.is_ent}, Continous: {not self.fwd_only}")

        if os.path.exists(ofile) and not force_graph_create:
            self.log.warning(f"\tDG found at: {ofile}")
            DG = nx.read_gpickle(ofile)
        else:
            self.log.warning(f"\tCreating Graph")
            DG = self.create_doc_graph(train_doc_ids, train_emb, td_T=td_T, ent_graph=ent_graph, verbose=verbose)
            nx.write_gpickle(DG, ofile)

        if DG.number_of_edges()<1:
            self.log.warning(f"\tSKIPPING: number of edges should be greater than 0.")
            return None


        self.log.warning(f"\tRunning {self.method.upper()}")
        if self.method == "npc":
            res_graph = self.find_inner_communities(DG, [DG.nodes], th="auto", outdegree_limit=1)
            communities = get_connected_nodes(res_graph)
        elif self.method in ["louvain", "leiden"]:
            res_graph = None
            from cdlib import algorithms

            for k, v in tqdm(nx.get_edge_attributes(DG, "weight").items()):
                if v < 0:
                    DG.add_edge(k[0], k[1], weight=0)
            UDG = DG.to_undirected()

            communities = algorithms.louvain(UDG, weight="weight") if self.method == "louvain" else algorithms.leiden(
                UDG, weights="weight")
            communities = communities.communities
        else:
            self.log.error(f"\tInvalid community detection method: {self.method}")
            return

        self.log.warning(f"\tGenerating Threads")
        threads, thread_similarity = get_graph_threads(communities, date_data, passage_emb, doc2id)

        return res_graph, threads, thread_similarity


def run_hint(config=None, hyperparams=None, emb="tfidflsa", time_decay=True, ent_similarity=True,
             community_detection="npc", force_graph_create=False,use_gpu=True):
    log = logging.getLogger(__name__)

    if config is None:
        config = load_config()
    if hyperparams is None:
        hyperparams = load_hyperparams()

    passage_doc_ids = read(config.passage_docids_file, compress=True)
    data5w1h_doc_ids = read(config.data5w1h_docids_file, compress=True)

    with np.load(os.path.join(config.emb_dir, f"{emb}_5w1h_emb.npz")) as data:
        data5w1h_emb = data["arr_0"]

    date_data = pd.to_datetime(pd.Series(read_dict(config.date_dict_file, compress=True))).to_dict()

    with np.load(config.date_features_file) as data:
        dt_feat = data["arr_0"]

    with np.load(os.path.join(config.emb_dir, f"psg_{emb}_emb.npz")) as data:
        passage_emb = data["arr_0"]

    doc_5w1h_parts = read_parts(config.data5w1h_parts_file)

    entity_dict = read_dict_dump(config.entity_dict_file)

    # #######################################################################################################
    # #######################################################################################################

    threads, thread_similarity = [], []
    res_graph = nx.DiGraph()
    pattern = f"{emb}_w{hyperparams.weight_threshold}"
    if time_decay:
        pattern += f"_td{hyperparams.alpha}"
    if ent_similarity:
        pattern += f"_ent{hyperparams.gamma}"

    for p_id, (p_st, p_en) in doc_5w1h_parts.items():
        log.warning(f"Part: {p_id + 1}")
        train_doc_ids = data5w1h_doc_ids[p_st:p_en]
        train_5w1h_emb = data5w1h_emb[p_st:p_en]
        train_dt_emb = dt_feat[p_st:p_en]

        if ent_similarity:
            G = create_entity_graph(entity_dict, train_doc_ids)
            log.warning(f"\tEntity Graph: Number of Nodes= {G.number_of_nodes()}. Number of Edges= {G.number_of_edges()}")
        else:
            G = None

        graph_ofile = os.path.join(config.hint_dir, f"psg_graph_{pattern}_{p_id}.gp")

        use_gpu = use_gpu and torch.cuda.is_available()

        gt = GraphThreadGenerator(hyperparams=hyperparams, is_td=time_decay, is_ent=ent_similarity,
                                  method=community_detection, use_gpu=use_gpu)

        res = gt.get_threads(ofile=graph_ofile, train_doc_ids=train_doc_ids,
                                                                       train_emb=train_5w1h_emb,
                                                                       passage_doc_ids=passage_doc_ids,
                                                                       passage_emb=passage_emb,
                                                                       date_data=date_data, dt_feat=train_dt_emb,
                                                                       ent_graph=G,
                                                                       force_graph_create=force_graph_create)

        if res is None:
            continue
        else:
            DG_comm, part_threads, part_thread_similarity=res
        threads.extend(part_threads)
        thread_similarity.append(part_thread_similarity)
        if community_detection == "npc":
            res_graph = nx.compose(res_graph, DG_comm)

    thread_similarity = np.hstack(thread_similarity)

    pattern = f"{community_detection}_{pattern}"

    ofile = os.path.join(config.threads_dir, f"threads_hint_{pattern}.p")
    with open(ofile, "wb") as f:
        pickle.dump(threads, f)
    log.warning(f"\tThreads stored at: {ofile}")
    np.savez_compressed(os.path.join(config.threads_dir, f"thread_sim_hint_{pattern}.npz"), thread_similarity)
    nx.write_gpickle(res_graph, os.path.join(config.hint_dir, f"{pattern}_graph.gp"))

    res = f"\nThread Count:{len(threads)}"
    res += f"\nAvg Length:{np.mean([len(i) for i in threads])}"
    res += f"\nCosine Score: {np.mean(thread_similarity[np.logical_not(np.isnan(thread_similarity))])}"
    log.warning(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', action='store', dest='emb', type=str, required=True,
                        help='embedding model: minilm; roberta;tfidflsa')

    parser.add_argument('--td', dest='td', action='store_true', help='time decay')
    parser.set_defaults(td=False)
    parser.add_argument('--ent', dest='ent', action='store_true', help='entity similarity')
    parser.set_defaults(ent=False)

    args = parser.parse_args()

    log = logging.getLogger(__name__)
    log.warning(args)

    run_hint(emb=args.emb, time_decay=args.td, ent_similarity=args.ent)
