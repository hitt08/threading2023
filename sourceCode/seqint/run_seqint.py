import os
import pickle
import logging
import argparse
import time
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import pandas as pd

from utils.data_utils import load_hyperparams,load_config, read, read_dict, read_dict_dump, read_parts
from utils.thread_utils import ev_thread_score,create_entity_graph
from utils.weight_functions import shortest_path_length,get_all_direct_paths



log = logging.getLogger(__name__)


alpha = 0
gamma = 0
mCE = 0  #Max common entities between any passage pair
mEB = 0  #Max entites between in the shortest path between any passage pair
w = 0 #Weight Threshold
G=None
doc_ids = None


def time_decay(x):  # Similarity
    global alpha
    T = np.max(x) - np.min(x)
    # y=np.full((x.shape[0],x.shape[0]),x)
    return np.exp(-alpha * np.abs(x[:, None] - x) / T)


def cos_time(a):  # distance
    tmp = cosine_similarity(a[:, :-1]) * time_decay(a[:, -1])
    cond = (1 - tmp) > w #weight threshold
    tmp[cond] = 0
    return 1 - tmp#cosine_similarity(a[:, :-1]) * time_decay(a[:, -1])


def cos_ent(a):  # distance
    tmp=cosine_similarity(a[:, :-1])
    cond = (1 - tmp) > w  # weight threshold
    tmp[cond] = 0
    cond = np.logical_not(cond)
    tmp[cond] = tmp[cond] * nxe_weight(a[:, -1], cond)
    return 1 -  tmp


def cos_ent_time(a):  # distance
    log = logging.getLogger(__name__)
    tmp = cosine_similarity(a[:, :-2]) * time_decay(a[:, -2])
    cond = (1 - tmp) > w #weight threshold
    tmp[cond] = 0
    cond = np.logical_not(cond)
    tmp[cond] = tmp[cond] * nxe_weight(a[:, -1], cond)

    return 1 - tmp

    # return 1 - cosine_similarity(a[:, :-2]) * time_decay(a[:, -2]) * nxe_weight(a[:, -1])


def nxe_weight(x,cond,verbose=False):  # Similarity
    global gamma,mCE,mEB,G,doc_ids
    log = logging.getLogger(__name__)

    items = doc_ids[x.astype(int)]
    # print(items)
    # temp=dict([(t, 0) for t in targets])
    ent_similarity = np.zeros((len(items), len(items)))

    if verbose:
        pbar=tqdm(total=len(items))

    for idx, s in enumerate(items):
        s_cond = cond[idx].copy()
        s_cond[:idx] = False #Compute only once for a pair of documents
        targets = items[s_cond]

        if verbose:
            pbar.set_description(f"Targets: {len(targets)}")

        temp = get_all_direct_paths(G, s, targets)


        # print(targets)
        # temp = dict([(t, len(list(nx.all_simple_paths(G, s, t, cutoff=2)))) for t in targets])
        # temp[t]==len(list(nx.all_simple_paths(G, s, t, cutoff=2)))

        targets_k, targets_p = [], []
        for t in targets:
            targets_k.append(t)
            targets_p.append(temp[t])
        targets_k = np.asarray(targets_k)
        targets_p = np.asarray(targets_p)

        is_path_weight = targets_p == 0
        ent_weight_args = np.argwhere(is_path_weight == False).squeeze(-1)
        path_weight_args = np.argwhere(is_path_weight).squeeze(-1)
        # print(targets_p,is_path_weight,path_weight_args,path_weight_args.shape)#,len(path_weight_args))
        res = np.zeros(len(targets))

        ent_weights = targets_p[ent_weight_args]

        # path_weights = np.asarray(get_shortest_path_lengths(G, s, targets_k[path_weight_args], cutoff=mEB * 2))

        path_weights = np.zeros_like(path_weight_args)
        for i, t in enumerate(targets_k[path_weight_args]):
            try:
                path_weights[i] = shortest_path_length(G, s, t, cutoff=mEB * 2)
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                path_weights[i] = mEB * 2  # G.number_of_nodes()

        ent_weights = 0.5 * (1 + (1 - np.exp(-gamma * (ent_weights / mCE))))
        # path_weights = 1 - (1 - item_weights[path_weight_args]) * 0.5 * (1 - np.log(path_weights / 2) / np.log(max_ent_between))
        path_weights = 0.5 * np.exp(-gamma * ((path_weights / 2) / mEB))

        res[ent_weight_args] = ent_weights
        res[path_weight_args] = path_weights

        ent_similarity[idx][s_cond] = res

        if verbose:
            pbar.update()

    if verbose:
        pbar.close()

    #Copy values a->b to b->a
    ent_similarity[np.tril_indices(ent_similarity.shape[0])] = ent_similarity[np.tril_indices(ent_similarity.shape[0])[::-1]]

    return ent_similarity[cond]


def get_threads(hac_model, model_doc_ids, date_data, passage_matrix, passage_doc2id):

    threads = []
    thread_similarity = []

    with tqdm(total=hac_model.n_clusters_) as pbar:
        for l in range(hac_model.n_clusters_):
            c_lbl_idx = np.argwhere(hac_model.labels_ == l).squeeze()
            thread = []
            c_lbls = np.array(model_doc_ids)[c_lbl_idx].tolist()
            if type(c_lbls) == str:
                c_lbls = [c_lbls]
            for i in c_lbls:
                thread.append((i, date_data[i]))
            thread = sorted(thread, key=lambda x: x[1])
            # thread_period = thread[-1][1] - thread[0][1]

            thread_sim = np.mean(ev_thread_score(thread, passage_doc2id, passage_matrix)) if len(thread) > 1 else np.nan

            threads.append(thread)
            thread_similarity.append(thread_sim)
            pbar.update()

    thread_similarity = np.asarray(thread_similarity)
    return threads, np.array(thread_similarity)


def get_hac_threads(ofile,n_clusters,distance_threshold,train_doc_ids,train_emb,passage_doc_ids,passage_emb,date_data,hyperparams=None,is_td=False,dt_feat=None, is_ent=False,ent_graph=None,log_time=False):
    global alpha,gamma,mCE,mEB,G,doc_ids,w
    log = logging.getLogger(__name__)
    if log_time:
        run_times={}
    doc2id = dict([(doc_id, idx) for idx, doc_id in enumerate(passage_doc_ids)])

    alpha=hyperparams.alpha
    gamma=hyperparams.gamma
    mCE=hyperparams.max_common_ent
    mEB=hyperparams.max_ent_bet
    w=hyperparams.weight_threshold
    G=ent_graph
    doc_ids=np.asarray(passage_doc_ids)

    ent_doc_ids_feat = np.asarray([doc2id[d] for d in train_doc_ids]) if is_ent else None

    linkage = "complete"
    if is_td and is_ent:
        log.warning("\tTime Decay with Entity Similarity")
        train_emb = np.hstack((train_emb, dt_feat[:, None],ent_doc_ids_feat[:, None]))  # term features, dt_feat, doc_ids
        affinity = cos_ent_time
    elif is_td:
        log.warning("\tTime Decay")
        train_emb = np.hstack((train_emb, dt_feat[:, None]))  # term features, dt_feat
        affinity = cos_time
    elif is_ent:
        log.warning("\tEntity Similarity")
        train_emb = np.hstack((train_emb, ent_doc_ids_feat[:, None]))  # term features, doc_ids
        affinity = cos_ent
    else:
        log.warning("\tWard")
        linkage = "ward"
        affinity = "euclidean"

    log.warning(f"\tClustering. Train Data: {train_emb.shape}")
    if log_time:
        cst=time.time()
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity, distance_threshold=distance_threshold).fit(train_emb)
    if log_time:
        run_times["cluster"]=time.time()-cst
    
    log.warning(f"\tCluster Found: {clustering.n_clusters_}")
    with open(ofile, "wb") as f:
        pickle.dump(clustering, f)
    log.warning(f"\tModel Dumped at: {ofile}")

    log.warning("\tGenerating Threads")
    if log_time:
        tst=time.time()
    threads, thread_similarity = get_threads(clustering, train_doc_ids,date_data, passage_emb, doc2id)
    
    res=[threads, thread_similarity]

    if log_time:
        run_times["thread"]=time.time()-tst
        run_times["total"]=time.time()-cst
        res.append(run_times)

    return res


#Clustering
def run_seqint(config=None,hyperparams=None,emb="tfidflsa",n_clusters=None,distance_threshold=0.8,time_decay=True, ent_similarity=True):
    if config is None:
        config = load_config()
    if hyperparams is None:
        hyperparams = load_hyperparams()


    passage_doc_ids = read(config.passage_docids_file,compress=True)
    data5w1h_doc_ids = read(config.data5w1h_docids_file,compress=True)

    with np.load(os.path.join(config.emb_dir,f"{emb}_5w1h_emb.npz")) as data:
        data5w1h_emb = data["arr_0"]

    date_data = pd.to_datetime(pd.Series(read_dict(config.date_dict_file,compress=True))).to_dict()

    with np.load(config.date_features_file) as data:
        dt_feat = data["arr_0"]

    with np.load(os.path.join(config.emb_dir,f"psg_{emb}_emb.npz")) as data:
        passage_emb = data["arr_0"]

    doc_5w1h_parts = read_parts(config.data5w1h_parts_file)


    entity_dict=read_dict_dump(config.entity_dict_file)


    threads,thread_similarity=[],[]

    pattern = f"{emb}_w{hyperparams.weight_threshold}"
    if time_decay:
        pattern += f"_td{hyperparams.alpha}"
    if ent_similarity:
        pattern += f"_ent{hyperparams.gamma}"
    if not time_decay and not ent_similarity:
        pattern += f"_ward"

    if n_clusters is not None:
        distance_threshold=None
    for p_id, (p_st, p_en) in doc_5w1h_parts.items():
        log.warning(f"\nPart: {p_id + 1}")
        if n_clusters[p_id]<1:
            log.warning(f"\tSKIPPING: n_clusters should be greater than 0. (value provided: {n_clusters[p_id]})")
            continue
        train_doc_ids = data5w1h_doc_ids[p_st:p_en]
        train_5w1h_emb=data5w1h_emb[p_st:p_en]
        train_dt_emb = dt_feat[p_st:p_en]

        if ent_similarity:
            G = create_entity_graph(entity_dict,train_doc_ids)
            log.warning(f"Entity Graph: Number of Nodes= {G.number_of_nodes()}. Number of Edges= {G.number_of_edges()}")
        else:
            G = None

        ofile = os.path.join(config.seqint_dir,f"hac_{pattern}_{p_id}.p")
        part_threads,part_thread_similarity=get_hac_threads(ofile, n_clusters[p_id],distance_threshold, train_doc_ids, train_5w1h_emb, passage_doc_ids, passage_emb, date_data, hyperparams=hyperparams,is_td=time_decay, dt_feat=train_dt_emb, is_ent=ent_similarity, ent_graph=G,log_time=False)

        threads.extend(part_threads)
        thread_similarity.append(part_thread_similarity)

    thread_similarity=np.hstack(thread_similarity)

    ofile = os.path.join(config.threads_dir,f"threads_seqint_{pattern}.p")
    with open(ofile, "wb") as f:
        pickle.dump(threads, f)
    log.warning(f"Threads stored at: {ofile}")
    np.savez_compressed(f"{config.threads_dir}/thread_sim_seqint_{pattern}.npz", thread_similarity)

    res = f"\nThread Count:{len(threads)}"
    res += f"\nAvg Length:{np.mean([len(i) for i in threads])}"
    res += f"\nCosine Score: {np.mean(thread_similarity[np.logical_not(np.isnan(thread_similarity))])}"
    log.warning(res)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', action='store', dest='emb', type=str, required=True,help='embedding model: minilm; roberta;tfidflsa')
    parser.add_argument('-c', action='store', dest='n_clusters', type=int, default=None, help='number of clusters')
    parser.add_argument('-t', action='store', dest='distance_threshold', type=float, default=0.8, help='distance threshold')

    parser.add_argument('--td', dest='td', action='store_true', help='time decay')
    parser.set_defaults(td=False)
    parser.add_argument('--ent', dest='ent', action='store_true', help='entity similarity')
    parser.set_defaults(ent=False)

    args = parser.parse_args()

    log = logging.getLogger(__name__)
    log.warning(args)

    run_seqint(emb=args.emb,n_clusters=args.n_clusters,distance_threshold=args.distance_threshold,time_decay=args.td, ent_similarity=args.ent)