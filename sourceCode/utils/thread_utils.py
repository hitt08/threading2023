from sklearn.metrics import homogeneity_score
from sklearn.metrics import  normalized_mutual_info_score as nmi
import time
import os
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cosine
import datetime
from collections import Counter
import logging
import networkx as nx

from utils.data_utils import read_json_dump,load_hyperparams

def create_entity_graph(entities_dict,doc_ids=None,verbose=False):
    log = logging.getLogger(__name__)
    if verbose:
        log.warning("Constructing Graph")

    G = nx.Graph()
    edges = []
    node_attr = []

    ent_docs=entities_dict.keys() if doc_ids is None else doc_ids

    if verbose:
        pbar=tqdm(total=len(ent_docs))
    for k in ent_docs:
        v=entities_dict[k]
        G.add_node(k, type="passage")
        for _, ents in v.items():
            for e in ents:
                edges.append((k, e, 1))
                node_attr.append((e, "entity"))

        if verbose:
            pbar.update()

    if verbose:
        pbar.close()

    G.add_weighted_edges_from(edges)
    nx.set_node_attributes(G, dict(node_attr), name="type")

    if verbose:
        log.warning(f"Entity Graph: Number of Nodes= {G.number_of_nodes()}. Number of Edges= {G.number_of_edges()}")

    return G

def delta_to_days(x):
    return x.days+((x.seconds/60)/60)/24

def tokenize_split(string,sep="~|~"):
    return string.split(sep)

def get_thread_file_pattern(method, emb, hyperparams=None, time_decay=True, ent_similarity=True,community_method="npc"):
    if hyperparams is None:
        hyperparams = load_hyperparams()

    pattern = f"{emb}_w{hyperparams.weight_threshold}"
    if time_decay:
        pattern += f"_td{hyperparams.alpha}"
    if ent_similarity:
        pattern += f"_ent{hyperparams.gamma}"
    if method=="seqint" and not time_decay and not ent_similarity:
        pattern += f"_ward"

    if method=="hint":
        pattern = f"{community_method}_{pattern}"

    return f"{method}_{pattern}"

def ev_thread_score(thread,passage_doc2id,points):
    prev=None
    thread_sim=[]
    for i,_ in thread:
        idx=passage_doc2id[i]
        # print(idx)
        # x=points[idx][None,:] if len(points[idx].shape)<2 else points[idx]
        x = points[idx]

        if prev is None:
            prev=x
            continue
        # print(prev.shape)
        # print(x)
        s=1-cosine(prev,x)
        thread_sim.append(s)
        prev=x
    return thread_sim


def eval_ev_threads(config, threads, min_len=0):
    paragraphDF = pd.DataFrame(read_json_dump(config.passage_dict_file))[["doc_id", "label"]].set_index("doc_id")

    thread_sens_label = {}
    sens_label = []

    thread_label = []
    thread_spans = []

    pred_label_dict = {}
    docs = set()


    for i, t in enumerate(threads):
        temp_pred = []
        temp_thread_sens_label = []
        temp_sens_label = []
        temp_docs = set()
        temp_label = {}
        for d, _ in t:
            if d not in docs:
                temp_pred.append(i)
                temp_docs.add(d)
                temp_label[d] = i

                sens = int(paragraphDF.at[d, "label"])
                temp_thread_sens_label.append(sens)
                temp_sens_label.append(sens)

        if len(temp_pred) >= min_len:
            thread_label.extend(temp_pred)

            thread_sens_label[i] = temp_thread_sens_label
            sens_label.extend(temp_sens_label)

            for d in temp_docs:
                docs.add(d)
                pred_label_dict[d] = temp_label[d]

            thread_spans.append(delta_to_days(t[-1][1] - t[0][1]))


    all_sens_threads=0
    sens_threads_bins={0.0:[], 0.25:[], 0.5:[], 0.75:[]}
    for k,v in thread_sens_label.items():
        if len(v)==sum(v):
            all_sens_threads+=1

        ratio=sum(v)/len(v)
        for l in sens_threads_bins.keys():
            r=l+0.25
            if r==1:
                r=1.1
            if ratio >= l and ratio < r:
                sens_threads_bins[l].append(ratio)


    # res="Selected:"
    # res=f"\n\nDocs: {len(thread_label)}, Pred Threads: {len(set(thread_label))}"


    res = f"\nTotal Docs: {len(paragraphDF)}, Total Sens Docs: {sum(paragraphDF['label'].values)}"
    res += f"\nThread Docs: {len(thread_label)}, Thread Sens Docs: {sum(sens_label)}"

    res += f"\nThread Count: {len(set(thread_label))}"
    res += f"\nMean Thread Len: {np.mean(list(Counter(thread_label).values())):.4f}"
    res += f"\nMean Thread days span: {np.mean(thread_spans):.4f}"
    res += f"\nAll Sens Threads: {all_sens_threads}"

    res += f"\nThread Sens Ratios"
    for l,ratios in sens_threads_bins.items():
        c = len(ratios)
        r = l + 0.25
        if c>0:
            m = np.mean(ratios)
            s = np.std(ratios)
        else:
            m, s = 0,0
        res += f"\n\t {l:.2f} - {r:.2f}\t: Count={c}\tMean={m:.4f}\tStd={s:.4f}"

    return res



def eval_filter_threads(config,file_pattern, params=None, print_res=True):
    log = logging.getLogger(__name__)

    with open(os.path.join(config.threads_dir,f"threads_{file_pattern}.p"),"rb") as f:
        threads = pickle.load(f)

    with np.load(os.path.join(config.threads_dir,f"thread_sim_{file_pattern}.npz")) as data:
        thread_similarity = data["arr_0"]

    if params is None:
        params= {'thread_len': [3, 100],
                 'similarity': [0.2, 0.8],
                 'days_num': [0, 2000]}
    log.warning(f"Filtering using the following parameters:\n{params}")

    l_range = params["thread_len"]
    s_range = params["similarity"]
    d_range = params["days_num"]

    new_threads, new_sim, new_span = [], [], []
    for i, s in zip(threads, thread_similarity):
        if l_range[0] <= len(i) <= l_range[1] \
                and datetime.timedelta(d_range[0]) <= (i[-1][1] - i[0][1]) <= datetime.timedelta(d_range[1]) \
                and s_range[0] <= s <= s_range[1]:

            new_threads.append(i)
            new_sim.append(s)


    res = eval_ev_threads(config, new_threads)
    res = f"\n\nMean Cosine Similarity (MPDCS): {np.mean(new_sim):.4f}" + res

    if print_res:
        print(res)

    ofile=f"{config.threads_dir}/eval_{file_pattern}.txt"
    with open(ofile,"w") as f:
        f.write(res)
    log.warning(f"\nResults Saved at: {config.threads_dir}/eval_{file_pattern}.txt")

    return res



def get_graph_threads(communities, date_data, passage_matrix, passage_doc2id,verbose=True):

    threads = []
    thread_similarity = []

    if verbose:
        pbar=tqdm(total=np.count_nonzero(communities))
    for com in communities:
        res = com
        thread = []
        for x in res:
            thread.append((x, date_data[x]))
        thread = np.asarray(thread)
        srt_ids = np.lexsort([thread[:, 0], thread[:, 1]])
        thread = thread[srt_ids].tolist()  # sorted(thread, key=lambda x: x[1])
        # thread_period = thread[-1][1] - thread[0][1]

        thread_sim = np.mean(ev_thread_score (thread, passage_doc2id, passage_matrix))  if len(thread) > 1 else np.nan

        threads.append(thread)
        thread_similarity.append(thread_sim)
        if verbose:
            pbar.update()
    if verbose:
        pbar.close()

    return threads, np.array(thread_similarity)
