import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from data.data_processing import read_parts
from utils.data_utils import read_json_dump,read_parts,load_config
import os
import logging
import pickle
from collections import Counter

log = logging.getLogger(__name__)


def thread_score(G,thread):
    prev=None
    thread_sim=[]
    for i in thread:
        if prev is None:
            prev=i
            continue
        s=nx.path_weight(G,(prev,i),weight="weight")
        thread_sim.append(s)
        prev=i
    return thread_sim

def get_sddp_threads(DG,threads,st_idx,date_data, passage_doc_ids):

    threads = list(set([tuple(i) for i in threads]))
    thread_similarity = [np.mean(thread_score(DG, thread)) if len(thread) > 1 else np.nan for thread in threads]

    threads = threads + st_idx

    for thread in threads:
        temp=[]
        for pid in thread:
            did=passage_doc_ids[pid]
            temp.append((did,date_data[did]))
        threads.append(temp)

    return threads, np.array(thread_similarity)


#Eval Threads
def eval_ksdpp(sourcePath, paragraphDF):#passage_doc_ids, sens_labels):
    dpp_path = f"{sourcePath}/dpp"
    sens_label = []
    thread_label = []
    thread_docs = []
    thread_sens_label = {}

    log.warning(f'{dpp_path}/threads_dpp_ids.p')
    with open(f'{dpp_path}/threads_dpp_ids.p', "rb") as f:
        threads=pickle.load(f)

    for i ,t in enumerate(threads):
        thread_sens_label[i] = []
        for d in t:
            thread_label.append(i)
            thread_docs.append(d)

            sens = int(paragraphDF.iloc[d]["label"])
            thread_sens_label[i].append(sens)
            sens_label.append(sens)

    thread_docs=list(set(thread_docs))
    thread_docs_sens_labels=[]
    for i in thread_docs:
        thread_docs_sens_labels.append(int(paragraphDF.iloc[i]["label"]))

    all_sens_threads=0
    sens_threads_bins={0:[], 0.25:[], 0.5:[], 0.75:[]}
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


    res = f"\n\nTotal Docs: {len(paragraphDF)}, Total Sens Docs: {sum(paragraphDF['label'].values)}"
    res += f"\nThread Docs: {len(thread_docs)}, Thread Sens Docs: {sum(thread_docs_sens_labels)}"

    res += f"\nPred Threads: {len(set(thread_label))}"
    res += f"\nMean Thread Len: {np.mean(list(Counter(thread_label).values()))}"
    res += f"\nAll Sens Threads: {all_sens_threads}"

    res += f"\nThread Sens Ratios"
    for l,ratios in sens_threads_bins.items():
        r=l+0.25
        c = len(ratios)
        m = np.mean(ratios)
        s = np.std(ratios)
        res += f"\n\t {l}-{r}\t: Count={c}\tMean={m}\tStd={s}"


    log.warning(f"Result saved to: {dpp_path}/threads_dpp_eval.txt")
    with open(f"{dpp_path}/threads_dpp_eval.txt","a") as f:
        f.write(res)
