import networkx as nx
from tqdm import tqdm
import pickle
import numpy as np
from scipy.io import savemat,loadmat
import matlab.engine
from matlab.engine import MatlabExecutionError
import argparse
import os
import pandas as pd
import logging
from utils.data_utils import read_parts, load_config, read, read_dict
from utils.thread_utils import eval_ev_threads

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

    tmp_threads = np.asarray(list(set([tuple(i) for i in threads])))
    thread_similarity = [np.mean(thread_score(DG, thread)) if len(thread) > 1 else np.nan for thread in tmp_threads]


    tmp_threads = tmp_threads + st_idx
    threads = []
    for thread in tmp_threads:
        temp=[]
        for pid in thread:
            did=passage_doc_ids[pid]
            temp.append((did,date_data[did]))
        threads.append(temp)

    return threads, np.array(thread_similarity)



#Run Mat
def run_ksdpp(config=None,epochs=100):
    if config is None:
        config = load_config()

    dpp_mat_path = os.path.join(config.dpp_dir,f"mat")
    psg_parts = read_parts(config.passage_parts_file)

    passage_doc_ids = read(config.passage_docids_file, compress=True)
    date_data = pd.to_datetime(pd.Series(read_dict(config.date_dict_file, compress=True))).to_dict()


    threads = []
    thread_similarity = []
    for p_id, (p_st, p_en) in psg_parts.items():
        log.warning(f"====Part#{p_id + 1}====")
        part_id=f"_{p_id}"
        gfile=os.path.join(config.dpp_dir,f"graph{part_id}.gp")
        if not os.path.exists(gfile):
            log.warning(f"\tSKIPPING: Graph not found: {gfile}.")
            continue

        Q1 = loadmat(os.path.join(dpp_mat_path, f"model_Q1{part_id}.mat"))["Q1"]
        G = loadmat(os.path.join(dpp_mat_path, f"model_G{part_id}.mat"))["G"]
        A = loadmat(os.path.join(dpp_mat_path, f"model_A{part_id}.mat"))["A"]
        savemat("model_Q1.mat", {'Q1': Q1})
        savemat("model_G.mat", {'G': G})
        savemat("model_A.mat", {'A': A})

        eng = matlab.engine.start_matlab()
        eng.startup(nargout=0)

        DG = nx.read_gpickle(gfile)

        part_threads = []
        err=0
        with tqdm(total=epochs) as pbar:
            for e in range(epochs):
                pbar.set_description(f"Total: {len(part_threads)}, Error:{err}")
                try:
                    eng.ksdpp_thread(nargout=0)
                except MatlabExecutionError:
                    err+=1
                    continue
                temp = loadmat("doc_thread_sdpp.mat")
                temp = temp["sdpp_sample"] - 1
                for i in range(temp.shape[1]):
                    if nx.is_path(DG, np.unique(temp[:, i])):
                        part_threads.append(temp[:, i])
                        pbar.set_description(f"Total: {len(part_threads)}, Error:{err}")
                if len(part_threads):
                    np.save(os.path.join(config.dpp_dir,f"all_threads{part_id}.npy"), np.stack(part_threads))
                pbar.update()
            pbar.set_description(f"Total: {len(part_threads)}, Error:{err}")
        if len(part_threads):
            part_threads = np.stack(part_threads)
            np.save(os.path.join(config.dpp_dir,f"all_threads{part_id}.npy"), part_threads)

            part_threads, part_thread_similarity = get_sddp_threads(DG, part_threads, p_st, date_data, passage_doc_ids)
            threads.extend(part_threads)
            thread_similarity.append(part_thread_similarity)

    thread_similarity = np.hstack(thread_similarity)

    ofile = os.path.join(config.threads_dir, f"threads_dpp.p")
    with open(ofile, "wb") as f:
        pickle.dump(threads, f)
    log.warning(f"Threads stored at: {ofile}")
    np.savez_compressed(os.path.join(config.threads_dir, f"thread_sim_dpp.npz"), thread_similarity)

    res = f"\nThread Count:{len(threads)}"
    res += f"\nAvg Length:{np.mean([len(i) for i in threads])}"
    res += f"\nCosine Score: {np.mean(thread_similarity[np.logical_not(np.isnan(thread_similarity))])}"
    log.warning(res)


def eval_sdpp_threads(config,file_pattern, min_len=2, print_res=True):
    log = logging.getLogger(__name__)

    with open(os.path.join(config.threads_dir,f"threads_{file_pattern}.p"),"rb") as f:
        threads = pickle.load(f)

    with np.load(os.path.join(config.threads_dir,f"thread_sim_{file_pattern}.npz")) as data:
        thread_similarity = data["arr_0"]

    res = eval_ev_threads(config, threads,min_len=min_len)
    res = f"\n\nMean Cosine Similarity (MPDCS): {np.mean(thread_similarity):.4f}" + res

    if print_res:
        print(res)

    ofile=f"{config.threads_dir}/eval_{file_pattern}.txt"
    with open(ofile,"w") as f:
        f.write(res)
    log.warning(f"\nResults Saved at: {config.threads_dir}/eval_{file_pattern}.txt")

    return res

if __name__ == "__main__":
    # global pnct_psg_tfidf_train_matrix,outdir
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', action='store', dest='epochs', type=int,default=100, help='epochs')
    args = parser.parse_args()

    run_ksdpp(epochs=args.epochs)



#Create Thread

#Eval Thread