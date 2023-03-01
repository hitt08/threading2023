import os
from multiprocessing import Pool,Manager, cpu_count
from multiprocessing.managers import BaseManager
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import numpy as np
import scipy
from tqdm import tqdm
import argparse
import logging
from scipy.io import savemat,loadmat

from utils.data_utils import read_parts,load_config
from utils.vectorise import reduce_dim


log = logging.getLogger(__name__)

# COS Similarity
def calc_cos(args):
    idx,x=args
    res = cosine_similarity(x, tfidf_train_matrix)
    np.savez_compressed(f"{outdir}/{idx}.npz", res)
    pbar.update(x.shape[0])

pbar=None
outdir=""
tfidf_train_matrix=None
def compute_passage_cosine(config=None,batch_size=100):
    global pbar,outdir,tfidf_train_matrix
    if config is None:
        config = load_config()

    st = 0
    en = batch_size
    pool_iter = []
    b = 0

    passage_tfidf_emb = scipy.sparse.load_npz(config.passage_tfidf_emb_file)
    psg_parts = read_parts(config.passage_parts_file)
    for p_id, (p_st, p_en) in psg_parts.items():
        log.warning(f"====Part#{p_id + 1}====")
        tfidf_train_matrix = passage_tfidf_emb[p_st:p_en]
        outdir = os.path.join(config.dpp_dir,f"cos_weights_{p_id}")

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        while st < tfidf_train_matrix.shape[0]:
            pool_iter.append((b, tfidf_train_matrix[st:en]))
            st = en
            en = st + batch_size
            b += 1

        BaseManager.register("pbar", tqdm)
        bmanager = BaseManager()
        bmanager.start()
        pbar = bmanager.pbar(total=tfidf_train_matrix.shape[0])

        with Pool(processes=int(cpu_count()/2)) as pool:
            pool.map(func=calc_cos, iterable=pool_iter)

        pbar.close()


#Graph Creation
def create_graphs(config=None,weight_threshold=0.2,cos_weight_batch_size=100):
    if config is None:
        config = load_config()

    passage_tfidf_emb = scipy.sparse.load_npz(config.passage_tfidf_emb_file)
    psg_parts = read_parts(config.passage_parts_file)
    for p_id, (p_st, p_en) in psg_parts.items():
        log.warning(f"====Part#{p_id + 1}====")
        tfidf_train_matrix = passage_tfidf_emb[p_st:p_en]
        cosdir = os.path.join(config.dpp_dir, f"cos_weights_{p_id}")
        part_id=f"_{p_id}"

        batch_size=cos_weight_batch_size
        st=0
        en=batch_size
        pool_iter=[]
        b=0
        while st<tfidf_train_matrix.shape[0]:
            pool_iter.append((b,tfidf_train_matrix[st:en]))
            st = en
            en = st + batch_size
            b+=1

        doc_features=None
        DG = nx.DiGraph()
        node_id=0 #Main Counter
        iterm_weights=None
        with tqdm(total=len(pool_iter)) as pbar:
            for idx,_ in pool_iter:
                with np.load(f"{cosdir}/{idx}.npz") as data:
                    iterm_weights=data["arr_0"]
                for weights in iterm_weights:
                    edges=[(node_id, node_id+j, w) for j,w in enumerate(weights[node_id:]) if w>weight_threshold]
                    DG.add_weighted_edges_from(edges)
                    node_id+=1

                if DG.number_of_nodes()<tfidf_train_matrix.shape[0]:
                    DG.add_nodes_from(set(range(tfidf_train_matrix.shape[0])).difference(DG.nodes))

                temp=iterm_weights
                a=np.argsort(temp).squeeze()[-1000:]
                mask = np.ones_like(temp,dtype=bool)
                for i in range(temp.shape[0]):
                    mask[i,a[i,:]] = False
                    temp[i,a[i,:]]=np.where(temp[i,a[i,:]],1,0)
                temp[mask]=0
                temp=scipy.sparse.csr_matrix(temp)
                if doc_features is None:
                    doc_features=temp.copy()
                else:
                    doc_features=scipy.sparse.vstack((doc_features,temp))
                pbar.update()
                if idx%100==0:
                    pbar.set_description(f"N:{DG.number_of_nodes()}, E:{DG.number_of_edges()}")
                    log.warning(f"N:{DG.number_of_nodes()}, E:{DG.number_of_edges()}")
            pbar.set_description(f"N:{DG.number_of_nodes()}, E:{DG.number_of_edges()}")

        if DG.number_of_edges()<1:
            log.warning(f"\tSKIPPING: number of edges should be greater than 0.")
            continue

        nx.write_gpickle(DG,os.path.join(config.dpp_dir,f"graph{part_id}.gp"))
        scipy.sparse.save_npz(os.path.join(config.dpp_dir,f"doc_features{part_id}.npz"),doc_features)

        dpp_doc_lsa,dpp_lsa_model=reduce_dim(doc_features,dim=50)
        np.savez_compressed(os.path.join(config.dpp_dir,f"doc_features_lsa{part_id}.npz"),dpp_doc_lsa)

        if DG.number_of_nodes():
            node_degree=nx.degree(DG,weight="weight")
            node_degree=np.array(sorted(dict(node_degree).items(),key=lambda x: x[0]))[:,1]
            adj=nx.adjacency_matrix(DG,nodelist=sorted(list(DG.nodes)),weight="weight")
        else: #Empty Files
            node_degree=np.array([])
            adj=scipy.sparse.csr_matrix(np.array([]))
        np.savez_compressed(os.path.join(config.dpp_dir,f"node_weights_lexrank{part_id}.npz"),node_degree)
        scipy.sparse.save_npz(os.path.join(config.dpp_dir,f"graph_adj{part_id}.npz"),adj)


def convert_to_mat(config=None):
    if config is None:
        config = load_config()

    dpp_mat_path = os.path.join(config.dpp_dir,"mat")
    if not os.path.exists(dpp_mat_path):
        os.mkdir(dpp_mat_path)


    psg_parts = read_parts(config.passage_parts_file)
    for p_id, (p_st, p_en) in psg_parts.items():
        log.warning(f"====Part#{p_id + 1}====")
        part_id=f"_{p_id}"

        gfile=os.path.join(config.dpp_dir, f"graph{part_id}.gp")
        if not os.path.exists(gfile):
            log.warning(f"\tSKIPPING: Graph not found: {gfile}.")
            continue

        #Q1
        with np.load(os.path.join(config.dpp_dir,f"node_weights_lexrank{part_id}.npz")) as data:
            temp=data["arr_0"]
            savemat(os.path.join(dpp_mat_path,f'model_Q1{part_id}.mat'),{'Q1':temp[None,:]})
            log.warning(f"Q1={temp.shape}")

        #G
        with np.load(os.path.join(config.dpp_dir,f"doc_features_lsa{part_id}.npz")) as data:
            temp=data["arr_0"]
            temp=np.hstack((temp,np.full(temp.shape[0],0.5)[:,None]))
            savemat(os.path.join(dpp_mat_path,f'model_G{part_id}.mat'),{'G':temp})
            log.warning(f"G={temp.shape}")

        # A
        temp=scipy.sparse.load_npz(os.path.join(config.dpp_dir,f"graph_adj{part_id}.npz"))
        temp.setdiag(0, k=0)
        savemat(os.path.join(dpp_mat_path,f'model_A{part_id}.mat'),{'A':temp})
        log.warning(f"A={temp.shape}")

if __name__ == "__main__":
    # global pnct_psg_tfidf_train_matrix,outdir
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='sourcePath', type=str,default=".", help='input file path')
    parser.add_argument('-b', action='store', dest='batch', type=int,default=100, help='batch size')
    parser.add_argument('-p', action='store', dest='part', type=int,default=None, help='part id')
    parser.add_argument('-b', action='store', dest='batch', type=int,default=100, help='batch size')
    parser.add_argument('-t', action='store', dest='threshold', type=float,default=0.1, help='edge threshold')
    args = parser.parse_args()
    log.warning(args)

    compute_passage_cosine(args.sourcePath,args.batch)
    create_graphs(args.sourcePath, weight_threshold=args.threshold, cos_weight_batch_size=args.batch)
    convert_to_mat(args.sourcePath)





#Mat Files