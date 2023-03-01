import numpy as np
import scipy
import os
from tqdm import tqdm
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.data_utils import write_dict,read_dict,read_dict_dump,write, read,read_parts, load_config, write_dict_dump
from utils.vectorise import sbert_vectorise, reduce_dim
from utils.tokenize import Tokenizers

#TFIDF Vectorize
def tfidf_vectorize(doc_ids, doc_data, max_df=0.95, nltk_dir="/app/nltk_data", lsa=True, lsa_dim=200):
    tokenizers = Tokenizers(nltk_dir=nltk_dir)
    tk_collection = dict([(k, tokenizers.tokenize_pnct(v)) for (k, v) in tqdm(zip(doc_ids,doc_data),total=len(doc_ids))])
    # write_dict_dump(outPath.replace("_emb.npz","_tk.json.gz"), tk_5w1h_pnct_collection,compress=True)

    doc_tk_data=tokenizers.merge_doc_tokens(list(tk_collection.values()))

    vect = TfidfVectorizer(tokenizer=tokenizers.tokenize_split, strip_accents='ascii', max_df=max_df, max_features=100000,token_pattern=None)
    features = vect.fit(doc_tk_data)
    train_matrix = features.transform(doc_tk_data)

    # scipy.sparse.save_npz(outPath, train_matrix)
    # write(outPath.replace("_emb.npz","_features.txt"),features.get_feature_names())


    if lsa:
        train_lsa, lsa_model = reduce_dim(train_matrix, dim=lsa_dim)
        return tk_collection, features, train_matrix, train_lsa
    else:
        return tk_collection, features, train_matrix



def vectorise_5w1h(config, lsa_dim=200, vect_tfidf=True,vect_minilm=True,vect_roberta=False,sbert_batch=10000,use_gpu=True):
    log = logging.getLogger(__name__)
    if config is None:
        config = load_config()

    data_5w1h = read_dict_dump(config.data5w1h_dict_file, compress=True)


    log.warning("Creating data dictionaries")
    #Create data dictionaries
    doc_ids = read(config.passage_docids_file,compress=True)
    doc_5w1h_data = []
    doc_5w1h_docids = []
    for d in doc_ids:
        if d in data_5w1h:
            doc_5w1h_data.append(". ".join([i for i in data_5w1h[d].values() if i]))
            
            doc_5w1h_docids.append(d)

    write(config.data5w1h_docids_file,doc_5w1h_docids,mode="wt",compress=True)

    docid_set = set(doc_5w1h_docids)
    # Identifying Collection Splits, i.e., Parts
    doc_parts=read_parts(config.passage_parts_file)
    doc_5w1h_parts = {}
    st=0
    for pidx, (p_st, p_en) in tqdm(doc_parts.items()):
        part_docs=[d for d in doc_ids[p_st:p_en] if d in docid_set]
        en = st+len(part_docs)
        assert part_docs == doc_5w1h_docids[st:en]
        doc_5w1h_parts[pidx] = (st, en)
        st = en
    write_dict(config.data5w1h_parts_file, doc_5w1h_parts)

    dt_feat = pd.Series(read_dict(config.date_dict_file,compress=True)).loc[doc_5w1h_docids]
    dt_feat = pd.to_datetime(dt_feat).apply(lambda x: x.timestamp()).values
    np.savez_compressed(config.date_features_file, dt_feat)






    # 5W1H vectorize
    tfidfPath = os.path.join(config.emb_dir, "tfidf_5w1h_emb.npz")
    lsaPath = os.path.join(config.emb_dir, "tfidflsa_5w1h_emb.npz")
    minilmPath = os.path.join(config.emb_dir, "minilm_5w1h_emb.npz")
    robertaPath = os.path.join(config.emb_dir, "roberta_5w1h_emb.npz")


    if vect_tfidf:
        log.warning("TFIDF Vectorise")
        tk_collection, features, tfidf_5w1h_matrix, lsa_5w1h_matrix = tfidf_vectorize(doc_5w1h_docids,doc_5w1h_data, max_df=0.95, nltk_dir=config.nltk_dir, lsa=True, lsa_dim=lsa_dim)

        scipy.sparse.save_npz(tfidfPath, tfidf_5w1h_matrix)
        log.warning(f"\tTFIDF Vectors Saved at: {tfidfPath}")
        np.savez_compressed(lsaPath, lsa_5w1h_matrix)
        log.warning(f"\tTFIDF-LSA Vectors Saved at: {lsaPath}")
        ofile = tfidfPath.replace("_emb.npz", "_features.txt.gz")
        write(ofile, features.get_feature_names(),mode="wt",compress=True)
        log.warning(f"\tTFIDF Features Saved at: {ofile}")
        ofile = os.path.join(config.data5w1h_dir,"tk_5w1h.json.gz")
        write_dict_dump(ofile, tk_collection, compress=True)
        log.warning(f"\tTokenised Collection Saved at: {ofile}")

    if vect_minilm:
        log.warning("MiniLM Vectorise")
        minilm_5w1h_matrix = sbert_vectorise(os.path.join(config.transformer_dir, "all-MiniLM-L6-v2"), doc_5w1h_data, batch=sbert_batch, use_gpu=use_gpu)
        np.savez_compressed(minilmPath, minilm_5w1h_matrix)
        log.warning(f"\tMiniLM Vectors Saved at: {minilmPath}")

    if vect_roberta:
        log.warning("Roberta Vectorise")
        minilm_5w1h_matrix = sbert_vectorise(os.path.join(config.transformer_dir, "all-distilroberta-v1"), doc_5w1h_data,batch=sbert_batch, use_gpu=use_gpu)
        np.savez_compressed(robertaPath, minilm_5w1h_matrix)
        log.warning(f"\tRoberta Vectors Saved at: {robertaPath}")
