import logging
import scipy
import os
import numpy as np
import pandas as pd

from utils.data_utils import read_json_dump,write,read, load_config
from utils.vectorise import tfidf_vectorize,sbert_vectorise


def vectorize_docs(config=None, lsa_dim=200, vect_tfidf=True,vect_minilm=True,vect_roberta=False,sbert_batch=10000, use_gpu=True):
    log = logging.getLogger(__name__)
    if config is None:
        config = load_config()

    paragraphDF = pd.DataFrame(read_json_dump(config.passage_dict_file))[["doc_id", "text"]].set_index("doc_id")
    #Ensure order
    paragraphDF = paragraphDF.loc[read(config.passage_docids_file,compress=True)]

    if vect_tfidf:
        # TF IDF Vectorise
        log.warning("TFIDF Vectorise")
        features, train_matrix, train_lsa_matrix = tfidf_vectorize(paragraphDF["text"].values, max_df=0.9, nltk_dir=config.nltk_dir, lsa=True, lsa_dim=lsa_dim)
        scipy.sparse.save_npz(config.passage_tfidf_emb_file, train_matrix)
        log.warning(f"\tTFIDF Vectors Saved at: {config.passage_tfidf_emb_file}")
        ofile = os.path.join(config.emb_dir, "psg_tfidflsa_emb.npz")
        np.savez_compressed(ofile, train_lsa_matrix)
        log.warning(f"\tTFIDF-LSA Vectors Saved at: {ofile}")
        write(config.passage_tfidf_features_file, features.get_feature_names(),mode="wt",compress=True)
        log.warning(f"\tTFIDF Features Saved at: {config.passage_tfidf_features_file}")

    if vect_minilm:
        # MiniLM Vectorise
        log.warning("MiniLM Vectorise")
        train_matrix = sbert_vectorise(os.path.join(config.transformer_dir, "all-MiniLM-L6-v2"), paragraphDF["text"].values, batch=sbert_batch, use_gpu=use_gpu)
        ofile = os.path.join(config.emb_dir, "psg_minilm_emb.npz")
        np.savez_compressed(ofile, train_matrix)
        log.warning(f"\tMiniLM Vectors Saved at: {ofile}")

    if vect_roberta:
        # DistilRoberta Vectorise
        log.warning("Roberta Vectorise")
        train_matrix = sbert_vectorise(os.path.join(config.transformer_dir, "all-distilroberta-v1"), paragraphDF["text"].values, batch=sbert_batch, use_gpu=use_gpu)
        ofile = os.path.join(config.emb_dir, "psg_roberta_emb.npz")
        np.savez_compressed(ofile, train_matrix)
        log.warning(f"\tRoberta Vectors Saved at: {ofile}")