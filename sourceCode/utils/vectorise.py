import logging
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

import torch
from sentence_transformers import SentenceTransformer

from utils.tokenize import Tokenizers


def reduce_dim(matrix, dim=300, lsa_model=None, random_seed=0):
    if (lsa_model is None):
        svd = TruncatedSVD(dim, random_state=random_seed)
        normalizer = Normalizer(copy=False)
        lsa_model = make_pipeline(svd, normalizer)

        X = lsa_model.fit_transform(matrix)

        return X, lsa_model
    else:
        return lsa_model.transform(matrix)

def tfidf_vectorize(doc_data, max_df=0.95, nltk_dir="/app/nltk_data", lsa=True, lsa_dim=200):
    tokenizers = Tokenizers(nltk_dir=nltk_dir)
    vect = TfidfVectorizer(tokenizer=tokenizers.tokenize_pnct_lemma, strip_accents='ascii', max_df=max_df, max_features=100000, token_pattern=None)
    features = vect.fit(doc_data)
    train_matrix = features.transform(doc_data)

    if lsa:
        train_lsa, lsa_model = reduce_dim(train_matrix, dim=lsa_dim)
        return features, train_matrix, train_lsa
    else:
        return features, train_matrix


def sbert_vectorise(model_name,doc_data,batch=10000,use_gpu=True):
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    sbert_model = SentenceTransformer(model_name,device=device)

    log = logging.getLogger(__name__)
    device_str="CPU" if "cpu" in device.type else "GPU"
    log.warning(f"\tRunning on {device_str}")

    sbert_model.eval()
    res=[]
    st=0
    en=batch

    with torch.no_grad():
        with tqdm(total=len(doc_data)) as pbar:
            while st<len(doc_data):
                d=doc_data[st:en]
                res.extend(sbert_model.encode(d))
                st=en
                en=st+batch
                pbar.update(len(d))
    res=np.stack(res)

    return res