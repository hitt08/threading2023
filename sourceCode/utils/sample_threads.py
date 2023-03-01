import os
from tqdm import tqdm
import numpy as np
from collections import Counter
from scipy.special import softmax
from utils.data_utils import read_dict_dump,read,write_json_dump,read_json_dump,write_dict_dump
import scipy
import pandas as pd
import logging
import datetime
from IPython.display import display, HTML as html_print
import pickle

log = logging.getLogger(__name__)


def merge_threads(d_id,d_s):
    global traversed,docs,thread_docs
    res_s=set(d_s)
    res_d=set([d_id])
    traversed.add(d_id)

    for s in d_s:
        for d in thread_docs[s]:
            if d not in docs:
                continue
            if d not in traversed:
                rs,rd=merge_threads(d,set(docs[d]).difference(res_s))
                res_s=res_s.union(rs)
                res_d=res_d.union(rd)
    return res_s,res_d

docs={}
traversed=set()
thread_docs=[]
def read_thread_file(url,check_duplicates=False):
    global traversed,docs,thread_docs
    with open(url, "rb") as f:
        threads=pickle.load(f)

    if check_duplicates:
        thread_docs=[[] for i in range(len(threads))]
        for idx,t in enumerate(threads):
            for d,v in t:
                thread_docs[idx].append(d)

        docs={}
        for idx,thread in enumerate(threads):
            for d,v in thread:
                if d not in docs:
                    docs[d]=set()
                docs[d].add(idx)

        new_threads_ids=[]
        new_thread_docs=[]
        traversed=set()
        for d in tqdm(docs.keys()):
            if d not in traversed:
                rs,rd=merge_threads(d,docs[d])
                if len(rd)>1:# and len(rd)<=10:
                    new_threads_ids.append(rs)
                    new_thread_docs.append(rd)

        new_threads=[]
        for t in new_threads_ids:
            temp=[]
            for idx in t:
                temp.extend(threads[idx])
            new_threads.append(temp)

        threads=new_threads
    return threads



def cstr(string,color="black"):
    return f"<text style='color:white;background:{color};font-weight:bold'>{string}</text>"

ques_color={'who':'#f44336', 'what':'#1565c0', 'when':'#d81b60', 'where':'#00695c', 'why':'#4a148c', 'how':'#880e4f'}

def print_thread(threads,idx,paragraphDF,data_5w1h,show_answers=True,outside_text=False):
    thread=threads[idx]
    print(f"Thread length: {len(thread)}")
    print(f"Period: {thread[0][1]} to {thread[-1][1]}\n")
    for i,(pid,date) in enumerate(thread):
        print(f"{i+1}) {pid} {date}")
        passage=paragraphDF.at[pid,"text"]
        if show_answers:
            legend_str="Legends: "
            for q,c in ques_color.items():
                legend_str+="\t"+cstr(q.title(),c)
            answers={}
            for q in ['how','what','why','who', 'when', 'where']:
                if data_5w1h[pid][q].strip():
                    temp_answer=cstr(data_5w1h[pid][q],ques_color[q])
                    if outside_text:
                        answers[q]=temp_answer
                    else:
                        passage=passage.replace(data_5w1h[pid][q],temp_answer)
            if outside_text:
                print(passage)
                answer_string=""
                for q in ques_color.keys():
                    if q in answers:
                        answer_string+="\t"+answers[q]
                display (html_print(answer_string))
            else:
                display (html_print(passage))
        else:
            print(passage)
        print("\n\n")
    if show_answers:
        display (html_print(legend_str))



def filter_threads(thread_df,min_length=0,max_length=None, min_days=0,max_days=None):
    len_cond=thread_df["length"]>=min_length
    if max_length:
        len_cond=np.logical_and(len_cond,thread_df["length"]<=max_length)

    date_cond=thread_df["period"]>=datetime.timedelta(min_days)
    if max_days:
        date_cond=np.logical_and(date_cond,thread_df["period"]<=datetime.timedelta(max_days))

    return thread_df[np.logical_and(len_cond,date_cond)]

def thread_cluster_keywords(corpus,feature_names,matrix,labels,top_n=3,max_df=0.9,filter_ids=None,print_text=False,sep="\t"):
    if filter_ids is not None:
        pred_labels=labels[filter_ids]
        train_matrix=matrix[filter_ids]
    else:
        pred_labels=labels
        train_matrix=matrix

    res={}
    features_set=set(feature_names)

    word2id=dict([(w,i) for i,w in enumerate(feature_names)])

    for idx in tqdm(set(pred_labels.tolist())):
        # t0=time.time()
        cond=pred_labels==idx
        if not np.any(cond):
            continue

        cluster=[corpus[i] for i in np.argwhere(labels==idx).squeeze(-1)]

        # t0=time.time()
        if print_text:
            print(f"Cluster {idx}",end=" ")
        vocab=[]
        for p in cluster:
            vocab.extend([t for t in set(p)])

        # t0=time.time()
        df=list(Counter(vocab).items())
        dfs=np.zeros(len(feature_names))
        for w,s in df:
            if w in features_set:
                widx=word2id[w]
                dfs[widx]=s/len(cluster)
                if dfs[widx]>=max_df:
                    dfs[widx]=0

        # t0=time.time()
        weight=np.asarray(np.abs(np.mean(train_matrix[pred_labels==idx],axis=0)))[0]

        # t0=time.time()
        tmpWeight=np.log(softmax(dfs,axis=-1))+np.log(softmax(weight,axis=-1))
        tempIdx=np.argsort(tmpWeight)[::-1][:top_n]
        wl=np.asarray(feature_names)[tempIdx].tolist()

        res[idx]=list(zip(wl,tmpWeight[tempIdx]))
        if print_text:
            for w in wl:
                print(repr(w),end=sep)
            print()
    return res

def get_keyword_df(thread_keywords,dfOutFile,key2threadOutFile):
    keyword_dict={}
    keyword2thread={}
    for t,wl in thread_keywords.items():
        for w,s in wl:
            if w not in keyword_dict:
                keyword_dict[w]={"threads":0,"mean_weight":0}
                keyword2thread[w]=[]
            keyword_dict[w]["threads"]+=1
            keyword_dict[w]["mean_weight"]+=s
            keyword2thread[w].append(t)
    keyword_df=pd.DataFrame(keyword_dict.values())
    keyword_df["word"]=keyword_dict.keys()
    keyword_df["mean_weight"]=keyword_df["mean_weight"]/keyword_df["threads"]
    keyword_df.set_index("word",inplace=True)
    keyword_df=keyword_df.sort_values(["threads","mean_weight"],ascending=[True,False])

    # outFile=f"{outPath}/{ofile}_keywords.json.gz"
    write_json_dump(dfOutFile,keyword_df.reset_index().to_dict(orient="records"),compress=True)
    log.warning(f"Keywords weights saved to : {dfOutFile}")
    write_dict_dump(key2threadOutFile,keyword2thread,compress=True)
    log.warning(f"Keywords2Thread dictionary saved to : {key2threadOutFile}")

    return keyword_df,keyword2thread

def get_thread_df(threads,thread_keywords,outFile):
    thread_dict=[]
    for idx,thread in enumerate(threads):
        temp={}
        temp["length"]=len(thread)
        temp["period"]=thread[-1][1]-thread[0][1]

        temp["keywords"]=[]
        temp["keyword_weight"]=0

        for w,s in thread_keywords[idx]:
            temp["keywords"].append(w)
            temp["keyword_weight"]+=s
        temp["keyword_weight"]/=3
        thread_dict.append(temp)
    thread_df=pd.DataFrame(thread_dict).sort_values("keyword_weight",ascending=False)

    # outFile=f"{outPath}/{ofile}_threads.json.gz"
    thread_df["period"]=thread_df["period"].apply(lambda x:str(x))
    write_json_dump(outFile,thread_df.reset_index().to_dict(orient="records"),compress=True)
    thread_df["period"]=pd.to_timedelta(thread_df["period"])
    log.warning(f"Thread weights saved to : {outFile}")

    return thread_df

def get_threads_keywords(config,ofile,threads,top_n=3,force=False):
    keywordFile = os.path.join(config.thread_sample_dir, f"{ofile}_keywords.json.gz")
    keyword2threadFile = os.path.join(config.thread_sample_dir, f"{ofile}_keyword2thread.json.gz")
    threadFile = os.path.join(config.thread_sample_dir, f"{ofile}_threads.json.gz")

    if not force and os.path.exists(keywordFile) and os.path.exists(keyword2threadFile) and os.path.exists(threadFile):
        log.warning(f"Threads Keywords Founds at:\n\t{keywordFile}\n\t{keyword2threadFile}\n\t{threadFile}")
        keyword_df=pd.DataFrame(read_json_dump(keywordFile,compress=True)).set_index("word")
        thread_df=pd.DataFrame(read_json_dump(threadFile,compress=True)).set_index("index")
        thread_df["period"]=pd.to_timedelta(thread_df["period"])
        keyword2thread=read_dict_dump(keyword2threadFile)

    else:
        data_5w1h=read_dict_dump(config.data5w1h_dict_file, compress=True)
        doc_5w1h_data=read_dict_dump(os.path.join(config.data5w1h_dir,"tk_5w1h.json.gz"), compress=True)
        features_5w1h=read(os.path.join(config.emb_dir, "tfidf_5w1h_features.txt.gz"),compress=True)
        tfidf_5w1h_matrix=scipy.sparse.load_npz(os.path.join(config.emb_dir, "tfidf_5w1h_emb.npz"))

        pred_label_dict={}
        for i,t in enumerate(threads):
            for d,_ in t:
                pred_label_dict[d]=i

        thread_pred=[]
        passage_filter_ids=[]
        for i,x in enumerate(data_5w1h.keys()):
            if x in pred_label_dict:
                thread_pred.append(pred_label_dict[x])
                passage_filter_ids.append(i)
            else:
                thread_pred.append(-1)
        thread_pred=np.asarray(thread_pred)

        thread_keywords=thread_cluster_keywords(list(doc_5w1h_data.values()),
                                                 features_5w1h,
                                                 tfidf_5w1h_matrix,
                                                 thread_pred,
                                                 top_n=top_n,
                                                 filter_ids=passage_filter_ids,print_text=False)

        keyword_df,keyword2thread = get_keyword_df(thread_keywords,keywordFile,keyword2threadFile)
        thread_df = get_thread_df(threads,thread_keywords,threadFile)

    return keyword_df,thread_df,keyword2thread