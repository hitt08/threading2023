import gzip
import json
from types import SimpleNamespace as Namespace
import os
import logging

def load_hyperparams(url="hyperparameters.json",overrides={}):
    with open(url,"r") as f:
        res = json.load(f)

    for k,v in overrides.items():
        res[k]=v
    
    return Namespace(**res)

def load_config(url="config.json",overrides={}):
    log = logging.getLogger()

    with open(url,"r") as f:
        tmp = json.load(f)
    
    res = dict([(k, v) for k, v in tmp["parent"].items()])

    #Overrides Parents
    dependent_overrides={}
    for k,v in overrides.items():
        if type(v) == str:
            res[k]=v
        elif type(v) == dict:
            dependent_overrides[k]=v
        else:
            log.warning(f"Skipping Invalid Override| {k}: {v}")

    #Resolving dependent paths
    for k, v in list(tmp["dependents"].items()) + list(dependent_overrides.items()):
        if k in res and k not in overrides:
            log.warning(f"Skipping Duplicate Configuration Key| {k}: {v}")
            continue

        if v["parent"] in res:
            res[k] = os.path.join(res[v["parent"]], v["name"])
        else:
            log.warning(f"Skipping Invalid Configuration| {k}: {v}")

    #Creating Directories
    for k,v in res.items():
        if k.endswith("dir"):
            if not os.path.exists(v):
                log.warning(f"Creating Directory: {v}")
                os.mkdir(v)

    return Namespace(**res)

def read_json_dump(url,compress=True):
    data = []
    if compress:
        f = gzip.open(url, "rt")
    else:
        f = open(url, "r")
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    return data


def write_json_dump(url, data,mode="wt",compress=True):
    if compress:
        if not url.endswith(".gz"):
            url=url + ".gz"
        f = gzip.open(url, mode) #Write in write/append mode
    else:
        f = open(url, mode) #Write in write/append mode

    out_data = [json.dumps(d) for d in data]
    for d in out_data:
        # if compress:
        #     d = d.encode()
        f.write(d)
        f.write('\n')
    f.close()

def write_dict_dump(url, data,mode="wt",compress=True):
    if compress:
        if not url.endswith(".gz"):
            url=url + ".gz"
        f = gzip.open(url, mode) #Write in write/append mode
    else:
        f = open(url, mode) #Write in write/append mode
    out_data = json.dumps(data)
    # if compress:
    #     out_data=out_data.encode()
    f.write(out_data)
    f.write("\n")
    f.close()

def read_dict_dump(url,compress=True):
    res={}
    if compress:
        f = gzip.open(url, "rt")
    else:
        f = open(url, "r")
    for line in f.read().splitlines():
        res.update(json.loads(line))
    f.close()
    return res

def get_data_split(doc_df,tk_collection):
    doc_ids,data,data_tk,labels = [],[],[],[]
    for i,r in doc_df.iterrows():
        doc_ids.append(i)
        data.append(r["text"])
        data_tk.append(tk_collection[i])
        labels.append(r["label"])
    return {"ids": doc_ids, "data": data, "data_tk": data_tk, "labels": labels}

def write(url, data, mode="w", compress = False):
    if compress:
        if not url.endswith(".gz"):
            url=url + ".gz"
        f = gzip.open(url, mode)
    else:
        f = open(url, mode)

    if type(data) == str:
        data = [data]

    for l in data:
        f.write(l)
        f.write('\n')

    f.close()

def read(url, compress = False):
    res=[]
    if compress:
        f = gzip.open(url, "rt")
    else:
        f = open(url, "r")

    for line in f.read().splitlines():
        res.append(line)
    f.close()

    return res

def write_dict(url, data, sep="~|~", mode="w",compress=False):
    if compress:
        if not url.endswith(".gz"):
            url=url + ".gz"
        f = gzip.open(url, mode)
    else:
        f = open(url, mode)
    for k, v in data.items():
        f.write(f"{k}{sep}{v}")
        f.write('\n')
    f.close()

def read_dict(url, sep="~|~",compress=False):
    res = {}
    if compress:
        fl = gzip.open(url, "rt")
    else:
        fl = open(url, "r")

    for line in fl.read().splitlines():
        k, v = line.split(sep)
        res[k.strip()] = v.strip()
    fl.close()

    return res

def read_parts(sourcePath):
    if not os.path.exists(sourcePath):
        return {}

    temp = read_dict(sourcePath)
    parts = {}
    for k, v in temp.items():
        parts[int(k)] = tuple(map(int, v.strip("()").split(",")))
    return parts