import logging
import os
import traceback
from multiprocessing import Pool,Manager,cpu_count
from multiprocessing.managers import BaseManager
from tqdm import tqdm
import math
from requests.exceptions import ReadTimeout

import spacy
import neuralcoref
from stanza.server import CoreNLPClient, StartServer
from stanza.server.client import AnnotationException
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor
from Giveme5W1H.extractor.tools.timex import Timex
from Giveme5W1H.extractor.document import Document

from extract_5w1h.beneparcomponent import BeneparComponent
from extract_5w1h.preprocessor import PreprocessorSpacy,remove_unserializable_results
from extract_5w1h.extractor import MasterExtractor
from utils.data_utils import read_json_dump,read,write,write_dict_dump,load_config



log=logging.getLogger(__name__)

QUESTIONS=["who","what","when","where","why","how"]
def extract_5w1h(doc):
    res_5w1h= {}
    for q in QUESTIONS:
        try:
            answer = doc.get_top_answer(q).get_parts_as_text()
            res_5w1h[q]=answer
        except IndexError:
            res_5w1h[q]=""
            pass

    return res_5w1h

ENT_TYPES ={"who" :set(["LOCATION" ,"ORGANIZATION" ,"MISC" ,"PERSON"]),
           "where" :set(["LOCATION"]),
           "when" :set(["DATE"])}
def parse_entities(doc_candidate,ent_type):
    if doc_candidate is None:
        return []

    res =[]
    cur_ent =""
    cur_ent_type =""
    cur_ent_idx =-1
    for parts in doc_candidate["parts"]:
        for part in parts:
            if type(part )!=dict:
                continue
            tk =part["nlpToken"]
            if tk["ner"] in ENT_TYPES[ent_type]:
                if tk["ner" ]==cur_ent_type and tk["index" ]==cur_ent_idx +1:  # Same Entity
                    if ent_type=="when":
                        continue
                    cur_ent+=tk["word" ] +tk["after"]
                else:
                    if cur_ent!="":
                        if type(cur_ent)==str:
                            cur_ent =cur_ent.strip()
                        res.append(cur_ent)
                    if ent_type=="when":
                        cur_ent =Timex.from_timex_text(tk["timex"]["value"]).get_start_date()
                    else:
                        cur_ent =tk["word" ] +tk["after"]
                    cur_ent_type =tk["ner"]
                cur_ent_idx =tk["index"]

    if cur_ent:  # Last Entity
        if type(cur_ent )==str:
            cur_ent =cur_ent.strip()
        res.append(cur_ent)

    return res


ENT_QUESTIONS=["who","where"]
def extract_entities(doc):
    res_entities={}
    for q in ENT_QUESTIONS:
        try:
            candidate = doc.get_top_answer(q)
            valid_entity=False
            for part_list in candidate.get_parts():
                for part in part_list:
                    if type(part)!=dict:
                        continue
                    if part["nlpToken"]["ner"]!="O":
                        valid_entity=True
                        break

            if valid_entity:
                res_entities[q]=candidate.get_json()
            else:
                res_entities[q]=None

            res_entities[q] = parse_entities(res_entities[q],q)
        except IndexError:
            pass

    return res_entities


def annotate_extract_spacy(passages,n_processes=1, batch_size=1):
    pbar_batch_desc=pbar.desc.replace(":","")
    docs={}
    try:
        for doc_dict in passages:
            docs[doc_dict["doc_id"]]=Document.from_text(doc_dict["text"], str(doc_dict["created"]))
    except Exception:
        traceback.print_exc()
        if(not SKIP_ERRORS):
            return

        # log.warning("Error. Skipping to next file")

    err_cnt = 0
    processed_ids = []
    data_5w1h = {}
    data_entities = {}
    annotated_docs = nlp.pipe([d.get_full_text() for d in docs.values()], n_process=n_processes, batch_size=batch_size)
    for (dId,raw_doc), annotation,doc_dict in zip(docs.items(), annotated_docs,passages):
        # errflag = False
        try:
            doc = preprocessor.preprocess(raw_doc, constituency(annotation))
            # dId=doc_dict["doc_id"]
            doc = extractor.parse(doc)
            data_5w1h[dId] = extract_5w1h(doc)
            data_entities[dId] = extract_entities(doc)
            processed_ids.append(dId)
        except Exception:
            # errflag = True
            err_cnt += 1
            if SHOW_ERRORS:
                traceback.print_exc()
            else:
                log.warning("Error. Skipping to next file")
            if (not SKIP_ERRORS):
                return
        pbar.set_description(f"{pbar_batch_desc}. Errors: {err_cnt}")
        pbar.update()
    return data_5w1h, data_entities, processed_ids


def annotate_corenlp(doc_dict):
    errorFlag=False
    try:
        words = doc_dict["text"].split(" ")
        if len(words) > 2000 or len(doc_dict["text"]) > 99980:  # 100000 max len core nlp
            raise AnnotationException

        doc = Document.from_text(doc_dict["text"], str(doc_dict["created"]))

    except AnnotationException:
        log.warning("Long Text Recorded. Skipping to next file")
        errorFlag=True
        if (not SKIP_ERRORS):
            return

    except TypeError:
        if SHOW_ERRORS:
            traceback.print_exc()
        else:
            log.warning("Error. Skipping to next file")
        errorFlag = True
        if (not SKIP_ERRORS):
            return

    if not errorFlag:
        try:
            preprocessor.preprocess(doc)
            dId = doc_dict["doc_id"]
            doc_5w1h[dId] = doc
        except (AnnotationException,ReadTimeout):
            if SHOW_ERRORS:
                traceback.print_exc()
            else:
                log.warning("Error. Skipping to next file")
            if (not SKIP_ERRORS):
                return

    pbar.update()

preprocessor=None
doc_5w1h={}
pbar = None
SKIP_ERRORS=True
SHOW_ERRORS=False
nlp=None
constituency=None
extractor=None
def run_5w1h_extract(config=None,n_processes=1,nlp_model="en_core_web_sm",use_gpu=True,force=False,threaded_extraction=True,skip_where=False,skip_errors=True,show_errors=True):
    global preprocessor
    global doc_5w1h
    global pbar
    global SKIP_ERRORS
    global SHOW_ERRORS
    global nlp
    global constituency
    global extractor

    if config is None:
        config = load_config()

    SKIP_ERRORS=skip_errors
    SHOW_ERRORS=show_errors
    PROCESSES = n_processes if n_processes <= cpu_count() else cpu_count()

    spacy_flag=True
    if nlp_model=="coreNLP":
        host = 'http://localhost:9000'
        THREADS = 6
        preprocessor = Preprocessor(host)
        preprocessor.cnlp = CoreNLPClient(endpoint=host, start_server=StartServer.DONT_START, threads=THREADS)
        spacy_flag=False
    else:
        preprocessor = PreprocessorSpacy()
        nlp = spacy.load(nlp_model)
        neuralcoref.add_to_pipe(nlp)
        nlp.add_pipe(remove_unserializable_results, last=True)
        constituency = BeneparComponent("benepar_en3", disable_tagger=True,use_gpu=use_gpu)

    temp_paragraphs = read_json_dump(config.passage_dict_file) #list of dict

    processedFile = os.path.join(config.data5w1h_dir,"5w1h_processed.txt") #processed passage ids

    processedParagraphs = set()
    if force:
        for path in [config.data5w1h_dict_file, processedFile]:
            if os.path.exists(path):
                os.remove(path)

        paragraphs=temp_paragraphs
    else: #Resume from previous batch
        if os.path.exists(processedFile):
            processedParagraphs = set(read(processedFile))
            log.warning(f"Processed paragraphs: {len(processedParagraphs)}")
        #Filter Processed passages
        paragraphs=[p for p in temp_paragraphs if p["doc_id"] not in processedParagraphs]


    nlp_engine="spaCy" if spacy_flag else "coreNLP"
    if PROCESSES>1:
        log.warning(f"Running {nlp_engine} {PROCESSES}-way parallel (Total CPUs: {cpu_count()})")
        if not spacy_flag:
            BaseManager.register("pbar", tqdm)
            bmanager = BaseManager()
            bmanager.start()
            pbar = bmanager.pbar(total=len(paragraphs))
            manager = Manager()
        else:
            pbar = tqdm(total=len(paragraphs))
    else:
        log.warning(f"Running {nlp_engine} Serially")
        pbar = tqdm(total=len(paragraphs))


    batch_size=1024
    st=0
    en=batch_size
    batch_cnt=1
    tot_batch=math.ceil(len(paragraphs)/batch_size)
    extractor = MasterExtractor(multithreaded=threaded_extraction,skip_where=skip_where)
    while st < len(paragraphs):
        pbar.set_description(f"Batch: {batch_cnt}/{tot_batch}")

        if spacy_flag:
            data_5w1h, data_entities, processed_ids = annotate_extract_spacy(paragraphs[st:en], PROCESSES, batch_size=128)
        else:
            data_5w1h = {}
            data_entities = {}
            if PROCESSES > 1:
                doc_5w1h = manager.dict()
            else:
                doc_5w1h = {}

            if PROCESSES>1:
                with Pool(processes=PROCESSES) as pool:
                    pool.map(func=annotate_corenlp, iterable=paragraphs[st:en])
            else:
                for i in paragraphs[st:en]:
                    annotate_corenlp(i)

            processed_ids=[]
            c=0
            for dId,doc in doc_5w1h.items():
                errflag=False
                err_cnt=0
                try:
                    doc = extractor.parse(doc)
                except Exception:
                    if(SKIP_ERRORS):
                        err_cnt += 1
                        errflag = True
                        if SHOW_ERRORS:
                            traceback.print_exc()
                        else:
                            log.warning("Error. Skipping to next file")
                    else:
                        return

                if not errflag:
                    data_5w1h[dId] = extract_5w1h(doc)
                    data_entities[dId] = extract_entities(doc)
                    processed_ids.append(dId)
                c+=1
                pbar.set_description(f"Batch: {batch_cnt}/{tot_batch}. Extracted: {c}/{len(doc_5w1h)}")


        st = en
        en = st + batch_size
        batch_cnt += 1

        write_dict_dump(config.data5w1h_dict_file, data_5w1h, mode="at",compress=True)
        write_dict_dump(config.entity_dict_file, data_entities, mode="at", compress=True)
        write(processedFile, processed_ids, mode="a")

    pbar.close()