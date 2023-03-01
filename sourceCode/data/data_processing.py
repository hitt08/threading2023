import os
import pandas as pd
import logging
import numpy as np
import pickle
import json

from data.dateparser import datetime_parsing
from data.docxparser import getCCD, getAnnotations
from data.ccdParser import parseDocument, parseParagraphs
from utils.data_utils import write_json_dump,write_dict,write,load_config

from tqdm import tqdm
from unidecode import unidecode

def clean_text(string):
    return unidecode(string).replace("\x7f","").replace("\n"," ")


class DataProcessor:
    def __init__(self, config=None, default_date=None, auto_extract_dates=False) -> None:
        if config is None:
            config = load_config()

        self.config = config
        self.log = logging.getLogger(__name__)
        self.default_date = pd.to_datetime('2000-01-01 00:00:00') if default_date is None else default_date  # Can be used in case there is no date field
        self.auto_extract_dates = auto_extract_dates

    # READ DOCS
    def read_docs(self, pathCCD, pathJSON):
        config = self.config

        files_to_process = []

        # getting a list of all #wc.docx files in subfolders of target folder
        for subdir, dirs, files in os.walk(config.source_dir):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith("#wc.docx") and not filepath.endswith("#wcmeta.docx"):  # ignore meta files
                    files_to_process = files_to_process + [filepath]

        # going through list and trying to create a ccd and json from each discovered file
        failed_files = []
        successful_files = []

        self.log.warning(f"Reading {len(files_to_process)} files")
        # get filenames and use docx parser to get ccd and json
        for filepath in tqdm(files_to_process):
            filename = os.path.basename(filepath)

            ccd = getCCD(filepath)
            jsn = getAnnotations(filepath)
            json_string = json.dumps(jsn)

            # TO DO: make ccdParser return 2 for json or ccd if either of those parsing processes fail for a doc

            # if parsing file processes successfully, save ccd and json
            if 2 not in [ccd, json]:
                with open(os.path.join(pathCCD, os.path.splitext(filename)[0] + ".xml"), 'w') as f:
                    f.write(ccd)
                    # f.close
                with open(os.path.join(pathJSON, os.path.splitext(filename)[0] + ".json"), 'w') as f:
                    f.write(json_string)
                successful_files = successful_files + [filepath]

            else:
                # i.e. if parsing failed add to failed files
                failed_files = failed_files + [filepath]

        return successful_files, failed_files

    def get_passages(self, successful_files, min_len=10, max_len=500):
        config = self.config
        paraTitles = []
        paraCollection = []
        paraLabels = []
        paraCreated = []

        totPassages = 0

        self.log.warning(f"Parsing passages from {len(successful_files)} files")
        # for filepaths to files that were succesfully parsed from ccd
        for filepath in tqdm(successful_files):
            # get filename from path
            fileN = filepath.rsplit('/', 1)[-1][0:-5]

            ####### CCD Document Parser #########
            # call paragraph parser which opens ccd and json on its own
            temp_paragarphs, temp_labelsP, doc_created = parseParagraphs(config.target_dir, fileN)
            ####################################

            # Assigning a dummy date or extract date from text in case the documents do not have create-datetime field
            if doc_created is None:
                candidates = []
                if self.auto_extract_dates:
                    for p in temp_paragarphs:
                        candidates, _ = datetime_parsing(p, self.default_date)
                        if len(candidates):  # Break after the first paragraph with dates
                            break
                doc_created = candidates[0] if len(candidates) else self.default_date  # Select the first date object

            # Assigning dummy labels in case the documents do not have sensitivity ground-truth
            if temp_labelsP is None:
                temp_labelsP = np.ones(len(temp_paragarphs), dtype=int).tolist()

            totPassages += len(temp_paragarphs)

            # Filter Passages based on number of words
            paragarphs, labelsP, titles = [], [], []
            para_id = 0
            for p, l in zip(temp_paragarphs, temp_labelsP):
                words = p.split()
                if len(words) >= min_len and len(words) <= max_len:
                    paragarphs.append(clean_text(p))
                    labelsP.append(l)
                    titles.append(fileN + "_" + str(para_id))
                para_id += 1

            # each paragraph that is returned needs to be linked back to its filename, so ([fileName] * (no_of_paras_returned))
            # tempPara = [fileN+"_"+str(i) for i in range(len(paragarphs))]

            tempCreated = [doc_created] * len(paragarphs)

            # add the titles, paragraphs and labels for each para to respective holding lists
            paraTitles = paraTitles + titles
            paraCollection = paraCollection + paragarphs
            paraLabels = paraLabels + labelsP
            paraCreated = paraCreated + tempCreated

        self.log.warning(
            f"Total Passages: {totPassages}. Filtered Passages (length between [{min_len},{max_len}]): {len(paraCollection)}")

        # once all docs have been parsed to paragraphs and stored in the above three lists, convert to dataframe
        # dataframes need to have [sourcefile, text, label]
        paragraphDF = pd.DataFrame(
            {'doc_id': paraTitles, 'text': paraCollection, 'label': paraLabels, 'created': paraCreated}).sort_values(
            ["created", "doc_id"])

        return paragraphDF

    def process(self, min_len=10, max_len=500, collection_split_size=40000, mockup=False):
        config = self.config

        #######CCD#########
        pathCCD = os.path.join(config.target_dir, "ccd")
        pathJSON = os.path.join(config.target_dir, "json")

        if not os.path.exists(pathCCD):
            os.mkdir(pathCCD)
        if not os.path.exists(pathJSON):
            os.mkdir(pathJSON)

        successful_files, failed_files = self.read_docs(pathCCD, pathJSON)
        #######CCD#########

        paragraphDF = self.get_passages(successful_files, min_len=min_len, max_len=max_len)

        paragraphDF["created"] = paragraphDF["created"].apply(lambda x: str(x) if x is not None else x)
        write(config.passage_docids_file, paragraphDF["doc_id"].values, mode="wt", compress=True)
        write_json_dump(config.passage_dict_file, paragraphDF.to_dict(orient="records"))
        write_dict(config.date_dict_file, paragraphDF.set_index("doc_id")["created"].to_dict(), mode="wt",
                   compress=True)
        paragraphDF["created"] = pd.to_datetime(paragraphDF["created"])

        if mockup:
            # START MOCKUP
            self.log.warning("\n::::MOCKING DATA::::\n")
            mock_passages = pickle.load(open(os.path.join(config.source_dir, "aug.p"), "rb"))
            paragraphDF = paragraphDF[:3]
            st_ind = np.max(paragraphDF.index.values) + 1
            doc_id = 0
            df_data = []
            for i in range(4):
                st = i * 300
                for p1, p2, p3 in zip(mock_passages[st:st + 100], mock_passages[st + 100:st + 200],
                                      mock_passages[st + 200:st + 300]):
                    y, m, d = np.random.randint(1995, 2000), np.random.randint(1, 12), np.random.randint(1, 28)
                    df_data.append(
                        [f"doc{doc_id}#wc_{0}", p1, np.random.choice([0, 1], 1)[0], "{}-{:02}-{:02}".format(y, m, d)])
                    df_data.append(
                        [f"doc{doc_id}#wc_{1}", p2, np.random.choice([0, 1], 1)[0], "{}-{:02}-{:02}".format(y, m, d)])
                    df_data.append(
                        [f"doc{doc_id}#wc_{2}", p3, np.random.choice([0, 1], 1)[0], "{}-{:02}-{:02}".format(y, m, d)])
                    doc_id += 1

            temp = pd.DataFrame(df_data, columns=paragraphDF.columns, index=range(st_ind, st_ind + len(df_data)))
            temp["created"] = pd.to_datetime(temp["created"])
            paragraphDF = pd.concat([paragraphDF, temp])

            paragraphDF["created"] = paragraphDF["created"].apply(lambda x: str(x) if x is not None else x)
            write(config.passage_docids_file, paragraphDF["doc_id"].values, mode="wt", compress=True)
            write_json_dump(config.passage_dict_file, paragraphDF.to_dict(orient="records"))
            write_dict(config.date_dict_file, paragraphDF.set_index("doc_id")["created"], mode="wt", compress=True)
            paragraphDF["created"] = pd.to_datetime(paragraphDF["created"])
            #### END MOCKUP ###

        # Identifying Collection Splits, i.e., Parts
        st = 0
        en = collection_split_size
        pidx = 0
        parts = {}
        while st < paragraphDF.shape[0]:
            parts[pidx] = (st, st + paragraphDF.iloc[st:en].shape[0])
            pidx += 1
            st = en
            en = st + collection_split_size

        write_dict(config.passage_parts_file, parts)
        self.log.warning(f"Identified {len(parts)} parts (i.e., splits) of the collection.")

        return paragraphDF