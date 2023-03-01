# import spacy
# import neuralcoref
# import benepar
import logging
import nltk

spacy2corenlp_ents = {
    "PERSON": "PERSON",
    "FAC": "LOCATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "EVENT": "MISC",
    "EVENT": "MISC",
    "LANGUAGE": "MISC",
    "LAW": "MISC",
    "NORP": "MISC",
    "PRODUCT": "MISC",
    "QUANTITY": "MISC",
    "WORK_OF_ART": "MISC",

    "MONEY": "MONEY",
    "CARDINAL": "NUMBER",
    "ORDINAL": "ORDINAL",
    "PERCENT": "PERCENT",

    "DATE": "DATE",
    "TIME": "TIME",

    "": "O"
}

MENTION_TYPE = {"PRONOMINAL": 0, "NOMINAL": 1, "PROPER": 2, "LIST": 3}
MENTION_LABEL = {0: "PRONOMINAL", 1: "NOMINAL", 2: "PROPER", 3: "LIST"}
ACCEPTED_ENTS = ["PERSON", "NORP", "FACILITY", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE", ]


def get_mention_type(span):
    """ Find the type of the Span """
    conj = ["CC", ","]
    prp = ["PRP", "PRP$"]
    proper = ["NNP", "NNPS"]
    if any(t.tag_ in conj and t.ent_type_ not in ACCEPTED_ENTS for t in span):
        mention_type = MENTION_TYPE["LIST"]
    elif span.root.tag_ in prp:
        mention_type = MENTION_TYPE["PRONOMINAL"]
    elif span.root.ent_type_ in ACCEPTED_ENTS or span.root.tag_ in proper:
        mention_type = MENTION_TYPE["PROPER"]
    else:
        mention_type = MENTION_TYPE["NOMINAL"]
    return mention_type



def remove_unserializable_results(doc):
    temp = doc._.coref_clusters

    doc_sent_ids = dict([(sent.start, idx) for idx, sent in enumerate(doc.sents)])

    doc_corefs = {}
    coref_idx = 1
    for c in list(temp):
        cluster = []
        for m in c.mentions:
            m_dict = {'id': str(coref_idx),
                      'text': m.text,
                      'type': MENTION_LABEL[get_mention_type(m)],
                      'number': 'SINGULAR',  # NA
                      'gender': '',  # NA
                      'animacy': 'ANIMATE',  # NA
                      'startIndex': m[0].i - m.sent.start + 1,
                      'endIndex': m[0].i - m.sent.start + len(m) + 1,
                      'headIndex': m.root.i - m.sent.start + 1,
                      'sentNum': doc_sent_ids[m[0].sent.start] + 1,
                      'position': [doc_sent_ids[m[0].sent.start] + 1, 1],  # NA
                      'isRepresentativeMention': m == c.main}

            cluster.append(m_dict)
            coref_idx += 1

        doc_corefs[coref_idx - 1] = cluster

    doc.user_data = {}
    doc.user_data = {"coref": doc_corefs}

    for x in dir(doc._):
        getattr(doc._, x)

    for x in dir(doc._):
        if x in ['get', 'set', 'has', 'coref_as_ner']: continue
        setattr(doc._, x, None)

    for token in doc:
        for x in dir(token._):
            if x in ['get', 'set', 'has', 'coref_as_ner', 'labels']: continue
            setattr(token._, x, None)
    return doc

class PreprocessorSpacy:
    log = None

    def __init__(self):#, spacy_model="en_core_web_sm"):
        """
        This preprocessor connects to an CoreNLP server to perform sentence splitting, tokenization, syntactic parsing,
        named entity recognition and coref-resolution on passed documents.
        :param host: the core-nlp host
        """

        self.log = logging.getLogger('GiveMe5W')
        # self.nlp = spacy.load(spacy_model)
        # neuralcoref.add_to_pipe(self.nlp)
        # self.nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
        #         self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

        self._token_index = None

    def _link_leaf_to_core_nlp(self, s):
        """
        this is where the magic happens add there additional information per candidate-part/token/leave
        char index information is in each nlpToken
        """
        if len(self._tokens) - 1 < self._token_index:
            # there seams a bug around numbers,
            # spitted numbers in the same token are called as they have been split to different tokens
            # this leads to a wrong index, everything in this sentence is lost till the end of that sentence
            # self.log.error('fix the doc around(reformat number,remove special characters):' + s)
            # # print the last two tokens to make it spotable
            # self.log.error(self._tokens[-1])
            # self.log.error(self._tokens[-2])

            # further we can`t return None because this would break extractors
            # therefore we use this bugfix object
            # TODO: reason if it make sense to reject these documents at all, because result isn`t reliable at all
            # TODO: flag document at least with some error flags
            result = {
                'nlpToken': {
                    'index': 7,
                    'word': 'BUGFIX',
                    'originalText': 'BUGFIX',
                    'lemma': 'BUGFIX',
                    'characterOffsetBegin': 0,
                    'characterOffsetEnd': 0,
                    'pos': 'BUGFIX',
                    'ner': 'BUGFIX',
                    'speaker': 'BUGFIX',
                    'before': ' ',
                    'after': ''
                }
            }

            if self._document:
                self._document.set_error_flag('core_nlp')


        else:
            result = {
                'nlpToken': self._tokens[self._token_index]
            }

        self._token_index = self._token_index + 1

        return result

    def _build_actual_config(self, document):
        """
        Creates the actual config, consisting of the base_config and dynamic_params. if the same key exists in both
        base_config and dynamic_params, the value will be used from dynamic_params, i.e., base_config will be overwritten.
        :param document:
        :return:
        """
        dynamic_config = {
            'date': document.get_date()
        }
        actual_config = {**self.base_config, **dynamic_config}
        return actual_config

    def preprocess(self, document,annotation):
        """
        Send the document to CoreNLP server to execute the necessary preprocessing.
        :param document: Document object to process.
        :type document: Document
        :return Document: The processed Document object.
        """

        # annotation = self.nlp(document.get_full_text())

        prev_ws = ""
        doc_sentences = []
        tree = []
        doc_sent_ids = {}

        tokens = []
        pos = []
        ner = []
        for sentence_idx, sentence in enumerate(annotation.sents):
            token_idx = 1

            s_tokens = []
            s_pos = []
            s_ner = []
            for token in sentence:
                token_dict = {'index': token_idx,
                              'word': token.text,
                              'originalText': token.text,
                              'lemma': token.lemma_,
                              'characterOffsetBegin': token.idx,
                              'characterOffsetEnd': token.idx + len(token.text),
                              'pos': token.tag_,
                              'ner': spacy2corenlp_ents[token.ent_type_],
                              'speaker': 'BUGFIX',
                              'before': prev_ws,
                              'after': token.whitespace_}

                s_tokens.append(token_dict)
                s_pos.append((token_dict['originalText'], token_dict['pos']))
                s_ner.append((token_dict['originalText'], token_dict['ner']))

                token_idx += 1
                prev_ws = token.whitespace_

                sentence_dict = {"index": sentence_idx,
                                 "parse": "(ROOT " + sentence._.parse_string + ")",
                                 "tokens": s_tokens
                                 }

            tokens.append(s_tokens)
            pos.append(s_pos)
            ner.append(s_ner)

            # that's a hack to add to every tree leave a the tokens result
            self._token_index = 0
            self._tokens = s_tokens
            sentence_tree = nltk.ParentedTree.fromstring(sentence_dict['parse'], read_leaf=self._link_leaf_to_core_nlp)

            # add a reference to the original data from parsing for this sentence
            sentence_tree.stanfordCoreNLPResult = sentence_dict

            tree.append(sentence_tree)

            doc_sentences.append(sentence_dict)
            doc_sent_ids[sentence.start] = sentence_idx

        document.set_sentences(doc_sentences, [], [])
        self._document = document
        document.set_trees(tree)

        # Parse Corefs
        # doc_corefs = {}
        # coref_idx = 1
        # for c in list(annotation._.coref_clusters):
        #     cluster = []
        #     for m in c.mentions:
        #         m_dict = {'id': str(coref_idx),
        #                   'text': m.text,
        #                   'type': MENTION_LABEL[get_mention_type(m)],
        #                   'number': 'SINGULAR',  # NA
        #                   'gender': '',  # NA
        #                   'animacy': 'ANIMATE',  # NA
        #                   'startIndex': m[0].i - m.sent.start + 1,
        #                   'endIndex': m[0].i - m.sent.start + len(m) + 1,
        #                   'headIndex': m.root.i - m.sent.start + 1,
        #                   'sentNum': doc_sent_ids[m[0].sent.start] + 1,
        #                   'position': [doc_sent_ids[m[0].sent.start] + 1, 1],  # NA
        #                   'isRepresentativeMention': m == c.main}
        #
        #         cluster.append(m_dict)
        #         coref_idx += 1
        #
        #     doc_corefs[coref_idx - 1] = cluster

        document.set_corefs(annotation.user_data["coref"])

        document.set_tokens(tokens)
        document.set_pos(pos)
        document.set_ner(ner)
        document.set_enhancement('coreNLP', None)
        document.is_preprocessed(True)
        return document