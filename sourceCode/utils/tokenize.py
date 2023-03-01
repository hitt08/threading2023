import spacy
import nltk

class Tokenizers:
    def __init__(self,nltk_dir="/app/nltk_data/",spacy_model="en_core_web_sm"):
        # nltk.download('stopwords',download_dir=nltk_dir)
        from nltk.corpus import stopwords
        self.stp=set(stopwords.words("english"))
        self.nlp=spacy.load(spacy_model,disable=[ "parser","ner"])  #Only Tokennize

    def nlp_tokenize(self,string):
        return [token.text for token in self.nlp(string)]

    def tokenize_lower(self,string):
        return [token.text.lower() for token in self.nlp(string) if token.text.lower() not in self.stp]

    def tokenize_pnct(self,string):
        return [token.text.lower() for token in self.nlp(string) if token.text.strip() != "" and token.text.lower() not in self.stp and not token.is_punct and "\n" not in token.text]

    def tokenize_pnct_lemma(self,string):
        return [token.lemma_.lower() for token in self.nlp(string) if token.text.strip() != "" and token.text.lower() not in self.stp and not token.is_punct and "\n" not in token.text]

    def tokenize_split(self,string, sep="~|~"):
        return string.split(sep)

    def merge_doc_tokens(self, docs, sep="~|~"):
        return [sep.join(doc) for doc in docs]

