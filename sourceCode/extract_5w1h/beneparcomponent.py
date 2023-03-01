import torch
from benepar.integrations.downloader import load_trained_model
from benepar.integrations.spacy_plugin import PartialConstituentData,SentenceWrapper
import logging

log=logging.getLogger(__name__)


class BeneparComponent:

    name = "benepar"

    def __init__(
            self,
            name,
            subbatch_max_tokens=500,
            disable_tagger=False,
            batch_size="ignored",
            use_gpu=True
    ):
        """Load a trained parser model.
        Args:
            name (str): Model name, or path to pytorch saved model
            subbatch_max_tokens (int): Maximum number of tokens to process in
                each batch
            disable_tagger (bool, default False): Unless disabled, the parser
                will set predicted part-of-speech tags for the document,
                overwriting any existing tags provided by spaCy models or
                previous pipeline steps. This option has no effect for parser
                models that do not have a part-of-speech tagger built in.
            batch_size: deprecated and ignored; use subbatch_max_tokens instead
        """
        self._parser = load_trained_model(name)
        if use_gpu and torch.cuda.is_available():
            self._parser.cuda()
            log.warning("Using GPU for Constituency parsing.")
        else:
            log.warning("Using CPU for Constituency parsing.")


        self.subbatch_max_tokens = subbatch_max_tokens
        self.disable_tagger = disable_tagger

        self._label_vocab = self._parser.config["label_vocab"]
        label_vocab_size = max(self._label_vocab.values()) + 1
        self._label_from_index = [()] * label_vocab_size
        for label, i in self._label_vocab.items():
            if label:
                self._label_from_index[i] = tuple(label.split("::"))
            else:
                self._label_from_index[i] = ()
        self._label_from_index = tuple(self._label_from_index)

        if not self.disable_tagger:
            tag_vocab = self._parser.config["tag_vocab"]
            tag_vocab_size = max(tag_vocab.values()) + 1
            self._tag_from_index = [()] * tag_vocab_size
            for tag, i in tag_vocab.items():
                self._tag_from_index[i] = tag
            self._tag_from_index = tuple(self._tag_from_index)
        else:
            self._tag_from_index = None

    def __call__(self, doc):
        """Update the input document with predicted constituency parses."""
        # TODO(https://github.com/nikitakit/self-attentive-parser/issues/16): handle
        # tokens that consist entirely of whitespace.
        constituent_data = PartialConstituentData()
        wrapped_sents = [SentenceWrapper(sent) for sent in doc.sents]
        for sent, parse in zip(
                doc.sents,
                self._parser.parse(
                    wrapped_sents,
                    return_compressed=True,
                    subbatch_max_tokens=self.subbatch_max_tokens,
                ),
        ):
            constituent_data.starts.append(parse.starts + sent.start)
            constituent_data.ends.append(parse.ends + sent.start)
            constituent_data.labels.append(parse.labels)

            if parse.tags is not None and not self.disable_tagger:
                for i, tag_id in enumerate(parse.tags):
                    sent[i].tag_ = self._tag_from_index[tag_id]

        doc._._constituent_data = constituent_data.finalize(doc, self._label_from_index)
        return doc