import logging
import queue
from threading import Thread

from Giveme5W1H.extractor.combined_scoring import distance_of_candidate
from Giveme5W1H.extractor.extractors import action_extractor, cause_extractor, method_extractor

from extract_5w1h.environment_extractor import EnvironmentExtractor

class Worker(Thread):
    def __init__(self, queue):
        ''' Constructor. '''
        Thread.__init__(self)
        self._queue = queue
        self.exception = None

    def run(self):
        while True:
            extractor, document = self._queue.get()
            if extractor and document:
                try:
                    extractor.process(document)
                except Exception as e:
                    self.exception = e
                self._queue.task_done()

    def get_exception(self):
        return self.exception

    def reset_exception(self):
        self.exception = None

class MasterExtractor:
    """
    The MasterExtractor bundles all parsing modules.
    """

    log = None
    extractors = []
    combinedScorers = None

    def __init__(self, extractors=None, combined_scorers=None, enhancement=None,multithreaded=True,skip_where=False):
        """
         Initializes the given preprocessor and extractors.
        :param extractors:
        :param combined_scorers: None will load defaults, [] will run without
        :param enhancement:
        """
        # RuntimeResourcesInstaller.check_and_install()

        # first initialize logger
        self.log = logging.getLogger('GiveMe5W')

        # initialize extractors
        if extractors is not None and len(extractors) > 0:
            self.extractors = extractors
        else:
            # the default extractor selection
            self.log.info('No extractors passed: initializing default configuration.')
            self.extractors = [
                action_extractor.ActionExtractor(),
                EnvironmentExtractor(skip_where=skip_where),
                cause_extractor.CauseExtractor(),
                method_extractor.MethodExtractor()
            ]

        if combined_scorers is not None:
            self.combinedScorers = combined_scorers
        else:
            self.log.info('No combinedScorers passed: initializing default configuration.')

            self.combinedScorers = [
                # ['what'], 'how'
                distance_of_candidate.DistanceOfCandidate()
            ]

        if multithreaded:
            self.q = queue.Queue()
            # creating worker threads
            self.threads=[]
            for i in range(len(self.extractors)):
                t = Worker(self.q)
                self.threads.append(t)

                t.daemon = True
                t.start()

        self.enhancement = enhancement
        self.multithreaded=multithreaded

    def parse(self, doc):
        """
        Pass a document to the preprocessor and the extractors
        :param doc: document object to parse
        :type doc: Document
        :return: the processed document
        """
        if self.multithreaded:
            #Reset Previous Exceptions if programs continues
            for t in self.threads:
                t.reset_exception()

            # run extractors in different threads
            for extractor in self.extractors:
                self.q.put((extractor, doc))

            # wait till oll extractors are done
            self.q.join()

            #Fail if any thread failed
            for t in self.threads:
                e = t.get_exception()
                if e:
                    raise e
        else:
            for extractor in self.extractors:
                extractor.process(doc)

        # apply combined_scoring
        if self.combinedScorers and isinstance(self.combinedScorers, list) and len(self.combinedScorers) > 0:
            for combinedScorer in self.combinedScorers:
                combinedScorer.score(doc)
        doc.is_processed(True)

        # enhancer: linking answers(candidate-Objects) to enhancer-data
        if self.enhancement:
            for enhancement in self.enhancement:
                enhancement.enhance(doc)

        return doc