FROM nvidia/cuda:10.2-runtime-ubuntu18.04

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata


RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    build-essential \
    curl \ 
    zip unzip\
    openjdk-8-jre \
    git-lfs \
    vim \
    software-properties-common \
    locales

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN locale-gen en_US.UTF-8
ENV LANGUAGE en_US:en
    
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.8 && \
	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2  &&\
	 update-alternatives --set python3 /usr/bin/python3.8 && apt-get install -y gcc python3-pip python3.8-venv python3.8-dev
RUN apt-get clean && apt-get -y autoremove && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/nltk_data/ /app/neuralcoref/
ENV NLTK_DATA=/app/nltk_data/
ENV NEURALCOREF_CACHE=/app/neuralcoref/
RUN pip3 install -U pip setuptools wheel && pip3 install jupyter -U && pip3 install jupyterlab

COPY requirements.txt /app
#COPY neuralcoref /usr/local/lib/python3.8/dist-packages/neuralcoref
RUN pip3 install --no-cache-dir -r /app/requirements.txt && \
	giveme5w1h-corenlp install

COPY neuralcoref /tmp/neuralcoref
COPY neuralcoref_cache /app/neuralcoref
RUN pip3 install -r /tmp/neuralcoref/requirements.txt && pip3 install /tmp/neuralcoref && pip3 install protobuf==3.19.6
RUN python3 -m spacy download en_core_web_sm && \
	python3 -c "import benepar; benepar.download('benepar_en3')" && \
	python3 -c "import nltk; nltk.download('wordnet','/app/nltk_data'); nltk.download('stopwords','/app/nltk_data'); nltk.download('omw-1.4','/app/nltk_data')"

RUN mkdir /app/SourceSets ; mkdir /app/DocumentSets ; mkdir /app/DocumentSets/ccd ; mkdir /app/DocumentSets/json   ; mkdir /app/sourceCode ; mkdir /app/transformer_models; mkdir -p  /usr/local/lib/python3.8/dist-packages/Giveme5W1H/examples/caches/

RUN git lfs install \
	&& git clone https://huggingface.co/sentence-transformers/all-distilroberta-v1 /app/transformer_models/all-distilroberta-v1 \
	&& git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 /app/transformer_models/all-MiniLM-L6-v2

RUN cd /tmp && \
	wget https://gla-my.sharepoint.com/:u:/g/personal/h_narvala_1_research_gla_ac_uk/Eb5Jm48Dd-ZIiAvHrYSdHTQBs_5LLR5TFGvnn1FNXJG36w?download=1 -O matlab-install.tar.gz && \
	tar -xvzf /tmp/matlab-install.tar.gz

COPY matlab_installer_input.txt /tmp/matlab_installer_input.txt
RUN cd /tmp/matlab-install/ && \
    chmod +x ./install && \
    ./install -mode silent \
        -inputFile /tmp/matlab_installer_input.txt \
        -outputFile /tmp/mlinstall.log \
        -destinationFolder /usr/local/MATLAB \
    ; EXIT=$? && cat /tmp/mlinstall.log && test $EXIT -eq 0 && rm -rf /tmp/matlab-install/
    
COPY startmatlab.sh /opt/startscript/
RUN chmod +x /opt/startscript/startmatlab.sh && \
    ln -s /usr/local/MATLAB/bin/matlab /usr/local/bin/matlab

COPY license.lic /usr/local/MATLAB/licenses/  
RUN cd /usr/local/MATLAB/extern/engines/python/ && python3 setup.py install #--prefix="/local/work/matlab20bPy36"

ENV TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN cd /app

COPY sourceCode/ /app/sourceCode
COPY 92Folder/ /app/92Folder

RUN cd /app && chmod 777 `find . -type d`  && chmod 666 `find . -type f` && chmod 777 /usr/local/lib/python3.8/dist-packages/Giveme5W1H/examples/caches


# jupyter notebook
EXPOSE 8888

COPY docker_startup.sh /tmp/
