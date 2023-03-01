giveme5w1h-corenlp 1>/tmp/corenlp_run.log 2>/tmp/corenlp_run.err &
jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token=${1}
