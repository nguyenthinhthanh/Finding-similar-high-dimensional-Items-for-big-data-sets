# src/worker_entrypoint.py
import os
import sys
from distributed.deploy.local import LocalCluster


# Start loader (if needed) then run dask-worker
addr = os.environ.get('DASK_SCHEDULER_ADDRESS', 'tcp://scheduler:8786')
# run the worker process
os.execvp('dask-worker', ['dask-worker', addr, '--nthreads', '2', '--memory-limit', '4GB'])