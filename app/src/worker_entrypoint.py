# src/worker_entrypoint.py
import os
import sys
from distributed.deploy.local import LocalCluster

# -----------------------------------------------------------
# Dask worker entrypoint script
# -----------------------------------------------------------
# This script is intended to be the startup entrypoint for a Dask
# worker container (as defined in docker-compose.yml).
#
# It performs the following steps:
#   1. Reads the scheduler address from the environment variable
#      DASK_SCHEDULER_ADDRESS (defaulting to 'tcp://scheduler:8786').
#   2. Replaces the current Python process with a new Dask worker process,
#      using os.execvp() so that this container *becomes* the worker.
# -----------------------------------------------------------

# Get the address of the Dask scheduler from environment variables.
# This tells the worker which scheduler to connect to.
addr = os.environ.get('DASK_SCHEDULER_ADDRESS', 'tcp://scheduler:8786')

# Replace the current process with the Dask worker process.
# The arguments below configure:
#   --nthreads=2      : number of threads for this worker
#   --memory-limit=4GB: maximum memory allocation for this worker
#
# Note: os.execvp() does not spawn a new process; it overwrites the current one.
# After this call, no Python code below will execute.
os.execvp('dask-worker', ['dask-worker', addr, '--nthreads', '2', '--memory-limit', '4GB'])
