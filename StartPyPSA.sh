#!/bin/bash
# Pass iteration number as command line argument
echo "Starting PyPSA in iteration $1"
cd "$(dirname "$0")"
source venv/bin/activate
snakemake -s Snakefile_REMIND_solve --profile simple/ export_all_networks_rm --config iter=$1