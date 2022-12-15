#!/bin/bash
# Script to start PyPSA from REMIND
# Pass iteration number and name of output directory as command line argument
echo "Starting PyPSA in iteration $1, output folder $2"
cd "$(dirname "$0")"
source venv/bin/activate
snakemake -s Snakefile_REMIND_solve --profile cluster_config/ export_all_networks_rm --config iter=$1 RDIR_rm=$2