#!/bin/bash
# Script to start PyPSA from REMIND
# Pass iteration number and name of output directory as command line argument
echo "Starting PyPSA in iteration $1, output folder $2"
cd "$(dirname "$0")"
source /home/adrianod/software/venv_pypsa-eur/bin/activate
snakemake -call -s Snakefile_remind --profile cluster_config/ export_all_networks_remind --config iter=$1 RDIR_remind=$2 opt=solve