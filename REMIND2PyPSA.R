# R script to pass data from REMIND to PyPSA
library(tidyverse)
library(quitte)
library(gdx)
library(gdxdt)

# Function arguments
args <- commandArgs(trailingOnly=TRUE)
RDIR_rm <- args[1]

# Set working directory to script directory
setwd("/p/tmp/adrianod/pypsa-eur/")

# Years to couple
years <- c(seq(2025, 2060, 5), seq(2070, 2110, 10), 2130, 2150)

# Function to scale load time series
rm2py_calc_load <- function(rm.file, py.load, out.folder){
  
  # Check if input files exist
  if (file.exists(rm.file) & file.exists(py.load)) print("All files found.")

  # Create output folder
  if (!dir.exists(out.folder)) dir.create(out.folder)

  # Read in secondary energy production
    seel <- readgdx(rm.file, "vm_prodSe") %>%
    as_tibble(.name_repair = "unique") %>%
    # Secondary electricity
    filter(all_enty...4 == "seel",
           all_regi == "DEU",
           tall %in% years) %>%
    group_by(tall, all_regi) %>%
    # Summarise over technologies
    summarise(value = sum(value)) %>%
    mutate(value = 1E6 * 8760 * value) %>%   # TWa to MWh
    revalue.levels(all_regi = c("DEU" = "DE"))
  
  # Import original load time series
  load <- read_csv(py.load, col_types = c("c", "n"))
  
  # Scale up time series
  for (y in years){
    seel.y <- seel %>% 
      filter(tall == y) %>% 
      pull(value)
    
    calc <- load %>% 
      mutate(DE = seel.y/sum(DE) * DE) %>% 
      mutate(DE = round(DE, 4))
    
    write_csv(calc, file.path(out.folder, paste0("load_y", y, ".csv")))
  }
}

# Call function
rm2py_calc_load(rm.file = "REMIND2PyPSA.gdx",
                py.load = "resources/RM_Py_default/load.csv",
                out.folder = file.path("resources", RDIR_rm))