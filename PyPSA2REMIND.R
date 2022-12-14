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

# PyPSA to REMIND technology mapping
py2rm_tech <- c(
  "biomass" = "biomass",
  "CCGT" = "CCGT",
  "coal" = "all_coal",  # Aggregate coal
  "lignite" = "all_coal",  # Aggregate coal
  "nuclear" = "nuclear",
  "OCGT" = "OCGT",
  "offwind-ac" = "all_offwind",  # Aggregate offshore wind
  "offwind-dc" = "all_offwind",  # Aggregate offshore wind
  "oil" = "oil",
  "onwind" = "onwind",
  "ror" = "ror",
  "solar" = "solar"
)

# PyPSA to REMIND region mapping
py2rm_region <- c(
  "DE0" = "DEU"
)

# Function to calculate capacity factor
py2rm_calc_capfac <- function(pypsa.folder) {
  # Initialise output
  cf <- NULL
  # Get directories
  dirs <- list.dirs(path = pypsa.folder,
                    recursive = FALSE)
  print(paste(length(dirs), "PyPSA output directories found."))
  # Loop over yearly PyPSA output
  for (d in dirs) {
    # Get year
    y <- str_extract(d, "(?<=y)\\d{4}")
    
    # Get production
    g_p <- read_csv(file.path(d, "generators-p.csv"),
                    col_types = "n") %>%
      rename(hour = ...1) %>%
      pivot_longer(
        cols = !hour,
        names_to = c("region", "bus", "tech"),
        names_sep = " ",
        values_to = "p"
      ) %>%
      # Sum over year
      group_by(region, tech, bus) %>% 
      summarise(p_sum = sum(p))
    
    # Get optimal nominal capacity
    g <- read_csv(file.path(d, "generators.csv"),
                  show_col_types = F) %>%
      select(name, p_nom_opt) %>%
      separate(name,
               into = c("region", "bus", "tech"),
               sep = " ")
    
    # Calculate capacity factor
    calc <- full_join(g, g_p) %>%
      # Aggregate technologies
      revalue.levels(tech = py2rm_tech,
                     region = py2rm_region) %>% 
      group_by(region, tech, bus) %>% 
      summarise(p_sum = sum(p_sum),
                p_nom_opt = sum(p_nom_opt)) %>% 
      # Calculate capacity factors over regions
      summarise(value = sum(p_sum) / (sum(p_nom_opt) * 8760)) %>%
      # Add year
      mutate(year = y,
             var = "capfac") %>% 
      select(year, region, tech, var, value)
    
    # Append to output
    cf <- bind_rows(cf, calc)
  }
  return(cf)
}

cf <- py2rm_calc_capfac(file.path("results", "networks", RDIR_rm))

# Aggregate data
PyPSA2REMIND <- bind_rows(cf)

# Write GDX parameter
writegdx.parameter("PyPSA2REMIND.gdx", data.table(PyPSA2REMIND),
                   name = "PyPSA2REMIND",
                   valcol = "value",
                   uelcols = c("tPy32", "regPy32", "tePyImp32", "varPyImp32"))