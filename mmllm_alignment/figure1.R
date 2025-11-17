# This R script is adapted from the supplementary code 
# provided in 'The Moral Machine Experiment',
# which can be accessed at https://goo.gl/JXRrBP.

library(ggplot2)
library(reshape2)
library(dplyr)
library(tidyr)
library(AER)
library(sandwich)
library(multiwayvcov)
library(data.table)
library(optparse)

# Define command-line arguments
option_list <- list(
  make_option(c("--model"), type="character", default="Qwen3-0.6B", 
              help="Model name [default= %default]", metavar="character"),
  make_option(c("--nb_scenarios"), type="integer", default=1000, 
              help="Number of scenarios [default= %default]", metavar="integer"),
  make_option(c("--save_dir"), type="character", default="../results", 
              help="Directory to save output [default= %default]", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
args <- parse_args(opt_parser)

# Construct input and output file paths
input_file <- file.path(args$save_dir, args$nb_scenarios, args$model, paste0("shared_responses_", args$model, ".csv"))
output_dir <- file.path(args$save_dir, args$nb_scenarios, args$model)
main_csv <- file.path(output_dir, paste0("plotdata_main_", args$model, ".csv"))
util_csv <- file.path(output_dir, paste0("plotdata_util_", args$model, ".csv"))
human_csv <- file.path(output_dir, paste0("plotdata_main_human_", args$model, ".csv"))
combined_csv <- file.path(output_dir, paste0("acme_", args$model, ".csv"))

# Ensure save directory exists
dir.create(output_dir, showWarnings=FALSE, recursive=TRUE)

source("chatbot_MMFunctionsShared.R")

# Loading data as a data.table
profiles <- fread(input=input_file)
profiles <- PreprocessProfiles(profiles)

# Compute ACME values
Coeffs.main <- GetMainEffectSizes(profiles, TRUE, 9)
plotdata.main <- GetPlotData(Coeffs.main, TRUE, 9)
print("plotdata.main:")
print(plotdata.main)  # View AMCE values
write.csv(plotdata.main, main_csv, row.names=FALSE)

# Compute additional ACME values
Coeffs.util <- GetMainEffectSizes.Util(profiles)
plotdata.util <- GetPlotData.Util(Coeffs.util)
print("plotdata.util:")
print(plotdata.util)  # View util values
write.csv(plotdata.util, util_csv, row.names=FALSE)

# Create and save plotdata.main.human
plotdata.main.human <- data.frame(
  Estimates = c(0.061, 0.097, 0.353, 0.119, 0.160, 0.345, 0.497, 0.651, 0.585),
  Label = c("Intervention", "Relation to AV", "Law", "Gender", "Fitness", "Social Status", "Age", "No. Characters", "Species")
)
print("plotdata.main.human:")
print(plotdata.main.human)  # View human AMCE values
write.csv(plotdata.main.human, human_csv, row.names=FALSE)

# Combine plotdata.main and plotdata.main.human (main only, exclude SE)
# Assuming plotdata.main has columns like 'Label', 'Estimates', and possibly 'SE'
# Select only 'Label' and 'Estimates' from plotdata.main
plotdata.main.selected <- plotdata.main[, c("Label", "Estimates")]
colnames(plotdata.main.selected) <- c("Label", "Model")  # Rename Estimates to Model

# Select 'Label' and 'Estimates' from plotdata.main.human and rename to Human
plotdata.human.selected <- plotdata.main.human[, c("Label", "Estimates")]
colnames(plotdata.human.selected) <- c("Label", "Human")

# Merge on Label
combined_data <- merge(plotdata.main.selected, plotdata.human.selected, by="Label")

# Reorder columns: Label, Model, Human
combined_data <- combined_data[, c("Label", "Model", "Human")]

print("Combined Data:")
print(combined_data)  # View combined data

# Save combined CSV
write.csv(combined_data, combined_csv, row.names=FALSE)

# Plot and save
PlotAndSave(plotdata.main, TRUE, file.path(output_dir, "MainChangePr"), plotdata.util)