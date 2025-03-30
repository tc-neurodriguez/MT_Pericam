# Load required libraries
library(readxl)
library(tidyverse)
library(reshape2)
library(ggplot2)
library(rstatix)
library(ggbeeswarm)
library(pheatmap)

# Set file path and define wells
file_path <- "C:/Users/Tyler/Box/SY5Y transduction Images/BioSpa_Data/Experiment_3_17_25/experiment 3_17_25/MT-Pericam_3_17_25_FCCP2.xlsx"
wells <- c('B3','B4','B5','B6','B7','B8','B9','B10',
           'C3','C4','C5','C6','C7','C8','C9','C10',
           'D3','D4','D5','D6','D7','D8','D9','D10',
           'E3','E4','E5','E6','E7','E8','E9','E10')

# Define groups
groups <- list(
  'Vehicle (0.1% DMSO)' = c('B3','B4','B5','B6','B7','B8','B9','B10'),
  '0.1_uM_FCCP' = c('C3','C4','C5','C6','C7','C8','C9','C10'),
  '1_uM_FCCP' = c('D3','D4','D5','D6','D7','D8','D9','D10'),
  '10_uM_FCCP' = c('E3','E4','E5','E6','E7','E8','E9','E10')
)

background_wells <- paste0('G', 3:10)

# 1. Data Loading function
load_and_process <- function(sheet_name, wavelength) {
  message("Loading ", wavelength, " from ", sheet_name)
  df <- read_excel(file_path, sheet = sheet_name) %>%
    rename_with(~ gsub("[^A-Za-z0-9]", "_", .x))  # Clean column names

  message("Initial columns: ", paste(colnames(df), collapse = ", "))

  # Melt data
  melted <- df %>%
    select(Time, all_of(intersect(wells, colnames(df)))) %>%
    pivot_longer(cols = -Time, names_to = "Well", values_to = wavelength)

  message("Melted data dimensions: ", paste(dim(melted), collapse = " x "))
  return(melted)
}
# 2. Load and merge data
cat("\n", strrep("=", 50), "\n")
cat("🚀 STARTING DATA PROCESSING\n")
cat(strrep("=", 50), "\n")

wavelengths <- list(
  'Sheet1' = '370_470',
  'Sheet2' = '415_518',
  'Sheet3' = '485_525',
  'Sheet4' = '555_586'
)

merged <- NULL
for (sheet in names(wavelengths)) {
  name <- wavelengths[[sheet]]
  df <- load_and_process(sheet, name)

  if (is.null(merged)) {
    merged <- df
  } else {
    merged <- merge(merged, df, by = c("Time", "Well"), all = TRUE)
  }

  cat(paste("🔗 Merged dimensions after", name, ":", paste(dim(merged), collapse = " x "), "\n"))
  cat(paste(" NA count:", sum(is.na(merged)), "\n"))
}

cat("\n🔎 Final merged data preview:\n")
print(head(merged))
cat("\n📉 Missing values per column:\n")
print(colSums(is.na(merged)))

# 3. Calculate ratios
cat("\n", strrep("=", 50), "\n")
cat("🧮 CALCULATING RATIOS\n")
cat(strrep("=", 50), "\n")

# Sort by time
merged <- merged %>% arrange(Time)

# Define baseline time point
baseline_time_point <- 0

# Calculate F0 values
F0_values <- merged %>%
  filter(Time == baseline_time_point) %>%
  summarise(
    F0_415_518 = first(`415_518`),
    F0_485_525 = first(`485_525`),
    F0_555_586 = first(`555_586`),
    F0_370_470 = first(`370_470`)
  )

# Add F0 columns to the DataFrame
merged <- merged %>%
  mutate(
    F0_415_518 = F0_values$F0_415_518,
    F0_485_525 = F0_values$F0_485_525,
    F0_555_586 = F0_values$F0_555_586,
    F0_370_470 = F0_values$F0_370_470
  )

# Calculate ratios and ΔF/F0
merged <- merged %>%
  mutate(
    `485_525/415_518` = `485_525` / `415_518`,
    `415_518/555_586` = `415_518` / `555_586`,
    `485_525/555_586` = `485_525` / `555_586`,
    `555_586/(370_470+555_586)` = `555_586` / (`370_470` + `555_586`),
    `Delta_F415_518/F0_415_518` = -(`415_518` - F0_415_518) / F0_415_518,
    `Delta_F485_525/F0_485_525` = -(`485_525` - F0_485_525) / F0_485_525,
    `Delta_F555_586/F0_555_586` = -(`555_586` - F0_555_586) / F0_555_586,
    `Delta_F370_470/F0_370_470` = -(`370_470` - F0_370_470) / F0_370_470,
    `Norm_485_525/415_518` = `Delta_F485_525/F0_485_525` / `Delta_F415_518/F0_415_518`,
    `Norm_415_518/555_586` = `Delta_F415_518/F0_415_518` / `Delta_F555_586/F0_555_586`,
    `Norm_485_525/555_586` = `Delta_F485_525/F0_485_525` / `Delta_F555_586/F0_555_586`,
    `Norm_555_586/(370_470+555_586)` = `Delta_F555_586/F0_555_586` / (`Delta_F370_470/F0_370_470` + `Delta_F555_586/F0_555_586`)
  ) %>%
  # Replace Inf with NA
  mutate(across(where(is.numeric), ~ ifelse(is.infinite(.), NA, .)))

# Filter to only include time points after baseline
merged <- merged %>% filter(Time > baseline_time_point)

# Get data at time = 10
merged_5 <- merged %>% filter(Time == 10, Well %in% wells)

# Calculate z-scores
merged_5 <- merged_5 %>%
  mutate(
    `485_525/415_518_zscore` = scale(`485_525/415_518`),
    `415_518/555_586_zscore` = scale(`415_518/555_586`),
    `485_525/555_586_zscore` = scale(`485_525/555_586`),
    `555_586/(370_470+555_586)_zscore` = scale(`555_586/(370_470+555_586)`),
    `Delta_F415_518/F0_415_518_zscore` = scale(`Delta_F415_518/F0_415_518`),
    `Delta_F485_525/F0_485_525_zscore` = scale(`Delta_F485_525/F0_485_525`),
    `Norm_485_525/415_518_zscore` = scale(`Norm_485_525/415_518`),
    `Norm_415_518/555_586_zscore` = scale(`Norm_415_518/555_586`),
    `Norm_485_525/555_586_zscore` = scale(`Norm_485_525/555_586`),
    `Norm_555_586/(370_470+555_586)_zscore` = scale(`Norm_555_586/(370_470+555_586)`)
  )

# 2. Create plate grid (fixed version)
create_plate_grid <- function(data, well_column, value_column) {
  plate <- matrix(NA, nrow = 8, ncol = 12)

  for (i in 1:nrow(data)) {
    well <- as.character(data[[well_column]][i])
    if (!is.na(well) && nchar(well) > 1) {
      row_char <- substr(well, 1, 1)
      col_num <- as.numeric(substr(well, 2, nchar(well)))

      row_idx <- which(LETTERS[1:8] == row_char)
      col_idx <- col_num

      if (length(row_idx) == 1 && !is.na(col_idx) && col_idx <= 12) {
        plate[row_idx, col_idx] <- data[[value_column]][i]
      }
    }
  }
  return(plate)
}

# Create grids for each z-score column
plate_485_525_415_518 <- create_plate_grid(merged_5, "Well", "485_525/415_518_zscore")
plate_415_518_555_586 <- create_plate_grid(merged_5, "Well", "415_518/555_586_zscore")
plate_485_525_555_586 <- create_plate_grid(merged_5, "Well", "485_525/555_586_zscore")
plate_555_586_370_470 <- create_plate_grid(merged_5, "Well", "555_586/(370_470+555_586)_zscore")

# Function to plot heatmap
plot_plate_heatmap <- function(plate, title, vmin = -2, vmax = 2) {
  rownames(plate) <- LETTERS[1:8]
  colnames(plate) <- 1:12

  pheatmap(plate,
           cluster_rows = FALSE,
           cluster_cols = FALSE,
           display_numbers = TRUE,
           number_format = "%.2f",
           main = title,
           breaks = seq(vmin, vmax, length.out = 100),
           color = colorRampPalette(c("blue", "white", "red"))(100),
           na_col = "gray",
           fontsize_number = 8,
           angle_col = 0)
}

# Create multi-panel heatmap plot
par(mfrow = c(2, 2))
plot_plate_heatmap(plate_485_525_415_518, "FCCP 485_525/415_518 Z-Scores")
plot_plate_heatmap(plate_415_518_555_586, "FCCP 415_518/555_586 Z-Scores")
plot_plate_heatmap(plate_485_525_555_586, "FCCP 485_525/555_586 Z-Scores")
plot_plate_heatmap(plate_555_586_370_470, "FCCP 555_586/(370_470+555_586) Z-scores")

# Kruskal-Wallis and Dunn's test analysis
# Convert groups to data frame
group_df <- map2_dfr(groups, names(groups), ~ tibble(Well = .x, Group = .y))

# Merge group information
gm_df <- merged_5 %>%
  left_join(group_df, by = "Well")

# Function to create boxplot with statistical annotations
create_boxplot <- function(data, y_col, title, ylabel) {
  # Kruskal-Wallis test
  kruskal_test <- data %>%
    kruskal_test(as.formula(paste(y_col, "~ Group")))

  cat(paste(title, "- Kruskal-Wallis p-value:", kruskal_test$p, "\n"))

  # Dunn's post hoc test
  dunn_test <- data %>%
    dunn_test(as.formula(paste(y_col, "~ Group")), p.adjust.method = "bonferroni")

  cat(paste(title, "- Dunn's test results:\n"))
  print(dunn_test)

  # Create plot
  p <- ggplot(data, aes(x = Group, y = !!sym(y_col), fill = Group)) +
    geom_boxplot(aes(fill = Group), outlier.shape = NA) +
    geom_beeswarm(size = 2, cex = 3) +
    stat_pvalue_manual(
      dunn_test %>% filter(p.adj < 0.05),
      label = "p.adj.signif",
      y.position = max(data[[y_col]], na.rm = TRUE) * (1 + 0.1 * (1:nrow(dunn_test %>% filter(p.adj < 0.05)))),
      step.increase = 0.1
    ) +
    labs(title = title, x = "Group", y = ylabel) +
    theme_minimal() +
    theme(legend.position = "none")

  return(p)
}

# Create multi-panel boxplot
p1 <- create_boxplot(gm_df, "485_525/415_518",
                    "Mitochondrial Function 3min exposure to FCCP",
                    "Pericam Fluorescence Ratio (485_525/415_518)")

p2 <- create_boxplot(gm_df, "415_518/555_586",
                    "Mitochondrial Calcium 3min exposure to FCCP",
                    "Pericam Fluorescence Ratio (415_518/555_586)")

p3 <- create_boxplot(gm_df, "485_525/555_586",
                    "Mitochondrial pH 3min exposure to FCCP",
                    "Pericam Fluorescence Ratio (485_525/555_586)")

p4 <- create_boxplot(gm_df, "555_586/(370_470+555_586)",
                    "Mitochondrial volume 3min exposure to FCCP",
                    "Pericam Fluorescence Ratio (555_586/(370_470+555_586))")

# Arrange plots
ggarrange(p1, p2, p3, p4, ncol = 2, nrow = 2) %>%
  annotate_figure(top = text_grob("Multipanel Figure: Mitochondrial Function via Ratiometric Pericam",
                                 face = "bold", size = 16))