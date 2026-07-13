#!/usr/bin/env Rscript
#
# scc_transition.R
#
# Transition zone enrichment analysis for SCC signatures.
# Tests whether SCC activity peaks at Stage 2 (transition zone) during NE progression.
#
# Analyses:
#   1. Stage-wise comparison (ANOVA + pairwise tests)
#   2. Z-progression binned analysis (identify peak location)
#   3. Transition index correlation
#   4. Cancer type stratification
#
# Input:
#   - Multi-signature SCC scores from 35_scc_multi_signature_analysis.R
#   - NESTOR scores with stage assignments
#
# Output:
#   - stage_statistics.csv: Stage-wise comparison results
#   - peak_analysis.csv: Peak location analysis
#   - cancer_type_analysis.csv: Cancer type stratified results
#
# Usage:
#   Rscript scripts/scc_transition.R

# ==============================================================================
# Load required packages
# ==============================================================================

required_packages <- c("ggplot2", "scales", "arrow", "patchwork", "dplyr",
                       "tidyr", "broom", "gridExtra")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
  library(arrow)
  library(patchwork)
  library(dplyr)
  library(tidyr)
  library(broom)
  library(gridExtra)
})

# ==============================================================================
# Source shared utilities
# ==============================================================================

# Get script directory using robust method for Rscript execution
get_script_dir <- function() {
  # Try command args first (works with Rscript)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grep("--file=", args)]
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }

  # Try sys.frame (works with source())
  tryCatch({
    ofile <- sys.frame(1)$ofile
    if (!is.null(ofile)) {
      return(dirname(normalizePath(ofile)))
    }
  }, error = function(e) {})

  # Fallback to working directory
  return(getwd())
}

script_dir <- get_script_dir()
utils_path <- file.path(script_dir, "shared", "utils.R")
if (file.exists(utils_path)) {
  source(utils_path)
}

# Load centralized color configuration
library(jsonlite)
project_root <- function() {
  if (dir.exists(file.path(getwd(), "plotting_utils"))) return(getwd())
  if (dir.exists(file.path(dirname(getwd()), "plotting_utils"))) return(dirname(getwd()))
  if (dir.exists(file.path(dirname(dirname(getwd())), "plotting_utils"))) return(dirname(dirname(getwd())))
  stop("Cannot find project root (plotting_utils/ not found)")
}
source(file.path(project_root(), "plotting_utils", "R_utils.R"))
amp_colors <- load_amphicrine_colors()
PROJECT_ROOT <- project_root()
source(file.path(PROJECT_ROOT, "config", "macaron_colors.R"))

# ==============================================================================
# Configuration
# ==============================================================================

CONFIG <- list(
  # Input files
  multi_scores_path = "outputs/scc_unbiased_study/multi_signature_scores.parquet",
  consensus_scores_path = "outputs/scc_unbiased_study/consensus_scc_scores.parquet",
  ne_scores_path = "outputs/nestor_scores.parquet",

  # Analysis parameters
  n_bins = 20,                    # Number of bins for z-progression analysis
  stage2_z_range = c(0.3, 0.6),  # Expected z_progression range for Stage 2 peak

  # Output
  output_dir = "outputs/scc_unbiased_study"
)

# Create output directory BEFORE setting up logging
dir.create(CONFIG$output_dir, recursive = TRUE, showWarnings = FALSE)

# Setup logging using shared utility
log_file <- file.path(CONFIG$output_dir, "transition_enrichment.log")
log_message <- create_logger(log_file)

log_message("============================================================")
log_message("SCC Transition Zone Enrichment Analysis")
log_message("Started: ", date())
log_message("============================================================")

# ==============================================================================
# Helper Functions (specific to this script)
# ==============================================================================

# Note: read_parquet_safe is now provided by shared/utils.R
# Fallback definition if shared utils not available
if (!exists("read_parquet_safe")) {
  read_parquet_safe <- function(path) {
    tryCatch({
      arrow::read_parquet(path, as_data_frame = TRUE)
    }, error = function(e) {
      log_message("Error reading parquet file '", path, "': ", e$message)
      stop("Failed to read parquet file '", path, "': ", e$message)
    })
  }
}

run_kruskal_wallis <- function(scores, stages) {
  test <- kruskal.test(scores ~ factor(stages))
  return(list(
    statistic = test$statistic,
    p_value = test$p.value,
    parameter = test$parameter
  ))
}

run_pairwise_wilcox <- function(scores, stages, comparisons) {
  results <- list()

  for (comp in comparisons) {
    group1_scores <- scores[stages == comp[1]]
    group2_scores <- scores[stages == comp[2]]

    test <- wilcox.test(group1_scores, group2_scores, alternative = "two.sided")

    results[[paste(comp[1], comp[2], sep = "_vs_")]] <- list(
      group1 = comp[1],
      group2 = comp[2],
      group1_mean = mean(group1_scores, na.rm = TRUE),
      group2_mean = mean(group2_scores, na.rm = TRUE),
      group1_median = median(group1_scores, na.rm = TRUE),
      group2_median = median(group2_scores, na.rm = TRUE),
      statistic = test$statistic,
      p_value = test$p.value
    )
  }

  return(results)
}

find_peak_bin <- function(scores, progression, n_bins = 20) {
  bin_edges <- seq(0, 1, length.out = n_bins + 1)
  bin_centers <- (bin_edges[-1] + bin_edges[-length(bin_edges)]) / 2
  bin_labels <- cut(progression, breaks = bin_edges, labels = FALSE, include.lowest = TRUE)

  bin_means <- tapply(scores, bin_labels, mean, na.rm = TRUE)

  peak_bin <- which.max(bin_means)
  peak_z <- bin_centers[peak_bin]
  peak_score <- bin_means[peak_bin]

  list(
    peak_bin = peak_bin,
    peak_z_progression = peak_z,
    peak_score = peak_score,
    bin_centers = bin_centers,
    bin_means = as.numeric(bin_means)
  )
}

# ==============================================================================
# Main Analysis
# ==============================================================================

main <- function() {
  # -------------------------------------------------------------------------
  # Step 1: Load data
  # -------------------------------------------------------------------------
  log_message("Step 1: Loading data...")

  # Load multi-signature scores
  multi_scores <- read_parquet_safe(CONFIG$multi_scores_path)
  consensus_scores <- read_parquet_safe(CONFIG$consensus_scores_path)

  # Load NESTOR scores
  ne_scores_raw <- read_parquet_safe(CONFIG$ne_scores_path)

  if ("__index_level_0__" %in% colnames(ne_scores_raw)) {
    sample_ids <- ne_scores_raw[["__index_level_0__"]]
    ne_scores_raw[["__index_level_0__"]] <- NULL
    ne_scores <- as.data.frame(ne_scores_raw)
    rownames(ne_scores) <- sample_ids
  } else if ("sample_id" %in% colnames(ne_scores_raw)) {
    sample_ids <- ne_scores_raw$sample_id
    ne_scores <- as.data.frame(ne_scores_raw)
    rownames(ne_scores) <- sample_ids
  } else {
    ne_scores <- as.data.frame(ne_scores_raw)
    rownames(ne_scores) <- as.character(seq_len(nrow(ne_scores)) - 1)
  }

  # Merge data (NESTOR scores have sample_id as a column, not row names)
  merged_df <- merge(multi_scores, ne_scores, by = "sample_id", all = FALSE)
  merged_df <- merge(merged_df, consensus_scores[, c("sample_id", "consensus_scc_zscore")],
                    by = "sample_id", all = FALSE)

  log_message(sprintf("Merged dataset: %d samples, %d columns", nrow(merged_df), ncol(merged_df)))

  # Define signatures to analyze
  signatures <- c("panc_scc", "krt_family", "tp63_targets", "consensus_scc_zscore")

  # -------------------------------------------------------------------------
  # Step 2: Stage-wise comparisons
  # -------------------------------------------------------------------------
  log_message("Step 2: Stage-wise comparisons...")

  stage_stats <- list()

  for (sig in signatures) {
    log_message(sprintf("  Analyzing %s...", sig))

    scores <- merged_df[[sig]]
    stages <- merged_df$stage_assignment

    # Overall Kruskal-Wallis test
    kw_result <- run_kruskal_wallis(scores, stages)

    # Pairwise Wilcoxon tests (Stage 2 vs 1, Stage 2 vs 3)
    comparisons <- list(c(1, 0), c(2, 0), c(2, 1), c(3, 2), c(3, 1))
    pairwise_results <- run_pairwise_wilcox(scores, stages, comparisons)

    # Store results
    stage_stats[[sig]] <- list(
      kruskal_wallis = kw_result,
      pairwise = pairwise_results,
      stage_means = tapply(scores, stages, mean, na.rm = TRUE),
      stage_medians = tapply(scores, stages, median, na.rm = TRUE),
      stage_sds = tapply(scores, stages, sd, na.rm = TRUE)
    )
  }

  # Create stage statistics table
  stage_table <- data.frame(
    signature = character(),
    stage_0_mean = numeric(),
    stage_1_mean = numeric(),
    stage_2_mean = numeric(),
    stage_3_mean = numeric(),
    kw_statistic = numeric(),
    kw_p_value = numeric(),
    stage2_vs_stage1_p = numeric(),
    stage2_vs_stage3_p = numeric(),
    stringsAsFactors = FALSE
  )

  for (sig in signatures) {
    stats <- stage_stats[[sig]]
    stage_table <- rbind(stage_table, data.frame(
      signature = sig,
      stage_0_mean = stats$stage_means["0"],
      stage_1_mean = stats$stage_means["1"],
      stage_2_mean = stats$stage_means["2"],
      stage_3_mean = stats$stage_means["3"],
      kw_statistic = stats$kruskal_wallis$statistic,
      kw_p_value = stats$kruskal_wallis$p_value,
      stage2_vs_stage1_p = stats$pairwise[["2_vs_1"]]$p_value,
      stage2_vs_stage3_p = stats$pairwise[["3_vs_2"]]$p_value
    ))
  }

  write.csv(stage_table, file.path(CONFIG$output_dir, "stage_statistics.csv"), row.names = FALSE)
  log_message("Saved stage statistics to: stage_statistics.csv")

  # -------------------------------------------------------------------------
  # Step 3: Peak location analysis
  # -------------------------------------------------------------------------
  log_message("Step 3: Peak location analysis...")

  peak_results <- data.frame(
    signature = character(),
    peak_bin = integer(),
    peak_z_progression = numeric(),
    peak_score = numeric(),
    in_transition_zone = logical(),
    stringsAsFactors = FALSE
  )

  for (sig in signatures) {
    log_message(sprintf("  Analyzing %s...", sig))

    scores <- merged_df[[sig]]
    progression <- merged_df$z_progression

    peak_info <- find_peak_bin(scores, progression, n_bins = CONFIG$n_bins)

    in_tz <- peak_info$peak_z_progression >= CONFIG$stage2_z_range[1] &
             peak_info$peak_z_progression <= CONFIG$stage2_z_range[2]

    peak_results <- rbind(peak_results, data.frame(
      signature = sig,
      peak_bin = peak_info$peak_bin,
      peak_z_progression = peak_info$peak_z_progression,
      peak_score = peak_info$peak_score,
      in_transition_zone = in_tz
    ))

    log_message(sprintf("    Peak at z_progression = %.3f (bin %d), in transition zone: %s",
                       peak_info$peak_z_progression, peak_info$peak_bin, in_tz))
  }

  write.csv(peak_results, file.path(CONFIG$output_dir, "peak_analysis.csv"), row.names = FALSE)
  log_message("Saved peak analysis to: peak_analysis.csv")

  # -------------------------------------------------------------------------
  # Step 4: Cancer type stratification
  # -------------------------------------------------------------------------
  log_message("Step 4: Cancer type stratification...")

  # Get top 10 cancer types by sample count
  cancer_counts <- sort(table(merged_df$cancer_type), decreasing = TRUE)
  top_cancers <- names(cancer_counts)[1:min(10, length(cancer_counts))]

  log_message(sprintf("Top 10 cancer types: %s", paste(top_cancers, collapse = ", ")))

  cancer_results <- list()

  for (cancer in top_cancers) {
    cancer_df <- merged_df[merged_df$cancer_type == cancer, ]

    if (nrow(cancer_df) < 50) {
      log_message(sprintf("  Skipping %s (n = %d < 50)", cancer, nrow(cancer_df)))
      next
    }

    log_message(sprintf("  Analyzing %s (n = %d)...", cancer, nrow(cancer_df)))

    cancer_peak <- find_peak_bin(cancer_df$consensus_scc_zscore,
                                 cancer_df$z_progression,
                                 n_bins = CONFIG$n_bins)

    # Stage-wise test for consensus score
    stages <- cancer_df$stage_assignment
    scores <- cancer_df$consensus_scc_zscore

    stage_means <- tapply(scores, stages, mean, na.rm = TRUE)

    cancer_results[[cancer]] <- list(
      n_samples = nrow(cancer_df),
      peak_z = cancer_peak$peak_z_progression,
      stage_0_mean = stage_means["0"],
      stage_1_mean = stage_means["1"],
      stage_2_mean = stage_means["2"],
      stage_3_mean = stage_means["3"]
    )
  }

  # Create cancer type results table
  cancer_table <- do.call(rbind, lapply(names(cancer_results), function(cancer) {
    data.frame(
      cancer_type = cancer,
      n_samples = cancer_results[[cancer]]$n_samples,
      peak_z_progression = cancer_results[[cancer]]$peak_z,
      stage_0_mean = cancer_results[[cancer]]$stage_0_mean,
      stage_1_mean = cancer_results[[cancer]]$stage_1_mean,
      stage_2_mean = cancer_results[[cancer]]$stage_2_mean,
      stage_3_mean = cancer_results[[cancer]]$stage_3_mean
    )
  }))

  write.csv(cancer_table, file.path(CONFIG$output_dir, "cancer_type_analysis.csv"), row.names = FALSE)
  log_message("Saved cancer type analysis to: cancer_type_analysis.csv")

  # -------------------------------------------------------------------------
  # Step 5: Create visualizations
  # -------------------------------------------------------------------------
  log_message("Step 5: Creating visualizations...")

  # Stage boxplot for consensus score
  p_boxplot <- ggplot(merged_df, aes(x = factor(stage_assignment), y = consensus_scc_zscore)) +
    geom_boxplot(aes(fill = factor(stage_assignment)), outlier.size = 0.5) +
    scale_fill_manual(values = colorRampPalette(c(amp_colors$diverging_endpoints$low, amp_colors$diverging_endpoints$mid, amp_colors$diverging_endpoints$high))(length(unique(merged_df$stage_assignment))), name = "Stage") +
    labs(
      title = "Consensus SCC Score by NE Progression Stage",
      x = "Stage",
      y = "Consensus SCC Score (Z-score)"
    ) +
    theme_bw(base_size = 12) +
    theme(legend.position = "none")

  ggsave(file.path(CONFIG$output_dir, "stage_boxplot.pdf"), p_boxplot,
         width = 8, height = 6, dpi = 300)
  save_publication_pdf(p_boxplot, "02_ne_progression/scc_trajectory/stage_boxplot.pdf", width = 8, height = 6)

  # Trajectory plot with peak highlighting
  bin_stats <- merged_df %>%
    mutate(bin = cut(z_progression, breaks = seq(0, 1, length.out = CONFIG$n_bins + 1),
                    labels = FALSE, include.lowest = TRUE)) %>%
    group_by(bin) %>%
    summarize(
      bin_center = mean(z_progression, na.rm = TRUE),
      n = n(),
      median = median(consensus_scc_zscore, na.rm = TRUE),
      q1 = quantile(consensus_scc_zscore, 0.25, na.rm = TRUE),
      q3 = quantile(consensus_scc_zscore, 0.75, na.rm = TRUE)
    )

  # Identify peak bin
  peak_bin <- bin_stats$bin[which.max(bin_stats$median)]
  peak_z <- bin_stats$bin_center[peak_bin]

  p_trajectory <- ggplot() +
    geom_ribbon(data = bin_stats, aes(x = bin_center, ymin = q1, ymax = q3),
                fill = amp_colors$correlation_colors$positive, alpha = 0.4) +
    geom_line(data = bin_stats, aes(x = bin_center, y = median),
              color = amp_colors$correlation_colors$positive, linewidth = 1.5) +
    geom_vline(xintercept = peak_z, color = amp_colors$log2fc_colors$increased, linetype = "dashed", linewidth = 1) +
    annotate("rect", xmin = CONFIG$stage2_z_range[1], xmax = CONFIG$stage2_z_range[2],
             ymin = -Inf, ymax = Inf, alpha = 0.1, fill = amp_colors$log2fc_colors$increased) +
    annotate("text", x = peak_z + 0.02, y = max(bin_stats$q3, na.rm = TRUE),
             label = sprintf("Peak: z = %.2f", peak_z), hjust = 0, color = amp_colors$log2fc_colors$increased) +
    labs(
      title = "Consensus SCC Score along NE Progression",
      subtitle = sprintf("Peak at z_progression = %.3f (Transition zone: 0.3-0.6)", peak_z),
      x = "NE Progression (z_progression)",
      y = "Consensus SCC Score (Median + IQR)"
    ) +
    theme_bw(base_size = 12) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2))

  ggsave(file.path(CONFIG$output_dir, "trajectory_with_peak.pdf"), p_trajectory,
         width = 10, height = 6, dpi = 300)
  save_publication_pdf(p_trajectory, "02_ne_progression/scc_trajectory/trajectory_with_peak.pdf", width = 10, height = 6)

  # -------------------------------------------------------------------------
  # Summary
  # -------------------------------------------------------------------------
  log_message("")
  log_message("============================================================")
  log_message("SUMMARY")
  log_message("============================================================")

  log_message("")
  log_message("Stage-wise statistics (Consensus SCC):")
  print(round(stage_stats[["consensus_scc_zscore"]]$stage_means, 3))

  log_message("")
  log_message("Peak locations:")
  print(peak_results[, c("signature", "peak_z_progression", "in_transition_zone")])

  log_message("")
  log_message("Key findings:")

  # Check if Stage 2 > Stage 1 and Stage 2 > Stage 3
  consensus_stats <- stage_stats[["consensus_scc_zscore"]]
  stage2_gt_stage1 <- consensus_stats$stage_means["2"] > consensus_stats$stage_means["1"]
  stage2_gt_stage3 <- consensus_stats$stage_means["2"] > consensus_stats$stage_means["3"]

  if (stage2_gt_stage1 && stage2_gt_stage3) {
    log_message("  STAGE 2 ENRICHMENT CONFIRMED: Stage 2 shows highest SCC activity")
  } else if (stage2_gt_stage1) {
    log_message("  PARTIAL ENRICHMENT: Stage 2 > Stage 1, but not > Stage 3")
  } else if (stage2_gt_stage3) {
    log_message("  PARTIAL ENRICHMENT: Stage 2 > Stage 3, but not > Stage 1")
  } else {
    log_message("  NO STAGE 2 ENRICHMENT: Stage 2 does not show peak SCC activity")
  }

  # Check peak location
  consensus_peak <- peak_results$peak_z_progression[peak_results$signature == "consensus_scc_zscore"]
  if (consensus_peak >= 0.3 && consensus_peak <= 0.6) {
    log_message(sprintf("  PEAK IN TRANSITION ZONE: Peak at z = %.3f (expected: 0.3-0.6)", consensus_peak))
  } else {
    log_message(sprintf("  PEAK OUTSIDE TRANSITION ZONE: Peak at z = %.3f (expected: 0.3-0.6)", consensus_peak))
  }

  log_message("")
  log_message("============================================================")
  log_message("ANALYSIS COMPLETE")
  log_message("Output directory: ", CONFIG$output_dir)
  log_message("============================================================")
}

# Run main
main()
