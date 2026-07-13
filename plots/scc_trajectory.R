#!/usr/bin/env Rscript
#
# scc_trajectory.R
#
# Squamous Cell Carcinoma (SCC) signature activity analysis along NE progression.
# Uses GSVA for signature scoring and visualizes changes along the NE progression axis.
#
# Based on:
#   - TET2_analysis.R methodology using UCell/multiScore
#   - 29_pathway_trajectory_analysis.R trajectory visualization
#
# Input:
#   - Z-scored expression data from AnnData
#   - NE_progression scores from tumor samples
#   - PanC_SCC gene signature (475 genes overlapping with dataset)
#
# Output:
#   - scc_scores.parquet: GSVA scores per sample
#   - scc_trajectory.pdf: Main trajectory visualization
#   - scc_trajectory_binned.pdf: Binned statistics visualization
#
# Usage:
#   Rscript scripts/scc_trajectory.R

# ==============================================================================
# Load required packages
# ==============================================================================

required_packages <- c("GSVA", "GSEABase", "mgcv", "ggplot2", "scales", "zoo",
                       "arrow", "patchwork", "dplyr")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (pkg %in% c("GSVA", "GSEABase")) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
      }
      BiocManager::install(pkg)
    } else {
      install.packages(pkg)
    }
  }
}

suppressPackageStartupMessages({
  library(GSVA)
  library(GSEABase)
  library(mgcv)
  library(ggplot2)
  library(scales)
  library(zoo)
  library(arrow)
  library(patchwork)
  library(dplyr)
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
  anndata_path = "data/processed/cbio_pan_cancer_ne_final_hgnc.h5ad",
  ne_scores_path = "outputs/nestor_scores_tumor_only.parquet",
  scc_genes_path = "config/panc_scc_genes.txt",

  # Analysis parameters
  n_bins = 50,                    # Number of bins for trend analysis
  gam_k = 10,                     # GAM spline basis dimension

  # Output
  output_dir = "outputs/scc_analysis",
  figures_dir = "outputs/scc_analysis/figures"
)

# Create output directories
dir.create(CONFIG$output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(CONFIG$figures_dir, recursive = TRUE, showWarnings = FALSE)

# Setup logging using shared utility
log_file <- file.path(CONFIG$output_dir, "scc_analysis.log")
log_message <- create_logger(log_file)

log_message("============================================================")
log_message("SCC Signature Trajectory Analysis along NE Progression")
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

load_expression_matrix <- function(anndata_path, ne_scores_path) {
  log_message("Loading expression data...")

  expr_parquet <- "data/processed/expression_matrix.parquet"
  var_parquet <- "data/processed/var.parquet"

  if (!file.exists(expr_parquet) || !file.exists(var_parquet)) {
    log_message("Exporting from AnnData (this may take a few minutes)...")

    python_script <- sprintf(
'import anndata
import pandas as pd
import numpy as np

adata = anndata.read_h5ad("%s")
ne_scores = pd.read_parquet("%s")

# Filter to common tumor samples
common = [s for s in ne_scores.index if s in adata.obs.index]
adata = adata[common]

# Export expression (genes as rows, samples as columns)
X = adata.X
if hasattr(X, "toarray"):
    X = X.toarray()

pd.DataFrame(X.T, index=adata.var_names, columns=adata.obs.index).to_parquet("%s")
adata.var.to_parquet("%s")
print(f"Exported {len(common)} samples, {adata.n_vars} genes")
',
      anndata_path, ne_scores_path, expr_parquet, var_parquet
    )

    temp_script <- tempfile(fileext = ".py")
    writeLines(python_script, temp_script)
    on.exit(unlink(temp_script))

    result <- system2("python3", args = temp_script, stdout = TRUE, stderr = TRUE)
    log_message(paste(result, collapse = "\n"))
  }

  expr <- as.matrix(arrow::read_parquet(expr_parquet, as_data_frame = TRUE))
  var_df <- arrow::read_parquet(var_parquet, as_data_frame = TRUE)
  rownames(expr) <- var_df$hgnc_symbol

  log_message(sprintf("Expression matrix: %d genes x %d samples", nrow(expr), ncol(expr)))
  return(expr)
}

load_scc_genes <- function(genes_path) {
  genes <- readLines(genes_path)
  genes <- trimws(genes)
  genes <- genes[genes != ""]
  log_message(sprintf("Loaded %d SCC signature genes from %s", length(genes), genes_path))
  return(genes)
}

run_gsva_single <- function(expr_matrix, signature_genes, signature_name = "PanC_SCC") {
  log_message("Running GSVA for SCC signature...")

  mode(expr_matrix) <- "numeric"

  # Handle NA values
  na_count <- sum(is.na(expr_matrix))
  if (na_count > 0) {
    na_frac <- na_count / (nrow(expr_matrix) * ncol(expr_matrix))
    log_message(sprintf("Expression matrix has %d NA values (%.2f%%), imputing with row means", na_count, na_frac * 100))

    global_mean <- mean(expr_matrix, na.rm = TRUE)
    row_means <- rowMeans(expr_matrix, na.rm = TRUE)
    row_means[is.na(row_means)] <- global_mean

    na_mask <- is.na(expr_matrix)
    expr_matrix[na_mask] <- row_means[row(expr_matrix)[na_mask]]
  }

  # Create gene set
  gene_set <- GeneSet(setName = signature_name, geneIds = signature_genes)
  gene_set_collection <- GeneSetCollection(gene_set)

  # Run GSVA
  gsva_par <- gsvaParam(
    exprData = expr_matrix,
    geneSets = gene_set_collection,
    kcdf = "Gaussian",
    minSize = 10,
    maxSize = Inf
  )

  gsva_scores <- gsva(gsva_par)

  # Extract scores as vector
  if (is.matrix(gsva_scores)) {
    scores <- as.numeric(gsva_scores[1, ])
    names(scores) <- colnames(gsva_scores)
  } else {
    scores <- as.numeric(gsva_scores)
    names(scores) <- colnames(expr_matrix)
  }

  log_message(sprintf("GSVA complete: %d samples, score range [%.3f, %.3f]",
                      length(scores), min(scores, na.rm = TRUE), max(scores, na.rm = TRUE)))

  return(scores)
}

fit_trajectory_gam <- function(scores, progression, k = 10) {
  df <- data.frame(score = scores, progression = progression)

  gam_fit <- gam(score ~ s(progression, k = k), data = df, method = "REML")

  pred_grid <- data.frame(progression = seq(0, 1, length.out = 200))
  predictions <- predict(gam_fit, newdata = pred_grid, se.fit = TRUE)
  predictions$progression <- pred_grid$progression

  # Compute derivative
  n_pred <- length(predictions$fit)
  h <- diff(predictions$progression)[1]
  deriv <- numeric(n_pred)
  deriv[1] <- (predictions$fit[2] - predictions$fit[1]) / h
  if (n_pred > 2) {
    deriv[2:(n_pred-1)] <- (predictions$fit[3:n_pred] - predictions$fit[1:(n_pred-2)]) / (2 * h)
  }
  deriv[n_pred] <- (predictions$fit[n_pred] - predictions$fit[n_pred-1]) / h
  predictions$derivative <- deriv

  gam_summary <- summary(gam_fit)
  r_squared <- gam_summary$r.sq
  edf <- gam_summary$edf[1]
  p_value <- gam_summary$s.pv

  list(
    gam_fit = gam_fit,
    predictions = predictions,
    edf = edf,
    r_squared = r_squared,
    p_value = p_value
  )
}

compute_binned_stats <- function(scores, progression, n_bins = 50) {
  bin_edges <- seq(0, 1, length.out = n_bins + 1)
  bin_centers <- (bin_edges[-1] + bin_edges[-length(bin_edges)]) / 2
  bin_labels <- cut(progression, breaks = bin_edges, labels = FALSE, include.lowest = TRUE)

  bin_stats <- data.frame(
    bin_center = bin_centers,
    n = as.numeric(table(bin_labels)),
    median = tapply(scores, bin_labels, median, na.rm = TRUE),
    mean = tapply(scores, bin_labels, mean, na.rm = TRUE),
    q1 = tapply(scores, bin_labels, quantile, probs = 0.25, na.rm = TRUE),
    q3 = tapply(scores, bin_labels, quantile, probs = 0.75, na.rm = TRUE),
    sd = tapply(scores, bin_labels, sd, na.rm = TRUE)
  )

  bin_stats$iqr <- bin_stats$q3 - bin_stats$q1
  bin_stats
}

# ==============================================================================
# Main Analysis
# ==============================================================================

main <- function() {
  # -------------------------------------------------------------------------
  # Step 1: Load data
  # -------------------------------------------------------------------------
  log_message("Step 1: Loading data...")

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

  log_message(sprintf("Loaded NE scores: %d samples", nrow(ne_scores)))

  expr_matrix <- load_expression_matrix(CONFIG$anndata_path, CONFIG$ne_scores_path)
  scc_genes <- load_scc_genes(CONFIG$scc_genes_path)

  # Filter SCC genes to those in expression matrix
  scc_genes_overlap <- intersect(scc_genes, rownames(expr_matrix))
  log_message(sprintf("SCC genes in expression matrix: %d / %d",
                      length(scc_genes_overlap), length(scc_genes)))

  # Align samples
  common_samples <- intersect(colnames(expr_matrix), rownames(ne_scores))
  expr_matrix <- expr_matrix[, common_samples]
  ne_scores <- ne_scores[common_samples, , drop = FALSE]
  progression <- ne_scores$NE_progression

  log_message(sprintf("Aligned: %d samples", length(common_samples)))

  # -------------------------------------------------------------------------
  # Step 2: Run GSVA for SCC signature
  # -------------------------------------------------------------------------
  scc_scores_file <- file.path(CONFIG$output_dir, "scc_scores.parquet")

  if (file.exists(scc_scores_file)) {
    log_message("Step 2: Loading existing SCC scores from: ", scc_scores_file)
    scc_df <- arrow::read_parquet(scc_scores_file, as_data_frame = TRUE)
    scc_scores <- setNames(scc_df$scc_score, scc_df$sample_id)
  } else {
    log_message("Step 2: Running GSVA for SCC signature...")
    scc_scores <- run_gsva_single(expr_matrix, scc_genes_overlap, "PanC_SCC")

    scc_df <- data.frame(
      sample_id = names(scc_scores),
      scc_score = as.numeric(scc_scores)
    )
    arrow::write_parquet(scc_df, scc_scores_file)
    log_message("Saved SCC scores to: ", scc_scores_file)
  }

  # -------------------------------------------------------------------------
  # Step 3: Calculate statistics and fit GAM
  # -------------------------------------------------------------------------
  log_message("Step 3: Calculating statistics and fitting GAM...")

  # Spearman correlation
  valid_idx <- !is.na(scc_scores) & !is.na(progression)
  cor_test <- cor.test(scc_scores[valid_idx], progression[valid_idx], method = "spearman")
  rho <- cor_test$estimate
  cor_pvalue <- cor_test$p.value

  log_message(sprintf("Spearman correlation: rho = %.4f, p = %.2e", rho, cor_pvalue))

  # Fit GAM
  gam_result <- fit_trajectory_gam(scc_scores[valid_idx], progression[valid_idx], k = CONFIG$gam_k)
  log_message(sprintf("GAM fit: R2 = %.4f, EDF = %.2f, p = %.2e",
                      gam_result$r_squared, gam_result$edf, gam_result$p_value))

  # Compute binned statistics
  bin_stats <- compute_binned_stats(scc_scores[valid_idx], progression[valid_idx], n_bins = CONFIG$n_bins)

  # -------------------------------------------------------------------------
  # Step 4: Create visualizations
  # -------------------------------------------------------------------------
  log_message("Step 4: Creating visualizations...")

  # Prepare plot data
  plot_df <- data.frame(progression = progression, score = scc_scores)

  # Subsample for visibility
  if (nrow(plot_df) > 5000) {
    set.seed(42)
    plot_df <- plot_df[sample(nrow(plot_df), 5000), ]
  }

  pred_df <- as.data.frame(gam_result$predictions)

  # Compute y-axis limits
  score_range <- range(scc_scores, na.rm = TRUE)
  score_pad <- diff(score_range) * 0.15
  y_min <- floor((score_range[1] - score_pad) * 10) / 10
  y_max <- ceiling((score_range[2] + score_pad) * 10) / 10

  # Main trajectory plot
  p1 <- ggplot() +
    geom_point(data = plot_df, aes(x = progression, y = score),
               alpha = 0.12, size = 0.6, color = amp_colors$correlation_colors$neutral) +
    geom_ribbon(data = bin_stats, aes(x = bin_center, ymin = q1, ymax = q3),
                fill = amp_colors$correlation_colors$positive, alpha = 0.35) +
    geom_line(data = bin_stats, aes(x = bin_center, y = median),
              color = amp_colors$correlation_colors$positive, linewidth = 1.8) +
    geom_line(data = pred_df, aes(x = progression, y = fit),
              color = amp_colors$ne_category_colors$`True NE`, linewidth = 1.2, linetype = "dashed") +
    labs(
      title = "Pan-Cancer SCC Signature Activity along NE Progression",
      subtitle = sprintf("Spearman rho = %.3f, p = %.2e | GAM R2 = %.3f | n = %d samples | %d genes",
                         rho, cor_pvalue, gam_result$r_squared, sum(valid_idx), length(scc_genes_overlap)),
      x = "NE Progression",
      y = "SCC Signature Score (GSVA)"
    ) +
    theme_bw(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10, color = "gray40")
    ) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    coord_cartesian(ylim = c(y_min, y_max)) +
    annotate("text", x = 0.02, y = y_max - 0.02, hjust = 0, vjust = 1,
             label = "Blue: Median (IQR) | Red dashed: GAM fit",
             size = 2.8, color = "gray30")

  # Derivative plot
  max_abs_deriv <- max(abs(pred_df$derivative), na.rm = TRUE)
  if (max_abs_deriv > 0) {
    pred_df$deriv_norm <- pred_df$derivative / max_abs_deriv
  } else {
    pred_df$deriv_norm <- pred_df$derivative
  }

  p2 <- ggplot() +
    geom_line(data = pred_df, aes(x = progression, y = deriv_norm),
              color = amp_colors$ne_category_colors$`True NE`, linewidth = 1) +
    geom_hline(yintercept = 0, color = amp_colors$correlation_colors$neutral, linewidth = 0.3) +
    labs(
      title = "Rate of Change",
      x = "NE Progression",
      y = "Derivative (normalized)"
    ) +
    theme_bw(base_size = 12) +
    coord_cartesian(ylim = c(-1, 1), xlim = c(0, 1))

  # Combined plot
  combined <- (p1 / p2) +
    plot_layout(heights = c(2, 1))

  output_path <- file.path(CONFIG$figures_dir, "scc_trajectory.pdf")
  ggsave(output_path, combined, width = 10, height = 7, dpi = 300)
  save_publication_pdf(combined, "02_ne_progression/scc_trajectory/scc_trajectory.pdf", width = 10, height = 7)
  log_message("Saved main plot: ", output_path)

  # Binned statistics heatmap-style plot
  p3 <- ggplot(bin_stats, aes(x = bin_center, y = median)) +
    geom_ribbon(aes(ymin = q1, ymax = q3), fill = amp_colors$correlation_colors$positive, alpha = 0.4) +
    geom_line(color = amp_colors$correlation_colors$positive, linewidth = 1.5) +
    geom_point(aes(size = n), alpha = 0.6, color = amp_colors$correlation_colors$positive) +
    scale_size_continuous(range = c(1, 4), name = "Sample count") +
    labs(
      title = "SCC Signature: Binned Statistics",
      x = "NE Progression",
      y = "SCC Score (Median)"
    ) +
    theme_bw(base_size = 11)

  output_path_binned <- file.path(CONFIG$figures_dir, "scc_binned_stats.pdf")
  ggsave(output_path_binned, p3, width = 8, height = 5, dpi = 300)
  save_publication_pdf(p3, "02_ne_progression/scc_trajectory/scc_binned_stats.pdf", width = 8, height = 5)
  log_message("Saved binned plot: ", output_path_binned)

  # -------------------------------------------------------------------------
  # Step 5: Summary statistics
  # -------------------------------------------------------------------------
  log_message("")
  log_message("============================================================")
  log_message("SUMMARY STATISTICS")
  log_message("============================================================")
  log_message(sprintf("Total samples: %d", sum(valid_idx)))
  log_message(sprintf("SCC genes used: %d", length(scc_genes_overlap)))
  log_message(sprintf("Score range: [%.3f, %.3f]", min(scc_scores, na.rm = TRUE), max(scc_scores, na.rm = TRUE)))
  log_message(sprintf("Score mean: %.3f", mean(scc_scores, na.rm = TRUE)))
  log_message(sprintf("Score median: %.3f", median(scc_scores, na.rm = TRUE)))
  log_message("")
  log_message(sprintf("Correlation with NE progression:"))
  log_message(sprintf("  Spearman rho: %.4f", rho))
  log_message(sprintf("  p-value: %.2e", cor_pvalue))
  log_message("")
  log_message(sprintf("GAM trajectory fit:"))
  log_message(sprintf("  R-squared: %.4f", gam_result$r_squared))
  log_message(sprintf("  Effective degrees of freedom: %.2f", gam_result$edf))
  log_message(sprintf("  p-value: %.2e", gam_result$p_value))

  # Trend interpretation
  if (rho < -0.3) {
    trend_interpretation <- "DECREASING: SCC signature declines with NE progression"
  } else if (rho > 0.3) {
    trend_interpretation <- "INCREASING: SCC signature rises with NE progression"
  } else {
    trend_interpretation <- "WEAK/NO TREND: SCC signature shows no clear monotonic relationship with NE progression"
  }
  log_message("")
  log_message(sprintf("Trend interpretation: %s", trend_interpretation))

  log_message("")
  log_message("============================================================")
  log_message("ANALYSIS COMPLETE")
  log_message("Output directory: ", CONFIG$output_dir)
  log_message("Figures: ", CONFIG$figures_dir, "/*.pdf")
  log_message("============================================================")
}

# Run main
main()
