# Ensure packages installed
required <- c("readr","readxl","polycor","psych","simstudy","mokken","lordif","lavaan","semTools")
new.packages <- required[!(required %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos="https://cloud.r-project.org")

library(readr)
library(readxl)
library(polycor)
library(psych)
library(simstudy)
library(mokken)
library(lordif)
library(lavaan)
library(semTools)

# 1. Load data
load_survey <- function(path) {
  ext <- tools::file_ext(path)
  if (tolower(ext) == "csv") return(read_csv(path))
  if (tolower(ext) %in% c("xls","xlsx")) return(read_excel(path))
  stop("Unsupported file type")
}

# 2. Scale definitions (edit as needed)
scales <- list(
  TCR = paste0("TCR", 1:14),
  WE  = paste0("WE", 1:17),
  EMP = paste0("EMP", 1:19),
  ALT = paste0("ALT", 1:8)
)

# 3. Identify Likert columns automatically
auto_likert <- function(df, min_cat=4, max_cat=7) {
  nums <- sapply(df, function(x) is.integer(x) || (is.numeric(x) && all(x == floor(x), na.rm=TRUE)))
  candidates <- names(df)[nums]
  Filter(function(col) {
    nlev <- length(unique(na.omit(df[[col]])))
    nlev >= min_cat && nlev <= max_cat
  }, candidates)
}

# 4. Reverse-code items
reverse_items <- function(df, items, thresh=-0.2) {
  total <- rowSums(df[items], na.rm=TRUE)
  bad <- sapply(items, function(it) {
    cor(df[[it]], total - df[[it]], use="complete.obs", method="spearman") < thresh
  })
  names(bad)[bad]
}
apply_reverse <- function(df, rev_items, min=1, max=6) {
  for (it in rev_items) df[[it]] <- max + min - df[[it]]
  df
}

# 5. Mokken scaling
do_mokken <- function(df, items) {
  mok <- check.monotonicity(df[,items])
  H <- coefH(df[,items])
  list(monotonicity=mok, H=H)
}

# 6. Factor analysis & ω hierarchical
factor_weights <- function(df, items) {
  het <- hetcor(df[,items])$correlations
  fa1 <- fa(het, nfactors=1, fm="ml")
  load <- fa1$loadings[,1]
  weights <- load / sum(abs(load))
  omega_h <- psych::omega(df[,items], nfactors=1, plot=FALSE)$omegaHier
  list(weights=weights, omega_h=omega_h)
}

# 7. Simulate ordinal data
simulate_ordinal <- function(df, items, n_sim=200) {
  margs <- t(sapply(items, function(it) {
    tbl <- table(factor(df[[it]], levels=sort(unique(df[[it]]))))
    as.numeric(tbl / sum(tbl))
  }))
  colnames(margs) <- items
  corM <- hetcor(df[,items])$correlations
  def <- defDataAdd(varname="id", formula=1:nrow(df), dist="nonrandom")
  ds <- genData(nrow(df), def)
  genOrdCat(ds, baseprobs=margs, corMatrix=corM, n=n_sim)
}

# 8. DIF testing
do_dif <- function(df, items, group) {
  lapply(items, function(it) {
    lordif(df[[it]], df[[group]], criterion="Chisqr")
  })
}

# 9. SEM & measurement invariance
do_sem <- function(df, model, group=NULL) {
  if (is.null(group)) {
    lavaan::cfa(model, data=df, ordered=TRUE)
  } else {
    semTools::measurementInvariance(model=model, data=df, group=group, ordered=TRUE)
  }
}

# 10. Main pipeline
detailed_analysis <- function(path, group=NULL, n_sim=200) {
  df <- load_survey(path)
  likert_auto <- auto_likert(df)
  rev <- reverse_items(df, likert_auto)
  df <- apply_reverse(df, rev)

  results <- list()
  for (sc in names(scales)) {
    items <- scales[[sc]]
    mok <- do_mokken(df, items)
    fw  <- factor_weights(df, items)
    sim <- simulate_ordinal(df, items, n_sim)
    results[[sc]] <- list(mokken=mok, weights=fw$weights,
                          omega=fw$omega_h, sim=sim)
  }
  if (!is.null(group)) {
    results$dif <- do_dif(df, likert_auto, group)
  }
  results
}

# Utility to save outputs
save_results <- function(res) {
  for (sc in names(res)) {
    w <- res[[sc]]$weights
    write.csv(w, paste0("weights_", sc, ".csv"))
    sim <- res[[sc]]$sim
    write.csv(sim, paste0("simulated_", sc, ".csv"))
  }
}

# If run as script
args <- commandArgs(trailingOnly=TRUE)
path <- args[1]
group <- ifelse(length(args)>=2, args[2], NULL)
n_sim <- ifelse(length(args)>=3, as.integer(args[3]), 200)
res <- detailed_analysis(path, group, n_sim)
save_results(res)
