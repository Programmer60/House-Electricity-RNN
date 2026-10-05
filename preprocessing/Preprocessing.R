# LAB PROJECT: Preprocessing "Individual Household Electric
#
# Dataset source: UCI Machine Learning Repository
# Records: ~2,075,259 minute-level readings
# Period : 16/12/2006 to 26/11/2010
#
## 0. PACKAGES
 
# lubridate  -> easy Date/Time parsing
# dplyr      -> data wrangling
# zoo        -> na.approx() for time-series interpolation
# ggplot2    -> quick plots to *show* missingness/distribution
#
# install.packages(c("data.table","lubridate","dplyr","zoo","ggplot2"))

library(data.table)
library(lubridate)
library(dplyr)
library(zoo)
library(ggplot2)

set.seed(42)  # reproducibility

 
## 1. LOAD RAW DATA


raw <- fread(
  "C:/Users/mishr/Desktop/R/individual+household+electric+power+consumption/household_power_consumption.txt",
  sep       = ";",
  na.strings = "?",
  header    = TRUE,
  colClasses = list(character = c("Date", "Time"))
)

cat("Raw dimensions:", dim(raw)[1], "rows x", dim(raw)[2], "cols\n")
str(raw)
head(raw)


# 2. BUILD A PROPER DATETIME COLUMN
# Date is dd/mm/yyyy and Time is hh:mm:ss as separate text
# columns. An RNN needs a single, ordered time index, so we
# fuse them into one POSIXct column and sort by it (the file
# is already sorted, but we double check).

raw[, Datetime := dmy_hms(paste(Date, Time))]
raw <- raw[order(Datetime)]

range(raw$Datetime)   # sanity check: should span Dec 2006 - Nov 2010


## 3. HANDLE MISSING VALUES
# fread() already turned every "?" into NA. Let's quantify
# how much is missing and where, then fix it.

na_counts <- colSums(is.na(raw))
print(na_counts)

# The missingness happens in contiguous BLOCKS (e.g. a
# sensor/logger outage for a few hours), not randomly
# scattered points. That matters: for a genuinely random
# missing point, filling with the previous value is fine;
# for a block, linear interpolation across the gap is the
# more defensible choice for a continuous physical signal
# like power draw. So we use zoo::na.approx() column by
# column on the numeric measurement columns.

num_cols <- c("Global_active_power", "Global_reactive_power",
              "Voltage", "Global_intensity",
              "Sub_metering_1", "Sub_metering_2", "Sub_metering_3")

raw[, (num_cols) := lapply(.SD, function(x) na.approx(x, na.rm = FALSE)),
    .SDcols = num_cols]

# na.approx can't fill NAs that occur at the very start/end
# of the series (nothing to interpolate from), so mop up any
# leftovers with the nearest available value.
raw[, (num_cols) := lapply(.SD, function(x) na.locf(na.locf(x, na.rm = FALSE),
                                                    fromLast = TRUE)),
    .SDcols = num_cols]

cat("Remaining NAs after cleaning:", sum(is.na(raw[, ..num_cols])), "\n")


## 4. RESAMPLE TO HOURLY RESOLUTION
# 2+ million MINUTE-level rows is unnecessarily heavy for a
# classroom RNN demo and mostly just noise at that scale.
# We aggregate to HOURLY means. This:
#   - cuts the series to ~34,600 rows (2 million / 60)
#   - smooths minute-to-minute sensor noise
#   - keeps the daily/weekly seasonal patterns an RNN
#     is actually meant to learn

raw[, HourBlock := floor_date(Datetime, unit = "hour")]

hourly <- raw[, lapply(.SD, mean, na.rm = TRUE),
              by = HourBlock, .SDcols = num_cols]

setnames(hourly, "HourBlock", "Datetime")
cat("Hourly dimensions:", dim(hourly)[1], "rows x", dim(hourly)[2], "cols\n")
head(hourly)


## 5. QUICK VISUAL CHECKS


ggplot(hourly, aes(x = Datetime, y = Global_active_power)) +
  geom_line(linewidth = 0.2) +
  labs(title = "Hourly Global Active Power over time",
       x = "Date", y = "kWh") +
  theme_minimal()

ggplot(hourly, aes(x = Global_active_power)) +
  geom_histogram(bins = 60, fill = "steelblue") +
  labs(title = "Distribution of Global Active Power (hourly mean)") +
  theme_minimal()


 
## 6. CHOOSE THE TARGET / FEATURES
 
# Global_active_power (household's total active power, kW)
# is the standard forecasting target for this dataset.
# We keep it plus the other electrical measurements as
# additional input features (a "multivariate" RNN input),
# which generally improves forecasts over using the target
# alone.

feature_cols <- num_cols        # all 7 numeric measurements
target_col   <- "Global_active_power"

model_data <- as.data.frame(hourly[, c("Datetime", feature_cols), with = FALSE])


 
## 7. CHRONOLOGICAL TRAIN / TEST SPLIT
 
# IMPORTANT: for time series we NEVER shuffle before
# splitting. The test set must be the most RECENT slice of
# time, simulating "the future the model hasn't seen".

n         <- nrow(model_data)
train_frac <- 0.8
split_idx  <- floor(train_frac * n)

train_raw <- model_data[1:split_idx, ]
test_raw  <- model_data[(split_idx + 1):n, ]

cat("Train rows:", nrow(train_raw), " | Test rows:", nrow(test_raw), "\n")
cat("Train period:", as.character(range(train_raw$Datetime)), "\n")
cat("Test  period:", as.character(range(test_raw$Datetime)), "\n")


 
## 8. SCALE THE FEATURES (Min-Max Normalization)
 
# RNNs (especially with tanh/sigmoid gates, as in LSTM)
# train far better when inputs are on a similar, small scale
# (typically [0,1]). We compute the min/max ONLY on the
# training set and apply those same values to the test set -
# using test-set statistics would leak future information
# into training, which is a common and important mistake

mins <- sapply(train_raw[, feature_cols], min)
maxs <- sapply(train_raw[, feature_cols], max)

minmax_scale <- function(df, mins, maxs, cols) {
  scaled <- df
  for (col in cols) {
    scaled[[col]] <- (df[[col]] - mins[col]) / (maxs[col] - mins[col])
  }
  scaled
}

train_scaled <- minmax_scale(train_raw, mins, maxs, feature_cols)
test_scaled  <- minmax_scale(test_raw,  mins, maxs, feature_cols)

# Save the scaling parameters - you'll need the SAME mins/maxs
# to un-scale (invert) predictions back to real kWh units later.
saveRDS(list(mins = mins, maxs = maxs), "scaling_params.rds")


 
## 9. BUILD SLIDING-WINDOW SEQUENCES FOR THE RNN
 
# An RNN doesn't take a flat table - it takes SEQUENCES.
# We slide a window of `timesteps` past hours across the
# series; each window is one training example (X), and the
# value immediately after the window is what we ask the
# network to predict (y).
#
# Example with timesteps = 24 (i.e. use the last 24 hours to
# predict the next hour):
#   X[1] = hours 1..24,  y[1] = hour 25
#   X[2] = hours 2..25,  y[2] = hour 26
#   ... and so on.

create_sequences <- function(data, feature_cols, target_col, timesteps) {
  n_samples <- nrow(data) - timesteps
  n_features <- length(feature_cols)
  
  X <- array(0, dim = c(n_samples, timesteps, n_features))
  y <- numeric(n_samples)
  
  feat_matrix <- as.matrix(data[, feature_cols])
  
  for (i in 1:n_samples) {
    X[i, , ] <- feat_matrix[i:(i + timesteps - 1), ]
    y[i]     <- data[[target_col]][i + timesteps]
  }
  
  list(X = X, y = y)
}

timesteps <- 24   # use the past 24 hours to predict the next hour


train_seq <- create_sequences(train_scaled, feature_cols, target_col, timesteps)
test_seq  <- create_sequences(test_scaled,  feature_cols, target_col, timesteps)

cat("Train X shape: (", paste(dim(train_seq$X), collapse = ", "), ")\n")
cat("Train y length:", length(train_seq$y), "\n")
cat("Test  X shape: (", paste(dim(test_seq$X), collapse = ", "), ")\n")

# The resulting X shape is (samples, timesteps, features) -
# exactly the 3D input shape Keras/TensorFlow RNN layers
# (layer_simple_rnn / layer_lstm / layer_gru) require.


 
## 10. SAVE THE PREPROCESSED DATA

# Save everything the training script will need, 
# preprocessing and model-training can be two separate,

saveRDS(train_seq, "train_sequences.rds")
saveRDS(test_seq,  "test_sequences.rds")
saveRDS(list(mins = mins, maxs = maxs, timesteps = timesteps,
             feature_cols = feature_cols, target_col = target_col),
        "preprocessing_metadata.rds")

cat("\nPreprocessing complete.\n")
cat("Files written: train_sequences.rds, test_sequences.rds,\n")
cat("scaling_params.rds, preprocessing_metadata.rds\n")

