# Generate alternative sequence lengths without modifying the locked RDS files.
# The preprocessing and scaling conventions match Preprocessing.R.

library(data.table)
library(lubridate)
library(zoo)

raw <- fread(
  "individual+household+electric+power+consumption/household_power_consumption.txt",
  sep = ";",
  na.strings = "?",
  header = TRUE,
  colClasses = list(character = c("Date", "Time"))
)
raw[, Datetime := dmy_hms(paste(Date, Time))]
raw <- raw[order(Datetime)]

feature_cols <- c(
  "Global_active_power", "Global_reactive_power", "Voltage",
  "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
)
target_col <- "Global_active_power"
raw[, (feature_cols) := lapply(.SD, function(x) na.approx(x, na.rm = FALSE)),
    .SDcols = feature_cols]
raw[, (feature_cols) := lapply(.SD, function(x) na.locf(na.locf(x, na.rm = FALSE),
                                                          fromLast = TRUE)),
    .SDcols = feature_cols]
raw[, HourBlock := floor_date(Datetime, unit = "hour")]
hourly <- raw[, lapply(.SD, mean, na.rm = TRUE),
              by = HourBlock, .SDcols = feature_cols]
setnames(hourly, "HourBlock", "Datetime")

model_data <- as.data.frame(hourly[, c("Datetime", feature_cols), with = FALSE])
split_idx <- floor(0.8 * nrow(model_data))
train_raw <- model_data[1:split_idx, ]
test_raw <- model_data[(split_idx + 1):nrow(model_data), ]

# These are learned only from the original chronological training period,
# exactly as in the locked 24-timestep preprocessing pipeline.
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
test_scaled <- minmax_scale(test_raw, mins, maxs, feature_cols)

create_sequences <- function(data, feature_cols, target_col, timesteps) {
  n_samples <- nrow(data) - timesteps
  feat_matrix <- as.matrix(data[, feature_cols])
  X <- array(0, dim = c(n_samples, timesteps, length(feature_cols)))
  y <- numeric(n_samples)
  for (i in 1:n_samples) {
    X[i, , ] <- feat_matrix[i:(i + timesteps - 1), ]
    y[i] <- data[[target_col]][i + timesteps]
  }
  list(X = X, y = y)
}

for (timesteps in c(6, 12, 24, 48)) {
  # The validation stream is the chronological tail of the original training
  # sequence stream. Its input windows retain earlier training-period context.
  all_train <- create_sequences(train_scaled, feature_cols, target_col, timesteps)
  train_count <- floor(0.9 * length(all_train$y))
  train_seq <- list(
    X = all_train$X[1:train_count, , , drop = FALSE],
    y = all_train$y[1:train_count]
  )
  validation_seq <- list(
    X = all_train$X[(train_count + 1):length(all_train$y), , , drop = FALSE],
    y = all_train$y[(train_count + 1):length(all_train$y)]
  )

  # The test partition is generated independently, matching the locked
  # 24-hour baseline: no test target or future observation enters an input.
  test_seq <- create_sequences(test_scaled, feature_cols, target_col, timesteps)
  saveRDS(train_seq, paste0("preprocessing/train_sequences_T", timesteps, ".rds"))
  saveRDS(validation_seq, paste0("preprocessing/validation_sequences_T", timesteps, ".rds"))
  saveRDS(test_seq, paste0("preprocessing/test_sequences_T", timesteps, ".rds"))
  cat(
    "T", timesteps,
    "train", length(train_seq$y),
    "validation", length(validation_seq$y),
    "test", length(test_seq$y), "\n"
  )
}

cat("Alternative sequence generation complete.\n")
