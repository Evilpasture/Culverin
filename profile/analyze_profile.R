# Culverin Physics Benchmark Analysis
# Requirements: install.packages(c("data.table", "ggplot2", "scales", "zoo"))

library(data.table)
library(ggplot2)
library(scales)
library(zoo)

# 1. LOAD DATA
csv_file <- "culverin_stepping_benchmark.csv"

if (!file.exists(csv_file)) {
  stop("CSV file not found. Run the Python benchmark script first!")
}

print(paste("Reading", csv_file, "..."))
# fread is significantly faster than read.csv for 1M+ rows
df <- fread(csv_file)

# 2. DATA TRANSFORMATION
# Convert nanoseconds to milliseconds for easier reading
df[, duration_ms := duration_ns / 1e6]

# Calculate a rolling average (window of 1000 steps) to see trends through jitter
print("Calculating moving averages...")
df[, roll_avg_ms := rollmean(duration_ms, k = 1000, fill = NA)]

# 3. DESCRIPTIVE STATISTICS
stats <- list(
  mean   = mean(df$duration_ms),
  median = median(df$duration_ms),
  p95    = quantile(df$duration_ms, 0.95),
  p99    = quantile(df$duration_ms, 0.99),
  max    = max(df$duration_ms),
  total_s = sum(df$duration_ns) / 1e9
)

cat("\n--- PERFORMANCE SUMMARY ---\n")
cat(sprintf("Average Step:     %.4f ms\n", stats$mean))
cat(sprintf("Median Step:      %.4f ms\n", stats$median))
cat(sprintf("95th Percentile:  %.4f ms (Tail Latency)\n", stats$p95))
cat(sprintf("99th Percentile:  %.4f ms\n", stats$p99))
cat(sprintf("Worst Case:       %.4f ms\n", stats$max))
cat(sprintf("Throughput:       %.0f steps/sec\n", nrow(df) / stats$total_s))
cat("---------------------------\n")

# 4. VISUALIZATION

print("Generating plots...")

# Plot A: Step Latency over time
plot_latency <- ggplot(df[seq(1, .N, by=100)], aes(x = step_index)) +
  geom_line(aes(y = duration_ms), color = "steelblue", alpha = 0.2) +
  geom_line(aes(y = roll_avg_ms), color = "firebrick", size = 1) +
  theme_minimal() +
  labs(title = "Culverin Stepping Latency",
       subtitle = "Blue: Raw Samples (sampled 1/100) | Red: 1k Step Moving Average",
       x = "Step Index",
       y = "Duration (ms)") +
  scale_y_continuous(labels = comma)

# Plot B: Smooth Latency Distribution
plot_dist <- ggplot(df, aes(x = duration_ms)) +
  geom_density(fill = "seagreen", color = "darkgreen", alpha = 0.7) +
  theme_minimal() +
  # Zoom tightly on the "meat" of the data
  coord_cartesian(xlim = c(stats$median * 0.8, stats$median * 1.2)) + 
  labs(title = "Refined Latency Distribution",
       subtitle = "Kernel Density Estimate (KDE) - High Resolution View",
       x = "Duration (ms)",
       y = "Density")

# 5. SAVE RESULTS
ggsave("culverin_latency_trend.png", plot_latency, width = 10, height = 6)
ggsave("culverin_distribution.png", plot_dist, width = 10, height = 6)

print("Analysis complete. Plots saved as 'culverin_latency_trend.png' and 'culverin_distribution.png'.")