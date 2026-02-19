# NYC Taxi Data - Questions & Answers Summary

## Question 1: PyArrow vs DuckDB Performance

**Why do you think PyArrow is giving you better performance than DuckDB? What are the main factors?**

PyArrow achieves 6.8x faster performance primarily through zero-copy operations. Since Parquet's native in-memory format is Arrow, PyArrow can read, filter, and write data without format conversions. The outlier detection task is simple (calculate percentile, then filter), which doesn't benefit from DuckDB's sophisticated SQL query engine—instead, that engine adds overhead through query parsing, planning, and type conversions. PyArrow's direct compute functions (`pc.quantile()`, `pc.filter()`) operate on contiguous Arrow arrays with SIMD vectorization, minimizing memory allocations and maximizing cache efficiency.

**[Complete Answer →](ANSWER1.md)**

---

## Question 2: Parquet File Optimizations

**If you could modify the Parquet files to speed up your solution (containing the same data), what would you change? Why?**

The key optimization is physically sorting records by trip distance in descending order and structuring files with exactly 10 equal-sized row groups. This transforms the problem—instead of scanning 100% of data to calculate the 90th percentile, you simply read row group 0, which is guaranteed to contain the top 10% of trips. Additional optimizations include normalizing column names across different file versions, converting string datetimes to proper timestamp types, and using Snappy compression with dictionary encoding. This approach achieved 11.1x average speedup and 47.6x memory reduction.

**[Complete Answer →](ANSWER2.md)**

---

## Question 3: Processing 30MB Parquet in 20ms from HTTP Server

**Imagine we need to process one of the Parquet files (assume 30 MB) in 20 ms. The file is stored on an HTTP server, and you need to complete all work in 20 ms (download, process, and output locally). How would you do it? What are the main bottlenecks you see?**

The solution uses smart partitioning (pre-sorted into 10 row groups, download only the first 3MB containing the top 10% by distance), connection pooling with keep-alive, co-located compute (Fly.io) and storage (Tigris S3), and PyArrow vectorized processing. While all optimizations were successfully implemented and processing achieves 3.5ms, the actual results showed network I/O as the immovable bottleneck: downloads took 120ms cold and 84.93ms with CDN caching—far exceeding the theoretical 12ms. The 20ms target was not achieved (~90ms best case), revealing that platform networking limitations can prevent sub-20ms operations requiring external storage.

**[Complete Answer →](ANSWER3.md)**

---

## Question 4: Production ETL Monitoring & Data Quality

**Imagine I use your script in a system that processes 100 files a day, I run them with CRON at 00:01 every night, I get the files from a S3 bucket, process them and push the result into a GCS bucket. How would you monitor this system? Don't try to get too fancy, just explain how you'd do it using tools you have used in the past. How would you ensure data quality? Same here, try to explain some tricks you used in any of the projects you worked on that may apply here.**

Use CloudWatch Logs for centralized logging with metric filters that trigger alarms on error keywords (`ERROR`, `FAIL`, `Exception`). Set up Lambda functions or SNS topics to send Slack/email alerts with log snippets. For debugging, create CLI workflows using AWS/GCS CLI tools to validate schemas, compare row counts, list suspicious files by size/date, and inject test files for re-runs. Ensure data quality with in-script checks (row count comparisons, schema validation, drop rate <5%) and automated CI/CD tests using known-good sample files.

**[Complete Answer →](ANSWER4.md)**
