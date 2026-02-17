1. Why do you think PyArrow is giving you better performance than DuckDB? What are the main factors?
2. If you could modify the Parquet files to speed up your solution (containing the same data), what would you change? Why?
3. Imagine we need to process one of the Parquet files (assume 30 MB) in 20 ms. The file is stored on an HTTP server, and you need to complete all work in 20 ms (download, process, and output locally). How would you do it? What are the main bottlenecks you see?
4. Imagine I use your script in a system that processes 100 files a day, I run them with CRON at 00:01 every night, I get the files from a S3 bucket, process them and push the result into a GCS bucket.
How would you monitor this system? Don't try to get too fancy, just explain how you'd do it using tools you have used in the past.
How would you ensure data quality? Same here, try to explain some tricks you used in any of the projects you worked on that may apply here.
