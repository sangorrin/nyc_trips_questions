# Question 4: Production ETL Monitoring & Data Quality

## Question
Imagine I use your script in a system that processes 100 files a day, I run them with CRON at 00:01 every night, I get the files from a S3 bucket, process them and push the result into a GCS bucket.

How would you monitor this system? Don't try to get too fancy, just explain how you'd do it using tools you have used in the past.

How would you ensure data quality? Same here, try to explain some tricks you used in any of the projects you worked on that may apply here.

## Answer

### 1. Monitoring Setup
**Core:** CRON script logs to CloudWatch Logs group `/aws/lambda/my-processor`. Alarm on errors; Lambda triggers on keywords for Slack/email.

**AWS Config Example:**
- Create Log Group: `aws logs create-log-group --log-group-name /aws/etl/processor`
- CloudWatch Alarm: Metric Filter for "ERROR|FAIL" → SNS Topic → Lambda.
```
aws logs put-metric-filter \
  --log-group-name /aws/etl/processor \
  --filter-name ErrorsFilter \
  --filter-pattern "ERROR OR FAIL" \
  --metric-transformations metricName=ErrorCount,metricValue=1
```
- Lambda Trigger: EventBridge rule at 00:01 cron; on alarm, invoke Lambda.
```python
# Lambda handler
import boto3
def lambda_handler(event, context):
    logs = boto3.client('logs').filter_log_events(logGroupName='/aws/etl/processor')
    if 'ERROR' in logs['events'][0]['message']:
        slack = boto3.client('sns').publish(TopicArn='arn:aws:sns:slack-alerts', Message=logs['events'][0]['message'])
```


**GCS Side:** Use `gsutil` metrics via Cloud Monitoring; alarm if no new blobs post-01:00.
```
gsutil ls -l gs://my-gcs-output/ | grep "$(date -d 'yesterday' +%Y%m%d)"
```

## 2. Alerting & Notification
**Triggers:** Lambda scans logs for "Exception|Failed|0 files processed" → Send Slack/email + summary (input/output counts).
**Daily Summary:** Script appends `echo "Processed: 100 files, Output: gs://bucket/20260218/" >> summary.log`; CloudWatch emails it.

**Advanced Tool:** AI CLI inspects logs → Opens GitHub issue, suggests fix, auto-PR/release.
- Use `aws logs filter-log-events --log-group /aws/etl/processor --filter-pattern "ERROR" --query 'events[].message'`
- Pipe to AI (e.g., local LLM): `| llm "Analyze error, suggest fix" → gh issue create --title "ETL Fail: [error]"` [reddit](https://www.reddit.com/r/devops/comments/j0xhnc/how_do_you_track_monitor_debug_cron_or_other/)

## 3. Debugging Tools
**CLI Debug Kit:**
- **Validate Structure:** Download + schema check.
  ```
  aws s3 cp s3://input/bad.parquet . && python -c "import pyarrow.parquet as pq; print(pq.read_schema('bad.parquet'))"
  gsutil cp gs://output/result.parquet . && parquet-tools row-count result.parquet  # Compare input/output rows
  ```
- **Find Matching Files:** List by condition (size, date).
  ```
  aws s3api list-objects-v2 --bucket input-bucket --query "Contents[?Size>10485760 && LastModified>'2026-02-17'].{Key:Key,Size:Size}"  # >10MB outliers
  gsutil ls -l gs://output-bucket/** | awk '$1 > 1000000 {print}'  # GCS large files
  ```
- **Test Injection:** Send known-good file, trigger re-run.
  ```
  aws s3 cp test-good.parquet s3://input/ && aws events put-events --entries '{"Source":"cron-trigger","Detail":"run now"}'
  ```
 [docs.aws.amazon](https://docs.aws.amazon.com/cli/latest/reference/s3/)

## 4. Data Quality Checks
**In-Script:** Row counts, schema match, % drop <5%.
```python
import pyarrow.parquet as pq
in_rows = pq.read_metadata('input.parquet').num_rows
out_rows = pq.read_metadata('output.parquet').num_rows
if out_rows / in_rows < 0.95: raise ValueError("Quality fail")
```
**Tests:** PR/CI with known files; nightly sample.
- `pytest test_etl.py --input test.parquet --expected output.json` [github](https://github.com/aws/aws-sdk-pandas/issues/1717)

## 5. Full Workflow Example
1. Alarm fires → Slack + log snippet.
2. Run CLI: Validate → List suspects.
3. Inject test → Re-run → AI analyzes new logs → GitHub issue/PR.
¡Simple, actionable, zero fancy tools! [aws.amazon](https://aws.amazon.com/blogs/big-data/monitor-data-pipelines-in-a-serverless-data-lake/)
