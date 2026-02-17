# NYC Taxi Outlier Detection

Performance comparison of different implementations for detecting outlier taxi trips using physics-based validation.

## Overview

This project implements outlier detection for NYC Yellow Taxi trip data using three approaches:
- **PyArrow**: Vectorized columnar operations with zero-copy filtering
- **PyArrow Optimized**: Leverages pre-sorted parquet files to read only top 10% (row group 0)
- **DuckDB**: SQL-based analytical queries with single-pass detection

All implementations identify trips violating physics-based constraints (invalid distance, speed, or duration) within the top percentile of trip distances.

## Installation

```bash
# Install in editable mode with dependencies
pip install -e .
```

## Usage

### Command Line

```bash
# PyArrow implementation
find-outliers-pyarrow parquets/yellow_tripdata_2023-01.parquet

# PyArrow Optimized (requires optimized parquet files)
find-outliers-pyarrow-optimized parquets_optimized/yellow_tripdata_2023-01.parquet

# DuckDB implementation
find-outliers-duckdb parquets/yellow_tripdata_2023-01.parquet

# Custom output format
find-outliers-pyarrow input.parquet --output-format csv --output results.csv
```

### Python API

```python
from detectors import detect_outliers_pyarrow, detect_outliers_pyarrow_optimized, detect_outliers_duckdb, DEFAULT_CONFIG

# Run detection with original PyArrow
result = detect_outliers_pyarrow('data.parquet', DEFAULT_CONFIG)
print(f"Found {result.outlier_count} outliers in {result.processing_time:.2f}s")

# Run detection with optimized PyArrow (on pre-sorted files)
result = detect_outliers_pyarrow_optimized('data_optimized.parquet', DEFAULT_CONFIG)
print(f"Found {result.outlier_count} outliers in {result.processing_time:.2f}s")

# Access results
outliers_table = result.outliers_table  # PyArrow table
stats = result.stats  # Detection statistics
```

### Benchmarking

#### Comparing PyArrow vs DuckDB

Compare the two implementation approaches (PyArrow's columnar operations vs DuckDB's SQL engine):

```bash
# Install with benchmark tools
pip install -e ".[benchmark]"

# Step 1: Benchmark both implementations on sample files
python scripts/benchmark.py \
  --parquets-path parquets \
  --samples 20 \
  --output-path scripts/benchmark/ \
  --implementations pyarrow duckdb

# Step 2: View results
open scripts/benchmark/benchmark_report.html
```

Generates in `scripts/benchmark/`:
- `results.json` - Raw benchmark data
- `benchmark_report.html` - Interactive report with plots and AI analysis
- `*.png` - Performance comparison plots

#### Comparing PyArrow vs PyArrow Optimized

Compare the original PyArrow implementation against the optimized version that leverages pre-sorted parquet files:

```bash
# Step 1: Optimize parquet files (sort by distance, create 10 row groups)
python scripts/optimize_parquets.py \
  --samples 50 \
  --parquets-path parquets \
  --output-path parquets_optimized

# Step 2: Benchmark both PyArrow versions on optimized files
python scripts/benchmark.py \
  --parquets-path parquets_optimized \
  --samples 50 \
  --output-path scripts/benchmark_optimized/ \
  --implementations pyarrow pyarrow_optimized

# Step 3: View results
open scripts/benchmark_optimized/benchmark_report.html
```

The optimized version reads only the first row group (top 10% by distance), eliminating percentile calculation and reducing I/O by ~90%.

### Profiling

```bash
# Install profiling dependencies
pip install -e ".[profiling]"

# Profile both implementations to identify bottlenecks
python scripts/profiling.py --samples 10

# With LLM analysis (requires ANTHROPIC_API_KEY and LLM_MODEL env vars)
export LLM_MODEL=claude-3-5-sonnet-20241022
export LLM_API_KEY=your-api-key
python scripts/profiling.py --samples 5

# View results
open scripts/profiling/profiling_report.html
```

Generates:
- `pyarrow_profile.json` / `duckdb_profile.json` - Function-level timing statistics
- `profiling_report.html` - Interactive report with top bottlenecks and AI analysis

## Outlier Detection Logic

**Phase 1**: Filter to top percentile by trip distance (default: 90th percentile)

**Phase 2**: Flag outliers violating any constraint:
- Distance: < 0.1 or > 800 miles
- Duration: ≤ 0 or > 10 hours
- Speed: < 2.5 or > 80 mph


