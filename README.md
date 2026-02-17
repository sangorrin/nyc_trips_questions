# NYC Taxi Outlier Detection

Performance comparison of PyArrow vs DuckDB for detecting outlier taxi trips using physics-based validation.

## Overview

This project implements outlier detection for NYC Yellow Taxi trip data using two different approaches:
- **PyArrow**: Vectorized columnar operations with zero-copy filtering
- **DuckDB**: SQL-based analytical queries with single-pass detection

Both implementations identify trips violating physics-based constraints (invalid distance, speed, or duration) within the top percentile of trip distances.

## Installation

```bash
# Install in editable mode with dependencies
pip install -e .

# Install with benchmark tools
pip install -e ".[benchmark]"
```

## Usage

### Command Line

```bash
# PyArrow implementation
find-outliers-pyarrow parquets/yellow_tripdata_2023-01.parquet

# DuckDB implementation
find-outliers-duckdb parquets/yellow_tripdata_2023-01.parquet

# Custom output format
find-outliers-pyarrow input.parquet --output-format csv --output results.csv
```

### Python API

```python
from detectors import detect_outliers_pyarrow, detect_outliers_duckdb, DEFAULT_CONFIG

# Run detection
result = detect_outliers_pyarrow('data.parquet', DEFAULT_CONFIG)
print(f"Found {result.outlier_count} outliers in {result.processing_time:.2f}s")

# Access results
outliers_table = result.outliers_table  # PyArrow table
stats = result.stats  # Detection statistics
```

### Benchmarking

```bash
# Benchmark all parquet files
python scripts/benchmark.py

# Sample 20 files distributed across date range
python scripts/benchmark.py --samples 20

# Generate comparison plots and analysis
python scripts/benchmark.py --output results.json --html report.html
```

## Outlier Detection Logic

**Phase 1**: Filter to top percentile by trip distance (default: 90th percentile)

**Phase 2**: Flag outliers violating any constraint:
- Distance: < 0.1 or > 800 miles
- Duration: ≤ 0 or > 10 hours
- Speed: < 2.5 or > 80 mph

## Project Structure

```
.
├── detectors/
│   ├── __init__.py                 # Module exports
│   ├── find_outliers_pyarrow.py    # PyArrow implementation
│   └── find_outliers_duckdb.py     # DuckDB implementation
├── scripts/
│   └── benchmark.py                # Performance benchmarking
├── parquets/                       # Input data directory
├── pyproject.toml                  # Package configuration
└── requirements.txt                # Editable install
```

## Performance Metrics

The benchmark tracks:
- **Processing time**: Data loading + outlier detection
- **Export time**: Writing results to file
- **Total time**: End-to-end execution
- **Memory usage**: Peak RSS memory consumption

## Requirements

- Python ≥ 3.14
- PyArrow ≥ 23.0.0
- DuckDB ≥ 1.4.4
- matplotlib ≥ 3.7.0 (for benchmarking)
- psutil ≥ 5.9.0 (for benchmarking)
