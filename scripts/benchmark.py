#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark for NYC Taxi Outlier Detection

Benchmarks PyArrow and DuckDB outlier detection implementations across
parquet files. Tracks processing time and memory usage, generates comparison
plots, and optionally produces LLM-analyzed HTML reports.

Usage:
    # Run on all parquet files
    python scripts/benchmark.py

    # Sample 20 files distributed across the date range
    python scripts/benchmark.py --samples 20

    # Specify output paths
    python scripts/benchmark.py --output my_results.json --html report.html

    # Run without LLM analysis (even if env vars set)
    python scripts/benchmark.py --no-llm

Implementations tested:
    - pyarrow: Zero-copy Arrow operations with percentile filtering
    - duckdb: SQL-based analytical engine with single-query detection

Outputs:
    - results.json: Raw and aggregated benchmark data
    - processing_time_by_date.png: Time comparison plot
    - memory_by_date.png: Memory comparison plot
    - benchmark_report.html: LLM-analyzed HTML report (optional)
"""

import argparse
import json
import os
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Import outlier detector implementations
from detectors.find_outliers_pyarrow import detect_outliers_pyarrow, DEFAULT_CONFIG
from detectors.find_outliers_duckdb import detect_outliers_duckdb

# Try importing optional dependencies
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting will be skipped.", file=sys.stderr)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not found. Memory tracking will be disabled.", file=sys.stderr)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Configuration
IMPLEMENTATIONS = {
    "pyarrow": detect_outliers_pyarrow,
    "duckdb": detect_outliers_duckdb,
}

def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract YYYY-MM date from parquet filename like yellow_tripdata_2013-05.parquet"""
    match = re.search(r'(\d{4})-(\d{2})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def get_parquet_files(parquets_dir: Path, samples: Optional[int] = None) -> List[Path]:
    """
    Get parquet files, optionally sampled evenly across the date range.

    Args:
        parquets_dir: Directory containing parquet files
        samples: Number of files to sample (None = all files)

    Returns:
        List of parquet file paths, sorted chronologically
    """
    parquet_files = sorted(parquets_dir.glob("*.parquet"))

    if not parquet_files:
        return []

    if samples is None or samples >= len(parquet_files):
        return parquet_files

    # Distribute samples evenly across date range
    total_files = len(parquet_files)
    indices = [int(i * total_files / samples) for i in range(samples)]
    return [parquet_files[i] for i in indices]


def benchmark_single_run(impl_name: str, impl_func: Callable, parquet_path: Path,
                         config: Dict) -> Dict[str, Any]:
    """
    Run a single benchmark of one implementation on one file.

    Measures both processing time and memory usage:
    - Time: From the implementation's internal timing
    - Memory: RSS (Resident Set Size) delta using psutil to capture C/C++ allocations

    Returns dict with: impl_name, date, processing_time, peak_memory_mb,
                       outlier_count, success, error
    """
    date_str = extract_date_from_filename(parquet_path.name)

    try:
        # Get file size
        file_size_mb = parquet_path.stat().st_size / 1024 / 1024

        # Measure memory if psutil available
        if HAS_PSUTIL:
            import gc
            # Force garbage collection to get baseline memory
            gc.collect()
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Run the implementation
            result = impl_func(str(parquet_path), config)

            # Get peak memory (RSS after processing)
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory_mb = max(0, memory_after - memory_before)
        else:
            # Run without memory tracking
            result = impl_func(str(parquet_path), config)
            peak_memory_mb = 0.0

        processing_time = result.processing_time
        outlier_count = result.outlier_count
        stats = result.stats

        return {
            "implementation": impl_name,
            "date": date_str,
            "filename": parquet_path.name,
            "file_size_mb": round(file_size_mb, 2),
            "processing_time": round(processing_time, 4),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "outlier_count": outlier_count,
            "total_trips": stats['total_rows'],
            "percentile_threshold": round(stats['percentile_threshold'], 2),
            "success": True,
            "error": None
        }

    except Exception as e:
        # Try to get file size even on error
        try:
            file_size_mb = parquet_path.stat().st_size / 1024 / 1024
        except:
            file_size_mb = None

        return {
            "implementation": impl_name,
            "date": date_str,
            "filename": parquet_path.name,
            "file_size_mb": round(file_size_mb, 2) if file_size_mb else None,
            "processing_time": None,
            "peak_memory_mb": None,
            "outlier_count": None,
            "total_trips": None,
            "percentile_threshold": None,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}"
        }


def aggregate_results(raw_results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate raw benchmark results by implementation and date.

    Returns:
        {
            "by_date": {
                "implementation_name": {
                    "2013-05": {"mean_time": X, "mean_memory": Y}, ...
                }
            },
            "overall_means": {
                "implementation_name": {"mean_time": X, "mean_memory": Y}
            }
        }
    """
    # Group by implementation and date
    by_impl_date = defaultdict(lambda: defaultdict(list))
    by_impl = defaultdict(list)

    for result in raw_results:
        if not result["success"]:
            continue

        impl = result["implementation"]
        date = result["date"]

        by_impl_date[impl][date].append({
            "time": result["processing_time"],
            "memory": result["peak_memory_mb"],
            "size": result["file_size_mb"],
            "outliers": result["outlier_count"]
        })
        by_impl[impl].append({
            "time": result["processing_time"],
            "memory": result["peak_memory_mb"],
            "size": result["file_size_mb"],
            "outliers": result["outlier_count"]
        })

    # Calculate means
    aggregated = {
        "by_date": {},
        "overall_means": {}
    }

    for impl, dates_data in by_impl_date.items():
        aggregated["by_date"][impl] = {}
        for date, values in dates_data.items():
            times = [v["time"] for v in values]
            memories = [v["memory"] for v in values]
            sizes = [v["size"] for v in values]
            outliers = [v["outliers"] for v in values]
            aggregated["by_date"][impl][date] = {
                "mean_time": round(sum(times) / len(times), 4),
                "mean_memory": round(sum(memories) / len(memories), 2),
                "mean_size": round(sum(sizes) / len(sizes), 2),
                "mean_outliers": round(sum(outliers) / len(outliers), 0),
                "count": len(values)
            }

    for impl, values in by_impl.items():
        times = [v["time"] for v in values]
        memories = [v["memory"] for v in values]
        sizes = [v["size"] for v in values]
        aggregated["overall_means"][impl] = {
            "mean_time": round(sum(times) / len(times), 4),
            "mean_memory": round(sum(memories) / len(memories), 2),
            "mean_size": round(sum(sizes) / len(sizes), 2),
            "count": len(values)
        }

    return aggregated


def plot_benchmarks(aggregated: Dict, output_dir: Path) -> Dict[str, str]:
    """
    Generate comparison plots showing relationships between metrics.

    Returns dict with plot paths for all generated visualizations.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plots: matplotlib not installed")
        return {}

    by_date = aggregated["by_date"]

    if not by_date:
        print("No data to plot")
        return {}

    # Plot 1: Processing Time vs File Size
    plt.figure(figsize=(14, 7))

    for impl_name in sorted(by_date.keys()):
        impl_data = by_date[impl_name]
        sizes = []
        times = []

        for date in sorted(impl_data.keys()):
            if "mean_size" in impl_data[date]:
                sizes.append(impl_data[date]["mean_size"])
                times.append(impl_data[date]["mean_time"])

        if sizes:
            plt.scatter(sizes, times, label=impl_name, s=100, alpha=0.7)
            # Add trend line
            z = np.polyfit(sizes, times, 1)
            p = np.poly1d(z)
            sorted_indices = np.argsort(sizes)
            plt.plot(np.array(sizes)[sorted_indices], p(np.array(sizes)[sorted_indices]),
                    '--', alpha=0.5, linewidth=2)

    plt.xlabel('Parquet File Size (MB)', fontsize=12, fontweight='bold')
    plt.ylabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Processing Time vs File Size',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    time_size_plot = output_dir / "processing_time_vs_size.png"
    plt.savefig(time_size_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {time_size_plot}")

    # Plot 2: Memory vs File Size
    plt.figure(figsize=(14, 7))

    for impl_name in sorted(by_date.keys()):
        impl_data = by_date[impl_name]
        sizes = []
        memories = []

        for date in sorted(impl_data.keys()):
            if "mean_size" in impl_data[date]:
                sizes.append(impl_data[date]["mean_size"])
                memories.append(impl_data[date]["mean_memory"])

        if sizes:
            plt.scatter(sizes, memories, label=impl_name, s=100, alpha=0.7)
            # Add trend line
            z = np.polyfit(sizes, memories, 1)
            p = np.poly1d(z)
            sorted_indices = np.argsort(sizes)
            plt.plot(np.array(sizes)[sorted_indices], p(np.array(sizes)[sorted_indices]),
                    '--', alpha=0.5, linewidth=2)

    plt.xlabel('Parquet File Size (MB)', fontsize=12, fontweight='bold')
    plt.ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    plt.title('Memory Usage vs File Size',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    memory_size_plot = output_dir / "memory_vs_size.png"
    plt.savefig(memory_size_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {memory_size_plot}")

    # Plot 3: Processing Time vs Number of Outliers
    plt.figure(figsize=(14, 7))

    for impl_name in sorted(by_date.keys()):
        impl_data = by_date[impl_name]
        outliers = []
        times = []

        for date in sorted(impl_data.keys()):
            if "mean_outliers" in impl_data[date]:
                outliers.append(impl_data[date]["mean_outliers"])
                times.append(impl_data[date]["mean_time"])

        if outliers:
            plt.scatter(outliers, times, label=impl_name, s=100, alpha=0.7)
            # Add trend line if there's variation
            if len(set(outliers)) > 1:
                z = np.polyfit(outliers, times, 1)
                p = np.poly1d(z)
                sorted_indices = np.argsort(outliers)
                plt.plot(np.array(outliers)[sorted_indices], p(np.array(outliers)[sorted_indices]),
                        '--', alpha=0.5, linewidth=2)

    plt.xlabel('Number of Outliers Detected', fontsize=12, fontweight='bold')
    plt.ylabel('Processing Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Processing Time vs Outlier Count',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    time_outliers_plot = output_dir / "processing_time_vs_outliers.png"
    plt.savefig(time_outliers_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {time_outliers_plot}")

    # Plot 4: Memory vs Number of Outliers
    plt.figure(figsize=(14, 7))

    for impl_name in sorted(by_date.keys()):
        impl_data = by_date[impl_name]
        outliers = []
        memories = []

        for date in sorted(impl_data.keys()):
            if "mean_outliers" in impl_data[date]:
                outliers.append(impl_data[date]["mean_outliers"])
                memories.append(impl_data[date]["mean_memory"])

        if outliers:
            plt.scatter(outliers, memories, label=impl_name, s=100, alpha=0.7)
            # Add trend line if there's variation
            if len(set(outliers)) > 1:
                z = np.polyfit(outliers, memories, 1)
                p = np.poly1d(z)
                sorted_indices = np.argsort(outliers)
                plt.plot(np.array(outliers)[sorted_indices], p(np.array(outliers)[sorted_indices]),
                        '--', alpha=0.5, linewidth=2)

    plt.xlabel('Number of Outliers Detected', fontsize=12, fontweight='bold')
    plt.ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    plt.title('Memory Usage vs Outlier Count',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    memory_outliers_plot = output_dir / "memory_vs_outliers.png"
    plt.savefig(memory_outliers_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {memory_outliers_plot}")

    # Plot 5: Outlier Count vs File Size (correlation analysis)
    plt.figure(figsize=(14, 7))

    for impl_name in sorted(by_date.keys()):
        impl_data = by_date[impl_name]
        sizes = []
        outliers = []

        for date in sorted(impl_data.keys()):
            if "mean_size" in impl_data[date] and "mean_outliers" in impl_data[date]:
                sizes.append(impl_data[date]["mean_size"])
                outliers.append(impl_data[date]["mean_outliers"])

        if sizes:
            plt.scatter(sizes, outliers, label=impl_name, s=100, alpha=0.7)
            # Add trend line to show correlation
            z = np.polyfit(sizes, outliers, 1)
            p = np.poly1d(z)
            sorted_indices = np.argsort(sizes)
            plt.plot(np.array(sizes)[sorted_indices], p(np.array(sizes)[sorted_indices]),
                    '--', alpha=0.5, linewidth=2)
            # Show correlation coefficient
            correlation = np.corrcoef(sizes, outliers)[0, 1]
            plt.text(0.02, 0.98 - (0.05 * list(sorted(by_date.keys())).index(impl_name)),
                    f'{impl_name} correlation: {correlation:.3f}',
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel('Parquet File Size (MB)', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Outliers Detected', fontsize=12, fontweight='bold')
    plt.title('Outlier Count vs File Size (Proportionality Analysis)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    outliers_size_plot = output_dir / "outliers_vs_size.png"
    plt.savefig(outliers_size_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved plot: {outliers_size_plot}")

    return {
        "time_size_plot": str(time_size_plot),
        "memory_size_plot": str(memory_size_plot),
        "time_outliers_plot": str(time_outliers_plot),
        "memory_outliers_plot": str(memory_outliers_plot),
        "outliers_size_plot": str(outliers_size_plot)
    }


def generate_html_report(results_data: Dict, plots: Dict, output_path: Path,
                        enable_llm: bool = True) -> Optional[str]:
    """
    Generate HTML report with optional LLM analysis.

    Returns HTML file path if successful, None otherwise.
    """
    # Check LLM availability
    llm_model = os.getenv('LLM_MODEL')
    llm_api_key = os.getenv('LLM_API_KEY')
    can_use_llm = HAS_ANTHROPIC and llm_model and llm_api_key and enable_llm

    if enable_llm and not can_use_llm:
        if not HAS_ANTHROPIC:
            print("Warning: anthropic package not installed. Skipping LLM analysis.")
        elif not llm_model or not llm_api_key:
            print("Warning: LLM_MODEL or LLM_API_KEY not set. Skipping LLM analysis.")

    # Extract just filenames for relative image paths (all files in same directory)
    time_size_plot_file = Path(plots.get('time_size_plot', '')).name if plots.get('time_size_plot') else ''
    memory_size_plot_file = Path(plots.get('memory_size_plot', '')).name if plots.get('memory_size_plot') else ''
    time_outliers_plot_file = Path(plots.get('time_outliers_plot', '')).name if plots.get('time_outliers_plot') else ''
    memory_outliers_plot_file = Path(plots.get('memory_outliers_plot', '')).name if plots.get('memory_outliers_plot') else ''
    outliers_size_plot_file = Path(plots.get('outliers_size_plot', '')).name if plots.get('outliers_size_plot') else ''

    # Create base HTML template
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYC Taxi Outlier Detection Benchmark Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #e74c3c;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metadata {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metadata-item {{
            margin: 8px 0;
        }}
        .metadata-label {{
            font-weight: bold;
            color: #555;
        }}
        .plots-section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .plot-description {{
            margin-top: 10px;
            color: #666;
            font-style: italic;
        }}
        .insights-box {{
            background: #fff9e6;
            border-left: 4px solid #f39c12;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .insights-box h3 {{
            margin-top: 0;
            color: #f39c12;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-table th {{
            background-color: #e74c3c;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .summary-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .summary-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .no-llm-note {{
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #777;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>🚕 NYC Taxi Outlier Detection Benchmark Report</h1>

    <div class="metadata">
        <h2>Benchmark Configuration</h2>
        <div class="metadata-item">
            <span class="metadata-label">Date:</span> {results_data['benchmark_config']['timestamp']}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Files Tested:</span> {results_data['benchmark_config']['num_files']}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Implementations:</span> {', '.join(results_data['benchmark_config']['implementations'])}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">System:</span> {results_data['benchmark_config']['system']}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Task:</span> Detect outlier taxi trips violating physics-based constraints
        </div>
    </div>

    <div class="plots-section">
        <h2>Performance Comparison Plots</h2>

        <div class="plot-container">
            <h3>Processing Time vs File Size</h3>
            <img src="{time_size_plot_file}" alt="Processing Time vs File Size">
            <p class="plot-description">
                Processing time scales with file size. Scatter plots with trend lines show how each
                implementation handles larger datasets. Lower values and flatter slopes indicate better scalability.
            </p>
        </div>

        <div class="plot-container">
            <h3>Memory Usage vs File Size</h3>
            <img src="{memory_size_plot_file}" alt="Memory Usage vs File Size">
            <p class="plot-description">
                Peak memory consumption relative to input file size. Shows memory efficiency and
                whether implementations load entire datasets into memory or use streaming approaches.
            </p>
        </div>

        <div class="plot-container">
            <h3>Processing Time vs Outlier Count</h3>
            <img src="{time_outliers_plot_file}" alt="Processing Time vs Outlier Count">
            <p class="plot-description">
                Relationship between number of outliers detected and processing time. Helps identify
                if performance degrades when more outliers need to be processed and filtered.
            </p>
        </div>

        <div class="plot-container">
            <h3>Memory Usage vs Outlier Count</h3>
            <img src="{memory_outliers_plot_file}" alt="Memory Usage vs Outlier Count">
            <p class="plot-description">
                Impact of outlier count on peak memory usage. Shows whether implementations need
                significant additional memory to store and process outlier results.
            </p>
        </div>

        <div class="plot-container">
            <h3>Outlier Count vs File Size (Proportionality)</h3>
            <img src="{outliers_size_plot_file}" alt="Outlier Count vs File Size">
            <p class="plot-description">
                Correlation between file size and number of outliers detected. Both implementations
                should detect identical outliers. Correlation coefficients show whether outlier rates
                are consistent across different data periods.
            </p>
        </div>
    </div>

    <div class="plots-section">
        <h2>Overall Performance Summary</h2>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Implementation</th>
                    <th>Mean Time (s)</th>
                    <th>Mean Memory (MB)</th>
                    <th>Files Tested</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add summary table rows
    for impl, stats in sorted(results_data['aggregated']['overall_means'].items()):
        html_template += f"""                <tr>
                    <td><strong>{impl}</strong></td>
                    <td>{stats['mean_time']:.4f}</td>
                    <td>{stats['mean_memory']:.2f}</td>
                    <td>{stats['count']}</td>
                </tr>
"""

    html_template += """            </tbody>
        </table>
    </div>

    <div id="llm-insights">
"""

    # Try to get LLM analysis
    if can_use_llm:
        print("\nGenerating LLM analysis...")
        try:
            # Prepare prompt
            results_json = json.dumps(results_data, indent=2)

            llm_prompt = f"""You are a performance analysis expert reviewing benchmark results for a NYC taxi outlier DETECTION system.

## Task Overview

This benchmark compares two implementations for DETECTING outlier taxi trips:

**PyArrow Implementation:**
- Uses zero-copy columnar operations
- Two-phase approach: calculate 90th percentile, then filter and validate
- Native Parquet support with compute functions (pc.quantile, pc.filter)
- Memory-efficient with direct Arrow operations

**DuckDB Implementation:**
- SQL-based analytical engine
- Single SQL query with CTEs (Common Table Expressions)
- Percentile calculation and filtering in one query
- Vectorized execution with native Parquet scanning

## Outlier Detection Strategy (Both Implementations)

**Phase 1:** Calculate 90th percentile of trip distances
**Phase 2:** Within top 10% of trips, find outliers violating ANY constraint:
- Distance: < 0.1 miles OR > 800 miles
- Duration: ≤ 0 hours OR > 10 hours
- Speed: < 2.5 mph OR > 80 mph

**Output:** Parquet files containing ONLY the detected outliers (not removed trips)

## Key Differences from Trip Removal

This is an OUTLIER DETECTION benchmark, not trip removal:
- **Goal:** Find problematic trips for inspection/analysis
- **Output:** Small set of outliers (typically < 1% of data)
- **Metric:** Number of outliers detected, not trips kept
- **Use case:** Data quality monitoring, anomaly investigation

## Analysis Task

Analyze the benchmark results below and provide insightful commentary in HTML format. Your analysis should include:

1. **Performance Comparison**: Which implementation is faster? By how much? Are there trade-offs in memory usage?

2. **Implementation Characteristics**:
   - Why might PyArrow or DuckDB perform better for this specific task?
   - How do zero-copy operations vs SQL engine affect performance?
   - Does the small output size (only outliers) favor one approach?

3. **Trends & Patterns**:
   - Any patterns across the chronological date range?
   - Do newer/older data files behave differently?
   - Does file size affect relative performance?

4. **Practical Recommendations**: Which implementation would you recommend for:
   - Maximum speed
   - Minimum memory usage
   - Production outlier detection systems
   - Interactive data quality analysis

Consider that outlier detection has different characteristics than bulk processing:
- Output is tiny (< 1% of input)
- Percentile calculation requires full scan
- Detection logic is relatively simple (WHERE clause filters)

Benchmark Results:
{results_json}

Provide ONLY the HTML content (no markdown, no code fences), starting with your analysis div.
Format as clean HTML with <h3> for sections, <h4> for subsections, <p> for paragraphs, and <ul>/<li> for lists.
"""

            # Strip anthropic/ prefix if present (for OpenAI Router compatibility)
            model = llm_model.replace('anthropic/', '') if 'anthropic/' in llm_model else llm_model

            # Call LLM
            client = anthropic.Anthropic(api_key=llm_api_key)
            message = client.messages.create(
                model=model,
                max_tokens=8000,
                messages=[
                    {"role": "user", "content": llm_prompt}
                ]
            )

            llm_response = message.content[0].text
            print(f"✓ LLM analysis completed (tokens: {message.usage.input_tokens} in, {message.usage.output_tokens} out)")

            html_template += f"""        <div class="insights-box">
            <h3>🤖 AI-Generated Performance Insights</h3>
            {llm_response}
        </div>
"""
        except Exception as e:
            print(f"Warning: LLM analysis failed: {e}")
            html_template += """        <div class="no-llm-note">
            <strong>Note:</strong> LLM analysis was attempted but failed. See console for details.
        </div>
"""
    else:
        html_template += """        <div class="no-llm-note">
            <strong>Note:</strong> LLM analysis was skipped. Set LLM_MODEL and LLM_API_KEY environment
            variables and install the anthropic package to enable AI-generated insights.
        </div>
"""

    html_template += """    </div>

    <footer>
        <p>Generated by NYC Taxi Outlier Detection Benchmark Suite | {timestamp}</p>
    </footer>
</body>
</html>
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"✓ HTML report saved: {output_path}")
    return str(output_path)


def main():
    """Main benchmark orchestration."""
    parser = argparse.ArgumentParser(
        description="Benchmark outlier detection implementations for NYC taxi data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--parquets-dir",
        type=Path,
        default=Path("parquets"),
        help="Directory containing parquet files (default: parquets)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of files to sample (evenly distributed across dates). Default: all files"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("scripts/benchmark/results.json"),
        help="Output path for JSON results (default: scripts/benchmark/results.json)"
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("scripts/benchmark/benchmark_report.html"),
        help="Output path for HTML report (default: scripts/benchmark/benchmark_report.html)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM analysis even if configured"
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=list(IMPLEMENTATIONS.keys()),
        default=list(IMPLEMENTATIONS.keys()),
        help="Specific implementations to test (default: all)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = args.output.parent
    if output_dir != Path('.') and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Validate parquets directory
    if not args.parquets_dir.exists():
        print(f"Error: Parquets directory not found: {args.parquets_dir}", file=sys.stderr)
        return 1

    # Get parquet files
    print(f"Scanning for parquet files in {args.parquets_dir}...")
    parquet_files = get_parquet_files(args.parquets_dir, args.samples)

    if not parquet_files:
        print("Error: No parquet files found", file=sys.stderr)
        return 1

    print(f"Found {len(parquet_files)} files to benchmark")
    if args.samples and args.samples < len(list(args.parquets_dir.glob("*.parquet"))):
        print(f"  (sampled {args.samples} files evenly across date range)")

    # Filter implementations
    selected_impls = {k: v for k, v in IMPLEMENTATIONS.items() if k in args.implementations}
    print(f"\nTesting {len(selected_impls)} implementations: {', '.join(selected_impls.keys())}")

    # Run benchmarks
    print("\nStarting benchmark suite...")
    print(f"  Total runs: {len(parquet_files)} files × {len(selected_impls)} implementations × 2 (time + memory) = {len(parquet_files) * len(selected_impls) * 2}")
    print("  Note: Each implementation runs twice per file (once for timing, once for memory)")
    print()

    raw_results = []

    # Create progress iterator
    total_runs = len(parquet_files) * len(selected_impls)
    if HAS_TQDM:
        progress = tqdm(total=total_runs, desc="Benchmarking", unit="run")
    else:
        progress = None
        print(f"Progress: 0/{total_runs} runs completed", end="\r")

    completed = 0
    for parquet_file in parquet_files:
        for impl_name, impl_func in selected_impls.items():
            result = benchmark_single_run(impl_name, impl_func, parquet_file, DEFAULT_CONFIG)
            raw_results.append(result)

            completed += 1
            if progress:
                progress.update(1)
                # Add info about current run
                status = "✓" if result["success"] else "✗"
                progress.set_postfix_str(f"{impl_name} on {result['date']} {status}")
            else:
                print(f"Progress: {completed}/{total_runs} runs completed", end="\r")

    if progress:
        progress.close()
    else:
        print()  # New line after progress

    # Count successes and failures
    successes = sum(1 for r in raw_results if r["success"])
    failures = sum(1 for r in raw_results if not r["success"])

    print(f"\n{'='*80}")
    print(f"Benchmark completed: {successes} successes, {failures} failures")
    print(f"{'='*80}\n")

    if failures > 0:
        print("Failed runs:")
        for result in raw_results:
            if not result["success"]:
                print(f"  ✗ {result['implementation']} on {result['filename']}: {result['error']}")
        print()

    # Aggregate results
    print("Aggregating results...")
    aggregated = aggregate_results(raw_results)

    # Print overall summary
    print("\nOverall Performance Summary:")
    print(f"{'='*80}")
    print(f"{'Implementation':<20} {'Mean Time (s)':<18} {'Mean Memory (MB)':<18} {'Files'}")
    print(f"{'-'*80}")
    for impl in sorted(aggregated['overall_means'].keys()):
        stats = aggregated['overall_means'][impl]
        print(f"{impl:<20} {stats['mean_time']:<18.4f} {stats['mean_memory']:<18.2f} {stats['count']}")
    print(f"{'='*80}\n")

    # Save results to JSON
    results_data = {
        "benchmark_config": {
            "timestamp": datetime.now().isoformat(),
            "num_files": len(parquet_files),
            "samples": args.samples,
            "implementations": list(selected_impls.keys()),
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count()
            }
        },
        "raw_results": raw_results,
        "aggregated": aggregated
    }

    print(f"Saving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    print("✓ Results saved\n")

    # Generate plots
    print("Generating plots...")
    plots_dir = args.output.parent if args.output.parent != Path('.') else Path('scripts/benchmark')
    if plots_dir != Path('.') and not plots_dir.exists():
        plots_dir.mkdir(parents=True, exist_ok=True)
    plots = plot_benchmarks(aggregated, plots_dir)
    print()

    # Generate HTML report
    print("Generating HTML report...")
    generate_html_report(results_data, plots, args.html, enable_llm=not args.no_llm)

    print(f"\n{'='*80}")
    print("✅ Benchmark suite completed successfully!")
    print(f"{'='*80}")
    print("\nOutputs:")
    print(f"  📊 Results JSON: {args.output}")
    if plots:
        print(f"  📈 Time plot: {plots.get('time_plot', 'N/A')}")
        print(f"  📈 Memory plot: {plots.get('memory_plot', 'N/A')}")
    print(f"  📄 HTML report: {args.html}")
    print()

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
