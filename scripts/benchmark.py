#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark for NYC Taxi Trip Analysis

Benchmarks all 7 implementations across all (or sampled) parquet files.
Tracks processing time and memory usage, generates comparison plots,
and optionally produces LLM-analyzed HTML reports.

Usage:
    # Run on all parquet files
    python scripts/benchmark.py

    # Sample 20 files distributed across the date range
    python scripts/benchmark.py --samples 20

    # Specify output paths
    python scripts/benchmark.py --output my_results.json --html report.html

    # Run without LLM analysis (even if env vars set)
    python scripts/benchmark.py --no-llm

All implementations tested:
    - baseline (2-pass, most accurate)
    - baseline_threads (2-pass multithreading)
    - baseline_process (2-pass multiprocessing)
    - onepass (1-pass with early percentile)
    - onepass_streaming (1-pass producer-consumer streaming)
    - pyarrow (2-pass zero-copy Arrow operations)
    - duckdb (SQL-based, typically fastest)

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
import tracemalloc
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Import all 7 implementations
from nyctrips.list_trips import analyze_trips as baseline_analyze
from nyctrips.list_trips_1pass import analyze_trips as onepass_analyze
from nyctrips.list_trips_1pass_streaming import analyze_trips as streaming_analyze
from nyctrips.list_trips_duckdb import analyze_trips as duckdb_analyze
from nyctrips.list_trips_pyarrow import analyze_trips as pyarrow_analyze
from nyctrips.list_trips_parallel_process import analyze_trips as parallel_process_analyze
from nyctrips.list_trips_parallel_threads import analyze_trips as parallel_threads_analyze
from nyctrips.trip_filters import DEFAULT_CONFIG

# Try importing optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting will be skipped.", file=sys.stderr)

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
    "baseline": baseline_analyze,
    "baseline_threads": parallel_threads_analyze,
    "baseline_process": parallel_process_analyze,
    "onepass": onepass_analyze,
    "onepass_streaming": streaming_analyze,
    "pyarrow": pyarrow_analyze,
    "duckdb": duckdb_analyze,
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

    Runs TWICE to avoid tracemalloc overhead affecting timing:
    - Run 1: Measure processing time WITHOUT tracemalloc (accurate timing)
    - Run 2: Measure memory WITH tracemalloc (accurate memory, timing ignored)

    Returns dict with: impl_name, date, processing_time, peak_memory_mb,
                       trips_count, success, error
    """
    date_str = extract_date_from_filename(parquet_path.name)

    try:
        # Prepare config with worker count for parallel implementations
        run_config = config.copy()
        if 'baseline_process' in impl_name or 'baseline_threads' in impl_name:
            run_config['num_workers'] = os.cpu_count()

        # === RUN 1: Measure TIME without tracemalloc overhead ===
        result_time = impl_func(str(parquet_path), run_config)
        processing_time = result_time.processing_time
        trips_count = len(result_time.df_full) if hasattr(result_time, 'df_full') else 0

        # === RUN 2: Measure MEMORY with tracemalloc ===
        tracemalloc.start()
        _ = impl_func(str(parquet_path), run_config)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_memory_mb = peak / 1024 / 1024

        return {
            "implementation": impl_name,
            "date": date_str,
            "filename": parquet_path.name,
            "processing_time": round(processing_time, 4),
            "peak_memory_mb": round(peak_memory_mb, 2),
            "trips_count": trips_count,
            "success": True,
            "error": None
        }

    except Exception as e:
        # Ensure tracemalloc cleanup on error
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        return {
            "implementation": impl_name,
            "date": date_str,
            "filename": parquet_path.name,
            "processing_time": None,
            "peak_memory_mb": None,
            "trips_count": None,
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
            "memory": result["peak_memory_mb"]
        })
        by_impl[impl].append({
            "time": result["processing_time"],
            "memory": result["peak_memory_mb"]
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
            aggregated["by_date"][impl][date] = {
                "mean_time": round(sum(times) / len(times), 4),
                "mean_memory": round(sum(memories) / len(memories), 2),
                "count": len(values)
            }

    for impl, values in by_impl.items():
        times = [v["time"] for v in values]
        memories = [v["memory"] for v in values]
        aggregated["overall_means"][impl] = {
            "mean_time": round(sum(times) / len(times), 4),
            "mean_memory": round(sum(memories) / len(memories), 2),
            "count": len(values)
        }

    return aggregated


def plot_benchmarks(aggregated: Dict, output_dir: Path) -> Dict[str, str]:
    """
    Generate time and memory comparison plots.

    Returns dict with plot paths: {"time_plot": path, "memory_plot": path}
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plots: matplotlib not installed")
        return {}

    by_date = aggregated["by_date"]

    # Get all dates and sort them chronologically
    all_dates = set()
    for impl_data in by_date.values():
        all_dates.update(impl_data.keys())
    sorted_dates = sorted(all_dates)

    if not sorted_dates:
        print("No data to plot")
        return {}

    # Plot 1: Processing Time by Date
    plt.figure(figsize=(14, 7))

    for impl_name in sorted(by_date.keys()):
        impl_data = by_date[impl_name]
        dates = []
        times = []

        for date in sorted_dates:
            if date in impl_data:
                dates.append(date)
                times.append(impl_data[date]["mean_time"])

        if dates:
            plt.plot(dates, times, 'o-', label=impl_name, linewidth=2, markersize=6)

    plt.xlabel('Date (YYYY-MM)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Processing Time (seconds)', fontsize=12, fontweight='bold')
    plt.title('Processing Time Comparison Across All Implementations\n(Chronological by Parquet File Date)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    time_plot_path = output_dir / "processing_time_by_date.png"
    plt.savefig(time_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved time plot: {time_plot_path}")

    # Plot 2: Memory by Date
    plt.figure(figsize=(14, 7))

    for impl_name in sorted(by_date.keys()):
        impl_data = by_date[impl_name]
        dates = []
        memories = []

        for date in sorted_dates:
            if date in impl_data:
                dates.append(date)
                memories.append(impl_data[date]["mean_memory"])

        if dates:
            plt.plot(dates, memories, 's-', label=impl_name, linewidth=2, markersize=6)

    plt.xlabel('Date (YYYY-MM)', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Peak Memory (MB)', fontsize=12, fontweight='bold')
    plt.title('Memory Usage Comparison Across All Implementations\n(Chronological by Parquet File Date)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    memory_plot_path = output_dir / "memory_by_date.png"
    plt.savefig(memory_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved memory plot: {memory_plot_path}")

    return {
        "time_plot": str(time_plot_path),
        "memory_plot": str(memory_plot_path)
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
    time_plot_file = Path(plots.get('time_plot', '')).name if plots.get('time_plot') else ''
    memory_plot_file = Path(plots.get('memory_plot', '')).name if plots.get('memory_plot') else ''

    # Create base HTML template
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYC Taxi Benchmark Report</title>
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
            border-bottom: 3px solid #3498db;
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
            background-color: #3498db;
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
    <h1>🚕 NYC Taxi Performance Benchmark Report</h1>

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
    </div>

    <div class="plots-section">
        <h2>Performance Comparison Plots</h2>

        <div class="plot-container">
            <h3>Processing Time by Date</h3>
            <img src="{time_plot_file}" alt="Processing Time Comparison">
            <p class="plot-description">
                This plot shows mean processing time across all implementations over the chronological
                date range of parquet files. Lower values indicate faster processing.
            </p>
        </div>

        <div class="plot-container">
            <h3>Memory Usage by Date</h3>
            <img src="{memory_plot_file}" alt="Memory Usage Comparison">
            <p class="plot-description">
                This plot shows mean peak memory usage across all implementations. Lower values
                indicate more memory-efficient processing.
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

            llm_prompt = f"""You are a performance analysis expert reviewing benchmark results for a NYC taxi data processing system.

## Fundamental Approaches: 2-Pass vs 1-Pass

**Two-Pass Approach:**
- **Pass 1**: Read entire dataset to calculate the 90th percentile distance threshold
- **Pass 2**: Read dataset again to filter trips above threshold and apply additional filters
- **Pros**: Most accurate results, straightforward logic, easier to maintain
- **Cons**: Reads data twice, slower overall processing time
- **Used by**: baseline, pyarrow, baseline_process, baseline_threads, duckdb

**One-Pass Approach:**
- **Single Pass**: Calculate percentile first (quick scan), then stream/filter data in one pass
- **Pros**: ~40-50% faster, reads data only once, lower I/O overhead
- **Cons**: Small accuracy trade-off (~1-2% difference in trip counts vs baseline)
- **Used by**: onepass, onepass_streaming

## Implementation Technical Details

Each implementation uses a different approach to process NYC taxi parquet files:

1. **baseline** - Two-pass algorithm using Pandas (Reference Implementation)
   - Pass 1: Calculate 90th percentile distance threshold
   - Pass 2: Filter trips above threshold and apply additional filters
   - Library: Pandas (DataFrame operations)
   - Approach: 2-pass
   - Strengths: Most accurate, well-tested reference implementation
   - Tradeoffs: Slower due to two full passes over data

2. **onepass** - Single-pass algorithm with early percentile calculation
   - Calculates percentile first, then filters in one pass
   - Library: Pandas
   - Approach: 1-pass
   - Strengths: ~40-50% faster than baseline, still accurate
   - Tradeoffs: Small accuracy difference (~1-2%) vs baseline

3. **onepass_streaming** - Producer-consumer streaming pattern based on 1-pass
   - Streams data through queue between reader and processor threads
   - Overlapped I/O and processing with single-pass filtering
   - Library: Pandas + threading (queue-based)
   - Approach: 1-pass
   - Strengths: Best for cold cache scenarios, hides I/O latency
   - Tradeoffs: More complex architecture, thread coordination overhead

4. **duckdb** - SQL-based query engine
   - Uses DuckDB's embedded analytical database
   - Single SQL query with native parquet reading (internally 2-pass for percentile)
   - Library: DuckDB (columnar execution)
   - Approach: 2-pass (internal)
   - Strengths: Often fastest overall, highly optimized query execution
   - Tradeoffs: Higher memory usage, different result ordering

5. **pyarrow** - Zero-copy Arrow operations
   - Uses PyArrow's native compute functions
   - Zero-copy filtering and column selection
   - Library: PyArrow (Apache Arrow)
   - Approach: 2-pass
   - Strengths: Most memory-efficient, no data copying
   - Tradeoffs: Limited by Arrow API capabilities, still needs two passes

6. **baseline_process** - Multiprocessing parallelism based on baseline
   - Uses ProcessPoolExecutor with {results_data['benchmark_config']['system']['cpu_count']} workers
   - True parallel processing (bypasses Python GIL)
   - Library: multiprocessing + Pandas
   - Approach: 2-pass (parallelized)
   - Strengths: Can utilize multiple CPU cores fully
   - Tradeoffs: Process spawning overhead, memory duplication per worker

7. **baseline_threads** - Multithreading parallelism based on baseline
   - Uses ThreadPoolExecutor with {results_data['benchmark_config']['system']['cpu_count']} workers
   - Thread-based parallelism (limited by Python GIL)
   - Library: threading + Pandas
   - Approach: 2-pass (parallelized)
   - Strengths: Lower overhead than multiprocessing
   - Tradeoffs: GIL contention limits actual parallelism for CPU-bound work

## Analysis Task

Analyze the benchmark results below and provide insightful commentary in HTML format. Your analysis should be placed inside a div and include:

1. **Key Performance Insights**: Which implementations perform best for time and memory? Are there clear winners? Reference the technical characteristics above to explain WHY certain implementations perform better.

2. **Algorithm Tradeoffs**: Discuss speed vs memory tradeoffs observed across implementations. Explain how architectural choices (single-pass vs two-pass, zero-copy, parallelism, SQL engine) affect performance.

3. **Trends & Patterns**: Any patterns across the chronological date range? Do newer data files behave differently? Does file size affect relative performance?

4. **Recommendations**: Which implementation would you recommend for:
   - Maximum speed (consider both cold and hot cache scenarios)
   - Minimum memory usage
   - Best balance of speed and memory
   - Production use (considering reliability, maintainability, and performance)

Consider the technical details when making recommendations. For example, streaming might excel with cold cache, pyarrow for memory constraints, duckdb for raw speed, etc.

Format your response as clean HTML (divs, paragraphs, lists, etc.) suitable for insertion into the page. Make it visually appealing and informative. Use <h3> for section headers and <h4> for subsections if needed.

Benchmark Results:
{results_json}

Provide ONLY the HTML content (no markdown, no code fences), starting with your analysis div.
"""

            # Handle model naming
            model = llm_model or 'claude-3-5-sonnet-20241022'
            if 'anthropic/' in model:
                model = model.replace('anthropic/', '')
            model_map = {
                'claude-sonnet-4-5': 'claude-sonnet-4-20250514',
                'claude-3-5-sonnet': 'claude-3-5-sonnet-20241022',
            }
            model = model_map.get(model, model)

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
        <p>Generated by NYC Taxi Benchmark Suite | {timestamp}</p>
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
        description="Comprehensive performance benchmark for NYC taxi implementations",
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
