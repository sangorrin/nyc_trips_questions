#!/usr/bin/env python3
"""
NYC Taxi Outlier Detection - Performance Profiling Script

Profiles PyArrow and DuckDB implementations to identify bottlenecks, aggregates
timing statistics across multiple runs, and generates HTML reports with LLM analysis.

Usage:
    # Profile on 10 sample files
    python scripts/profiling.py --samples 10

    # Custom output directory
    python scripts/profiling.py --samples 5 --output-dir custom_results/

    # Skip LLM analysis
    python scripts/profiling.py --samples 5 --no-llm
    - {output_dir}/profiling_report.html: HTML report with LLM analysis
"""

import argparse
import cProfile
import html
import json
import os
import pstats
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import io
import statistics

# Optional dependencies with graceful degradation
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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors.find_outliers_pyarrow import detect_outliers_pyarrow
from detectors.find_outliers_duckdb import detect_outliers_duckdb


# Constants
DEFAULT_OUTPUT_DIR = Path("scripts/profiling")
DEFAULT_PARQUET_DIR = Path("parquets")
TOP_N_FUNCTIONS = 50  # Number of top functions to keep per implementation


@dataclass
class FunctionStats:
    """Statistics for a single function across multiple profiling runs."""
    name: str
    mean_cumtime: float
    mean_calls: float
    mean_percall: float
    std_cumtime: float
    total_cumtime: float
    min_cumtime: float
    max_cumtime: float


@dataclass
class ProfilingResult:
    """Results from profiling a single run."""
    implementation: str
    filename: str
    file_size_mb: float
    processing_time: float
    function_stats: Dict[str, Dict[str, float]]  # {function_name: {cumtime, calls, percall}}
    success: bool
    error: Optional[str] = None


class CProfileWrapper:
    """Wrapper for Python's built-in cProfile profiler."""

    def __init__(self):
        self.profiler = None
        self.stats = None

    def start(self):
        """Start profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def stop(self):
        """Stop profiling and extract statistics."""
        if self.profiler:
            self.profiler.disable()

    def get_function_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Extract per-function statistics from cProfile data.

        Returns:
            Dict mapping function names to their stats (cumtime, calls, percall)
        """
        if not self.profiler:
            return {}

        # Create Stats object from profiler
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)

        function_stats = {}

        # Extract statistics
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            # func is tuple: (filename, line_number, function_name)
            filename, line_num, func_name = func

            # Create readable function identifier
            # Extract module name from filename
            path_obj = Path(filename)
            if 'detectors' in path_obj.parts:
                module = 'detectors.' + path_obj.stem
            elif 'site-packages' in str(filename):
                # Extract package name for external dependencies
                parts = str(filename).split('site-packages/')
                if len(parts) > 1:
                    pkg_path = parts[1].split('/')[0]
                    module = pkg_path
                else:
                    module = path_obj.stem
            else:
                module = path_obj.stem

            full_name = f"{module}.{func_name}"

            function_stats[full_name] = {
                'cumtime': ct,  # Cumulative time
                'calls': nc,    # Number of calls
                'percall': ct / nc if nc > 0 else 0.0,  # Time per call
                'tottime': tt,  # Total time (excluding subcalls)
            }

        return function_stats


def get_parquet_files(parquet_dir: Path, sample_size: Optional[int] = None) -> List[Path]:
    """
    Get list of parquet files, optionally sampling evenly across date range.

    Args:
        parquet_dir: Directory containing parquet files
        sample_size: If specified, return evenly distributed sample

    Returns:
        List of parquet file paths
    """
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    # Get all parquet files and sort by name (chronological for yellow_tripdata_YYYY-MM.parquet)
    all_files = sorted(parquet_dir.glob("*.parquet"))

    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    if sample_size is None or sample_size >= len(all_files):
        return all_files

    # Sample evenly across the date range
    step = len(all_files) / sample_size
    sampled_indices = [int(i * step) for i in range(sample_size)]
    sampled_files = [all_files[i] for i in sampled_indices]

    return sampled_files


def profile_single_run(
    parquet_path: Path,
    implementation: str
) -> ProfilingResult:
    """
    Profile a single outlier detection run using cProfile.

    Args:
        parquet_path: Path to parquet file
        implementation: 'pyarrow' or 'duckdb'

    Returns:
        ProfilingResult with function-level statistics
    """
    file_size_mb = parquet_path.stat().st_size / (1024 * 1024)

    # Select detector function
    if implementation == 'pyarrow':
        detector_func = detect_outliers_pyarrow
    elif implementation == 'duckdb':
        detector_func = detect_outliers_duckdb
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    # Default config
    config = {
        'min_distance_miles': 0.1,
        'max_distance_miles': 800,
        'min_speed_mph': 2.5,
        'max_speed_mph': 80,
        'max_trip_hours': 10,
        'percentile': 0.90
    }

    try:
        # Use cProfile
        profiler = CProfileWrapper()

        profiler.start()
        start_time = time.time()
        result = detector_func(str(parquet_path), config)
        processing_time = time.time() - start_time
        profiler.stop()

        function_stats = profiler.get_function_stats()
        return ProfilingResult(
            implementation=implementation,
            filename=parquet_path.name,
            file_size_mb=file_size_mb,
            processing_time=processing_time,
            function_stats=function_stats,
            success=True,
            error=None
        )

    except Exception as e:
        return ProfilingResult(
            implementation=implementation,
            filename=parquet_path.name,
            file_size_mb=file_size_mb,
            processing_time=0.0,
            function_stats={},
            success=False,
            error=str(e)
        )


def aggregate_profiling_results(results: List[ProfilingResult]) -> List[FunctionStats]:
    """
    Aggregate profiling results across multiple runs.

    Args:
        results: List of ProfilingResult objects

    Returns:
        List of FunctionStats sorted by mean cumulative time (descending)
    """
    # Collect all function data across runs
    function_data: Dict[str, List[Dict[str, float]]] = {}

    for result in results:
        if not result.success:
            continue

        for func_name, stats in result.function_stats.items():
            if func_name not in function_data:
                function_data[func_name] = []
            function_data[func_name].append(stats)

    # Calculate statistics for each function
    aggregated_stats = []

    for func_name, runs in function_data.items():
        cumtimes = [run['cumtime'] for run in runs]
        calls = [run['calls'] for run in runs]
        percalls = [run['percall'] for run in runs]

        mean_cumtime = statistics.mean(cumtimes)
        std_cumtime = statistics.stdev(cumtimes) if len(cumtimes) > 1 else 0.0

        aggregated_stats.append(FunctionStats(
            name=func_name,
            mean_cumtime=mean_cumtime,
            mean_calls=statistics.mean(calls),
            mean_percall=statistics.mean(percalls),
            std_cumtime=std_cumtime,
            total_cumtime=sum(cumtimes),
            min_cumtime=min(cumtimes),
            max_cumtime=max(cumtimes)
        ))

    # Sort by mean cumulative time (biggest bottlenecks first)
    aggregated_stats.sort(key=lambda x: x.mean_cumtime, reverse=True)

    # Keep top N functions
    return aggregated_stats[:TOP_N_FUNCTIONS]


def save_profile_json(
    stats: List[FunctionStats],
    output_path: Path,
    num_files: int,
    implementation: str
) -> None:
    """
    Save profiling statistics to JSON file.

    Args:
        stats: List of FunctionStats
        output_path: Output file path
        num_files: Number of files profiled
        implementation: Implementation name
    """
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'profiler': 'cprofile',
            'implementation': implementation,
            'num_files': num_files,
            'top_n_functions': len(stats)
        },
        'functions': [asdict(stat) for stat in stats]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved {implementation} profile to {output_path}")


def get_llm_analysis(
    pyarrow_stats: List[FunctionStats],
    duckdb_stats: List[FunctionStats],
    pyarrow_source: str,
    duckdb_source: str
) -> Optional[str]:
    """
    Get LLM analysis of profiling results.

    Args:
        pyarrow_stats: PyArrow function statistics
        duckdb_stats: DuckDB function statistics
        pyarrow_source: Full source code of PyArrow implementation
        duckdb_source: Full source code of DuckDB implementation

    Returns:
        LLM analysis text or None if unavailable
    """
    if not HAS_ANTHROPIC:
        return None

    llm_model = os.getenv('LLM_MODEL')
    llm_api_key = os.getenv('LLM_API_KEY')

    if not llm_model or not llm_api_key:
        return None

    # Prepare profiling summary for LLM
    pyarrow_top10 = pyarrow_stats[:10]
    duckdb_top10 = duckdb_stats[:10]

    pyarrow_summary = "\n".join([
        f"{i+1}. {stat.name}: {stat.mean_cumtime:.4f}s (±{stat.std_cumtime:.4f}s), {stat.mean_calls:.0f} calls"
        for i, stat in enumerate(pyarrow_top10)
    ])

    duckdb_summary = "\n".join([
        f"{i+1}. {stat.name}: {stat.mean_cumtime:.4f}s (±{stat.std_cumtime:.4f}s), {stat.mean_calls:.0f} calls"
        for i, stat in enumerate(duckdb_top10)
    ])

    # Construct prompt
    prompt = f"""You are analyzing performance profiling results from two implementations of an outlier detection algorithm for NYC Taxi trip data:

## Profiling Results

### PyArrow Implementation - Top 10 Bottlenecks:
{pyarrow_summary}

### DuckDB Implementation - Top 10 Bottlenecks:
{duckdb_summary}

## Source Code

### PyArrow Implementation (detectors/find_outliers_pyarrow.py):
```python
{pyarrow_source}
```

### DuckDB Implementation (detectors/find_outliers_duckdb.py):
```python
{duckdb_source}
```

## Analysis Task

Based on the profiling results and source code, please:

1. **Identify Key Bottlenecks**: What are the main performance bottlenecks in each implementation?

2. **Explain Performance Differences**: Why is PyArrow significantly faster than DuckDB (approximately 7x based on benchmarks)? What architectural or implementation differences explain this?

3. **Root Cause Analysis**: For the top bottleneck functions, explain WHY they are slow. Consider:
   - Algorithmic complexity
   - Data copying vs zero-copy operations
   - Query planning overhead
   - I/O patterns
   - Memory allocation patterns

4. **Optimization Opportunities**: Propose specific, actionable optimizations for both implementations. Focus on:
   - Quick wins (low effort, high impact)
   - Architectural improvements
   - Algorithm changes

5. **Trade-offs**: Discuss any trade-offs between the two approaches (e.g., code simplicity vs performance, memory vs speed).

Please provide a detailed technical analysis with specific references to the profiling data and code patterns.
"""

    try:
        # Strip anthropic/ prefix if present (for OpenAI Router compatibility)
        model = llm_model.replace('anthropic/', '') if 'anthropic/' in llm_model else llm_model

        client = anthropic.Anthropic(api_key=llm_api_key)

        print("Requesting LLM analysis...")
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        analysis = response.content[0].text
        print("LLM analysis completed")
        return analysis

    except Exception as e:
        print(f"Warning: LLM analysis failed: {e}")
        return None


def generate_html_report(
    pyarrow_stats: List[FunctionStats],
    duckdb_stats: List[FunctionStats],
    output_path: Path,
    llm_analysis: Optional[str],
    num_files: int
) -> None:
    """
    Generate fancy HTML report with profiling results and LLM analysis.

    Args:
        pyarrow_stats: PyArrow function statistics
        duckdb_stats: DuckDB function statistics
        output_path: Output HTML file path
        llm_analysis: Optional LLM analysis text
        num_files: Number of files profiled
    """
    # Prepare data for tables (top 20)
    pyarrow_top20 = pyarrow_stats[:20]
    duckdb_top20 = duckdb_stats[:20]

    # Generate table rows
    def make_table_rows(stats: List[FunctionStats]) -> str:
        rows = []
        for i, stat in enumerate(stats):
            # Color coding based on cumulative time
            if stat.mean_cumtime > 1.0:
                badge_class = "badge-danger"
            elif stat.mean_cumtime > 0.5:
                badge_class = "badge-warning"
            elif stat.mean_cumtime > 0.1:
                badge_class = "badge-info"
            else:
                badge_class = "badge-success"

            # HTML-escape function name to handle < > characters
            escaped_name = html.escape(stat.name)

            rows.append(f"""
                <tr>
                    <td>{i+1}</td>
                    <td class="function-name">{escaped_name}</td>
                    <td><span class="badge {badge_class}">{stat.mean_cumtime:.4f}s</span></td>
                    <td>{stat.std_cumtime:.4f}s</td>
                    <td>{int(stat.mean_calls)}</td>
                    <td>{stat.mean_percall*1000:.2f}ms</td>
                </tr>
            """)
        return "\n".join(rows)

    pyarrow_rows = make_table_rows(pyarrow_top20)
    duckdb_rows = make_table_rows(duckdb_top20)

    # LLM analysis section
    llm_section = ""
    if llm_analysis:
        # Convert markdown to HTML properly
        analysis_html = llm_analysis

        # Convert code blocks (```...```)
        analysis_html = re.sub(
            r'```(?:\w+)?\n(.*?)\n```',
            r'<pre><code>\1</code></pre>',
            analysis_html,
            flags=re.DOTALL
        )

        # Convert inline code (`...`)
        analysis_html = re.sub(r'`([^`]+)`', r'<code>\1</code>', analysis_html)

        # Convert headers (must be done before escaping)
        analysis_html = re.sub(r'^#### (.*?)$', r'<h5>\1</h5>', analysis_html, flags=re.MULTILINE)
        analysis_html = re.sub(r'^### (.*?)$', r'<h4>\1</h4>', analysis_html, flags=re.MULTILINE)
        analysis_html = re.sub(r'^## (.*?)$', r'<h3>\1</h3>', analysis_html, flags=re.MULTILINE)
        analysis_html = re.sub(r'^# (.*?)$', r'<h2>\1</h2>', analysis_html, flags=re.MULTILINE)

        # Convert bold (**text**)
        analysis_html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', analysis_html)

        # Convert unordered lists
        analysis_html = re.sub(r'^- (.*?)$', r'<li>\1</li>', analysis_html, flags=re.MULTILINE)
        analysis_html = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', analysis_html, flags=re.DOTALL)
        # Fix nested ul tags
        analysis_html = re.sub(r'</ul>\s*<ul>', '', analysis_html)

        # Convert paragraphs (double newlines become paragraph breaks)
        # First split by existing HTML tags to avoid breaking them
        lines = analysis_html.split('\n')
        in_block = False
        result_lines = []
        current_para = []

        for line in lines:
            stripped = line.strip()
            # Check if line is an HTML tag
            is_tag = stripped.startswith('<') and '>' in stripped

            if is_tag or stripped.startswith('<ul>') or stripped.startswith('</ul>'):
                if current_para:
                    result_lines.append('<p>' + ' '.join(current_para) + '</p>')
                    current_para = []
                result_lines.append(line)
            elif stripped:
                current_para.append(stripped)
            elif current_para:
                result_lines.append('<p>' + ' '.join(current_para) + '</p>')
                current_para = []

        if current_para:
            result_lines.append('<p>' + ' '.join(current_para) + '</p>')

        analysis_html = '\n'.join(result_lines)

        llm_section = f"""
        <div class="section">
            <h2>🤖 AI-Powered Bottleneck Analysis</h2>
            <div class="llm-analysis">
                {analysis_html}
            </div>
        </div>
        """

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYC Taxi Outlier Detection - Profiling Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .metadata {{
            background: #f8f9fa;
            padding: 20px 40px;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }}

        .metadata-item {{
            text-align: center;
            padding: 10px;
        }}

        .metadata-label {{
            font-size: 0.85em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}

        .metadata-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}

        .section {{
            padding: 40px;
        }}

        .section h2 {{
            font-size: 2em;
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}

        .comparison {{
            display: flex;
            flex-direction: column;
            gap: 30px;
            margin-top: 30px;
        }}

        .implementation-card {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .implementation-card h3 {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #495057;
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 10px;
        }}

        .pyarrow-card h3 {{
            border-left: 5px solid #28a745;
        }}

        .duckdb-card h3 {{
            border-left: 5px solid #dc3545;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 10px;
            overflow: hidden;
        }}

        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 1px;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e9ecef;
        }}

        tbody tr:hover {{
            background: #f8f9fa;
            transition: background 0.2s;
        }}

        .function-name {{
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #495057;
        }}

        .badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            display: inline-block;
        }}

        .badge-danger {{
            background: #dc3545;
            color: white;
        }}

        .badge-warning {{
            background: #ffc107;
            color: #333;
        }}

        .badge-info {{
            background: #17a2b8;
            color: white;
        }}

        .badge-success {{
            background: #28a745;
            color: white;
        }}

        .llm-analysis {{
            background: #f8f9fa;
            border-left: 5px solid #667eea;
            padding: 25px;
            border-radius: 10px;
            line-height: 1.8;
            margin-top: 20px;
        }}

        .llm-analysis h3, .llm-analysis h4, .llm-analysis h5 {{
            color: #667eea;
            margin-top: 20px;
            margin-bottom: 10px;
        }}

        .llm-analysis strong {{
            color: #495057;
        }}

        .llm-analysis p {{
            margin: 15px 0;
        }}

        .llm-analysis pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}

        .llm-analysis code {{
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #d63384;
        }}

        .llm-analysis pre code {{
            background: transparent;
            padding: 0;
            color: #f8f8f2;
        }}

        .llm-analysis ul {{
            margin: 15px 0;
            padding-left: 30px;
        }}

        .llm-analysis li {{
            margin: 8px 0;
        }}

        footer {{
            background: #343a40;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .metadata {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚕 NYC Taxi Outlier Detection</h1>
            <p>Performance Profiling Report: PyArrow vs DuckDB</p>
        </header>

        <div class="metadata">
            <div class="metadata-item">
                <div class="metadata-label">Profiler</div>
                <div class="metadata-value">CPROFILE</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Files Analyzed</div>
                <div class="metadata-value">{num_files}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Top Functions</div>
                <div class="metadata-value">{TOP_N_FUNCTIONS}</div>
            </div>
            <div class="metadata-item">
                <div class="metadata-label">Generated</div>
                <div class="metadata-value">{datetime.now().strftime('%Y-%m-%d')}</div>
            </div>
        </div>

        <div class="section">
            <h2>⚡ Performance Bottlenecks Comparison</h2>
            <p style="margin-top: 15px; color: #6c757d; font-size: 1.1em;">
                Top 20 functions by cumulative execution time (mean across {num_files} runs)
            </p>

            <div class="comparison">
                <div class="implementation-card pyarrow-card">
                    <h3>PyArrow Implementation</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Function</th>
                                <th>Mean Time</th>
                                <th>Std Dev</th>
                                <th>Calls</th>
                                <th>Per Call</th>
                            </tr>
                        </thead>
                        <tbody>
                            {pyarrow_rows}
                        </tbody>
                    </table>
                </div>

                <div class="implementation-card duckdb-card">
                    <h3>DuckDB Implementation</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Function</th>
                                <th>Mean Time</th>
                                <th>Std Dev</th>
                                <th>Calls</th>
                                <th>Per Call</th>
                            </tr>
                        </thead>
                        <tbody>
                            {duckdb_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        {llm_section}

        <footer>
            <p>NYC Taxi Performance Analysis • Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="margin-top: 5px; opacity: 0.7;">Profiled with cProfile • Python Performance Profiling</p>
        </footer>
    </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Generated HTML report: {output_path}")


def main():
    """Main profiling workflow."""
    parser = argparse.ArgumentParser(
        description="Profile NYC Taxi outlier detection implementations using cProfile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of files to sample (default: all files)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--parquet-dir',
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help=f'Directory containing parquet files (default: {DEFAULT_PARQUET_DIR})'
    )

    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM analysis'
    )

    args = parser.parse_args()

    # Get parquet files
    print(f"Scanning for parquet files in {args.parquet_dir}...")
    parquet_files = get_parquet_files(args.parquet_dir, args.samples)
    print(f"Found {len(parquet_files)} parquet files")

    if args.samples:
        print(f"Sampling {args.samples} files evenly across date range")

    # Profile both implementations
    implementations = ['pyarrow', 'duckdb']
    all_results: Dict[str, List[ProfilingResult]] = {impl: [] for impl in implementations}

    for impl in implementations:
        print(f"\n{'='*60}")
        print(f"Profiling {impl.upper()} implementation")
        print(f"{'='*60}")

        if HAS_TQDM:
            pbar = tqdm(total=len(parquet_files), desc=f"{impl.upper()}", unit="file")

        for parquet_file in parquet_files:
            if not HAS_TQDM:
                print(f"  Profiling {parquet_file.name}...")

            result = profile_single_run(parquet_file, impl)
            all_results[impl].append(result)

            if HAS_TQDM:
                pbar.update(1)
                if not result.success:
                    tqdm.write(f"  Failed: {parquet_file.name} - {result.error}")
            else:
                if result.success:
                    print(f"    ✓ {result.processing_time:.3f}s")
                else:
                    print(f"    ✗ Error: {result.error}")

        if HAS_TQDM:
            pbar.close()

    # Aggregate results
    print(f"\n{'='*60}")
    print("Aggregating profiling results...")
    print(f"{'='*60}")

    pyarrow_stats = aggregate_profiling_results(all_results['pyarrow'])
    duckdb_stats = aggregate_profiling_results(all_results['duckdb'])

    print(f"\nPyArrow: {len(pyarrow_stats)} top functions identified")
    print(f"DuckDB: {len(duckdb_stats)} top functions identified")

    # Display top 5 bottlenecks for each
    print(f"\n{'='*60}")
    print("Top 5 Bottlenecks")
    print(f"{'='*60}")

    print("\nPyArrow:")
    for i, stat in enumerate(pyarrow_stats[:5], 1):
        print(f"  {i}. {stat.name}")
        print(f"     {stat.mean_cumtime:.4f}s ± {stat.std_cumtime:.4f}s")

    print("\nDuckDB:")
    for i, stat in enumerate(duckdb_stats[:5], 1):
        print(f"  {i}. {stat.name}")
        print(f"     {stat.mean_cumtime:.4f}s ± {stat.std_cumtime:.4f}s")

    # Save JSON reports
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    save_profile_json(
        pyarrow_stats,
        args.output_dir / 'pyarrow_profile.json',
        len(parquet_files),
        'pyarrow'
    )

    save_profile_json(
        duckdb_stats,
        args.output_dir / 'duckdb_profile.json',
        len(parquet_files),
        'duckdb'
    )

    # Get LLM analysis
    llm_analysis = None
    if not args.no_llm:
        print(f"\n{'='*60}")
        print("Requesting LLM analysis...")
        print(f"{'='*60}")

        # Read source code
        pyarrow_source_path = Path(__file__).parent.parent / 'detectors' / 'find_outliers_pyarrow.py'
        duckdb_source_path = Path(__file__).parent.parent / 'detectors' / 'find_outliers_duckdb.py'

        try:
            with open(pyarrow_source_path, 'r') as f:
                pyarrow_source = f.read()
            with open(duckdb_source_path, 'r') as f:
                duckdb_source = f.read()

            llm_analysis = get_llm_analysis(
                pyarrow_stats,
                duckdb_stats,
                pyarrow_source,
                duckdb_source
            )

            if llm_analysis:
                print("✓ LLM analysis completed")
            else:
                print("⚠ LLM analysis skipped (missing API key or dependencies)")

        except Exception as e:
            print(f"⚠ Failed to read source files or get LLM analysis: {e}")

    # Generate HTML report
    print(f"\n{'='*60}")
    print("Generating HTML report...")
    print(f"{'='*60}")

    generate_html_report(
        pyarrow_stats,
        duckdb_stats,
        args.output_dir / 'profiling_report.html',
        llm_analysis,
        len(parquet_files)
    )

    print(f"\n{'='*60}")
    print("✓ Profiling complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - pyarrow_profile.json")
    print(f"  - duckdb_profile.json")
    print(f"  - profiling_report.html")
    print(f"\nOpen profiling_report.html in your browser to view the analysis.")


if __name__ == '__main__':
    main()
