#!/usr/bin/env python3
"""
Parquet Optimization Script for NYC Taxi Data

Optimizes parquet files for efficient outlier detection by:
1. Normalizing column names to standard format
2. Fixing datetime column types (string -> timestamp[us])
3. Sorting by trip_distance descending
4. Creating exactly 10 row groups (row group 0 = top 10% by distance)

This structure enables optimized detectors to read only the first row group,
eliminating the need for percentile calculation, type checking, and full table scans.

Usage:
    # Optimize 20 sample files
    python scripts/optimize_parquets.py --samples 20

    # Optimize all files from a specific directory
    python scripts/optimize_parquets.py --parquets-path parquets/

    # Specify custom output directory
    python scripts/optimize_parquets.py --output-path parquets_optimized/
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from tqdm import tqdm


def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract YYYY-MM date from parquet filename like yellow_tripdata_2013-05.parquet"""
    match = re.search(r'(\d{4})-(\d{2})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return None


def get_parquet_files(parquets_path: Path, samples: Optional[int] = None) -> List[Path]:
    """
    Get parquet files, optionally sampled evenly across the date range.

    Args:
        parquets_path: Directory containing parquet files
        samples: Number of files to sample (None = all files)

    Returns:
        List of parquet file paths, sorted chronologically
    """
    parquet_files = sorted(parquets_path.glob("*.parquet"))

    if not parquet_files:
        return []

    if samples is None or samples >= len(parquet_files):
        return parquet_files

    # Distribute samples evenly across date range
    total_files = len(parquet_files)
    indices = [int(i * total_files / samples) for i in range(samples)]
    return [parquet_files[i] for i in indices]


def resolve_column_names(schema: pa.Schema) -> Dict[str, str]:
    """
    Resolve column name variations across different years (2009-2025).

    Known variations:
    - distance: 'trip_distance', 'Trip_Distance'
    - pickup: 'tpep_pickup_datetime', 'Trip_Pickup_DateTime', 'pickup_datetime'
    - dropoff: 'tpep_dropoff_datetime', 'Trip_Dropoff_DateTime', 'dropoff_datetime'

    Args:
        schema: PyArrow schema of the table

    Returns:
        Dict mapping logical names to actual column names

    Raises:
        ValueError: If required columns not found
    """
    col_map = {}
    col_names_lower = {col.lower(): col for col in schema.names}

    # Distance column
    for variant in ['trip_distance', 'Trip_Distance']:
        if variant.lower() in col_names_lower:
            col_map['distance'] = col_names_lower[variant.lower()]
            break
    else:
        # Fallback: any column containing '_distance'
        for col in schema.names:
            if '_distance' in col.lower():
                col_map['distance'] = col
                break

    # Pickup datetime column
    for variant in ['tpep_pickup_datetime', 'Trip_Pickup_DateTime', 'pickup_datetime']:
        if variant.lower() in col_names_lower:
            col_map['pickup'] = col_names_lower[variant.lower()]
            break
    else:
        # Fallback: any column containing 'pickup'
        for col in schema.names:
            if 'pickup' in col.lower() and 'datetime' in col.lower():
                col_map['pickup'] = col
                break

    # Dropoff datetime column
    for variant in ['tpep_dropoff_datetime', 'Trip_Dropoff_DateTime', 'dropoff_datetime']:
        if variant.lower() in col_names_lower:
            col_map['dropoff'] = col_names_lower[variant.lower()]
            break
    else:
        # Fallback: any column containing 'dropoff'
        for col in schema.names:
            if 'dropoff' in col.lower() and 'datetime' in col.lower():
                col_map['dropoff'] = col
                break

    # Verify all required columns found
    missing = []
    for logical_name in ['distance', 'pickup', 'dropoff']:
        if logical_name not in col_map:
            missing.append(logical_name)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {schema.names}\n"
            f"Expected: trip_distance, tpep_pickup_datetime, tpep_dropoff_datetime (or variations)"
        )

    return col_map


def optimize_parquet_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Optimize a single parquet file.

    Args:
        input_path: Input parquet file path
        output_path: Output parquet file path

    Returns:
        Dict with optimization statistics
    """
    # Read schema and resolve column names
    schema = pq.read_schema(input_path)
    col_map = resolve_column_names(schema)

    # Read full table
    table = pq.read_table(input_path, use_threads=True)

    # Get original file size and row count
    original_size_mb = input_path.stat().st_size / 1024 / 1024
    total_rows = len(table)

    # Normalize column names to standard format
    rename_map = {
        col_map['distance']: 'trip_distance',
        col_map['pickup']: 'tpep_pickup_datetime',
        col_map['dropoff']: 'tpep_dropoff_datetime'
    }
    new_names = [rename_map.get(name, name) for name in table.column_names]
    table = table.rename_columns(new_names)

    # Fix datetime column types (convert strings to proper timestamps)
    # This eliminates the need for type checking in the detector
    pickup_col = table['tpep_pickup_datetime']
    dropoff_col = table['tpep_dropoff_datetime']

    if pa.types.is_string(pickup_col.type) or pa.types.is_large_string(pickup_col.type):
        pickup_col = pc.strptime(pickup_col, format='%Y-%m-%d %H:%M:%S', unit='us')
        table = table.set_column(table.schema.get_field_index('tpep_pickup_datetime'), 'tpep_pickup_datetime', pickup_col)

    if pa.types.is_string(dropoff_col.type) or pa.types.is_large_string(dropoff_col.type):
        dropoff_col = pc.strptime(dropoff_col, format='%Y-%m-%d %H:%M:%S', unit='us')
        table = table.set_column(table.schema.get_field_index('tpep_dropoff_datetime'), 'tpep_dropoff_datetime', dropoff_col)

    # Sort by trip_distance descending
    sorted_indices = pc.sort_indices(table, sort_keys=[('trip_distance', 'descending')])
    table = pc.take(table, sorted_indices)

    # Calculate row group size for exactly 10 row groups
    # Use ceiling division to ensure all rows fit within 10 groups (no 11th partial group)
    row_group_size = (total_rows + 9) // 10

    # Ensure we have at least some reasonable row group size
    if row_group_size < 1000:
        row_group_size = 1000

    # Write optimized parquet file
    # Note: We keep write_statistics=True (default) as statistics are useful
    # for query optimization even though our detector always reads row group 0
    # The sorting_columns parameter records that data is sorted by trip_distance descending
    # Convert sort order to SortingColumn objects
    sorting_columns = pq.SortingColumn.from_ordering(
        table.schema,
        [('trip_distance', 'descending')]
    )

    pq.write_table(
        table,
        output_path,
        row_group_size=row_group_size,
        compression='snappy',
        use_dictionary=True,
        write_statistics=True,
        sorting_columns=sorting_columns
    )

    # Get output file size
    output_size_mb = output_path.stat().st_size / 1024 / 1024

    return {
        'filename': input_path.name,
        'total_rows': total_rows,
        'row_group_size': row_group_size,
        'num_row_groups': (total_rows + row_group_size - 1) // row_group_size,  # Ceiling division
        'original_size_mb': round(original_size_mb, 2),
        'optimized_size_mb': round(output_size_mb, 2),
        'size_ratio': round(output_size_mb / original_size_mb, 3) if original_size_mb > 0 else 1.0,
        'success': True,
        'error': None
    }


def main():
    """Main optimization orchestration."""
    parser = argparse.ArgumentParser(
        description="Optimize parquet files for efficient outlier detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--parquets-path",
        type=Path,
        default=Path("parquets"),
        help="Directory containing input parquet files (default: parquets)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of files to optimize (evenly distributed across dates). Default: all files"
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=Path,
        default=Path("parquets_optimized"),
        help="Output directory for optimized parquet files (default: parquets_optimized)"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.parquets_path.exists():
        print(f"Error: Input directory not found: {args.parquets_path}", file=sys.stderr)
        return 1

    # Create output directory
    if not args.output_path.exists():
        args.output_path.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {args.output_path}")

    # Get parquet files
    print(f"Scanning for parquet files in {args.parquets_path}...")
    parquet_files = get_parquet_files(args.parquets_path, args.samples)

    if not parquet_files:
        print("Error: No parquet files found", file=sys.stderr)
        return 1

    print(f"Found {len(parquet_files)} files to optimize")
    if args.samples and args.samples < len(list(args.parquets_path.glob("*.parquet"))):
        print(f"  (sampled {args.samples} files evenly across date range)")

    # Optimize files
    print("\nStarting optimization...\n")
    results = []

    # Create progress iterator with tqdm
    for parquet_file in tqdm(parquet_files, desc="Optimizing", unit="file"):
        output_file = args.output_path / parquet_file.name

        try:
            result = optimize_parquet_file(parquet_file, output_file)
            results.append(result)

        except Exception as e:
            error_result = {
                'filename': parquet_file.name,
                'total_rows': None,
                'row_group_size': None,
                'num_row_groups': None,
                'original_size_mb': None,
                'optimized_size_mb': None,
                'size_ratio': None,
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}"
            }
            results.append(error_result)

    # Count successes and failures
    successes = sum(1 for r in results if r['success'])
    failures = sum(1 for r in results if not r['success'])

    print(f"\n{'='*80}")
    print(f"Optimization completed: {successes} successes, {failures} failures")
    print(f"{'='*80}\n")

    if failures > 0:
        print("Failed optimizations:")
        for result in results:
            if not result['success']:
                print(f"  ✗ {result['filename']}: {result['error']}")
        print()

    # Print summary statistics
    if successes > 0:
        successful_results = [r for r in results if r['success']]

        total_original_size = sum(r['original_size_mb'] for r in successful_results)
        total_optimized_size = sum(r['optimized_size_mb'] for r in successful_results)
        avg_row_groups = sum(r['num_row_groups'] for r in successful_results) / len(successful_results)

        print("Optimization Summary:")
        print(f"{'='*80}")
        print(f"Files processed:        {successes}")
        print(f"Total original size:    {total_original_size:,.2f} MB")
        print(f"Total optimized size:   {total_optimized_size:,.2f} MB")
        print(f"Size ratio:             {total_optimized_size/total_original_size:.3f}x")
        print(f"Average row groups:     {avg_row_groups:.1f} per file")
        print(f"{'='*80}\n")

        # Show per-file details
        print("Per-File Details:")
        print(f"{'='*80}")
        print(f"{'Filename':<40} {'Rows':>10} {'RG Size':>10} {'#RGs':>6} {'Orig MB':>10} {'Opt MB':>10}")
        print(f"{'-'*80}")

        for result in successful_results:
            print(f"{result['filename']:<40} "
                  f"{result['total_rows']:>10,} "
                  f"{result['row_group_size']:>10,} "
                  f"{result['num_row_groups']:>6} "
                  f"{result['original_size_mb']:>10.2f} "
                  f"{result['optimized_size_mb']:>10.2f}")

        print(f"{'='*80}\n")

    print(f"✅ Optimization complete! Optimized files saved to: {args.output_path}")
    print()

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
