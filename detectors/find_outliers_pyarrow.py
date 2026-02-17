#!/usr/bin/env python3
"""
NYC Taxi Trip Analysis - Outlier Detection

PURPOSE:
--------
Detect outlier taxi trips that violate physics-based constraints. These outliers
likely represent data quality issues, meter errors, or data corruption.

OUTLIER DETECTION STRATEGY:
---------------------------
Two-phase approach to efficiently identify outliers:

Phase 1: Percentile-based filtering (default: 90th percentile)
  - Calculate percentile threshold on raw trip distance data
  - Focus analysis on trips above threshold (top 10% by distance)
  - Rationale: Outliers more likely in long-distance trips

Phase 2: Physics-based validation
  A trip in the top percentile is classified as an outlier if it violates ANY constraint:

  1. Distance bounds: < 0.1 miles OR > 800 miles
     - Rationale: Below 0.1 mi suggests stationary meter; above 800 mi exceeds
       theoretical maximum (10 hours @ 80 mph)

  2. Duration bounds: <= 0 hours OR > 10 hours
     - Rationale: Negative/zero duration indicates timestamp errors; above 10 hours
       exceeds NYC TLC regulatory limits (fatigued driving prevention rules:
       https://www.nyc.gov/site/tlc/about/fatigued-driving-prevention-frequently-asked-questions.page)

  3. Speed bounds: < 2.5 mph OR > 80 mph
     - Rationale: Below 2.5 mph suggests parked meter running; above 80 mph indicates
       data corruption or calculation errors

WHY PYARROW:
------------
- Columnar processing: Efficient memory usage and computation
- Native Parquet support: Fast I/O without format conversion
- Single dependency: Minimal installation complexity
- Compute functions: Built-in vectorized operations (pc.quantile, pc.invert, pc.filter)

INPUT:
------
- NYC Yellow Taxi Parquet files (2009-2025)
- Automatically handles column name variations across years

OUTPUT:
-------
- Parquet (default) or CSV file containing ONLY outliers from top percentile
- Sorted by distance (descending) to show most extreme cases first
- All original columns PLUS computed columns (duration, speed)
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


# ============================================================================
# RESULT DATA STRUCTURE
# ============================================================================

@dataclass
class DetectionResult:
    """
    Result from outlier detection.

    Attributes:
        outliers_table: PyArrow table containing only outlier records
        processing_time: Detection time in seconds (excluding I/O)
        stats: Statistics dictionary with counts, percentiles, etc.
    """
    outliers_table: pa.Table
    processing_time: float
    stats: Dict

    @property
    def outlier_count(self) -> int:
        """Number of outliers detected."""
        return len(self.outliers_table)


# ============================================================================
# COLUMN NAME RESOLUTION
# ============================================================================

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


def normalize_table_columns(table: pa.Table, col_map: Dict[str, str]) -> pa.Table:
    """
    Rename columns to standard names for consistent access.

    Args:
        table: Input table with original column names
        col_map: Mapping from logical to actual names

    Returns:
        Table with renamed columns
    """
    # Build rename mapping (original_name -> standard_name)
    rename_map = {
        col_map['distance']: 'trip_distance',
        col_map['pickup']: 'tpep_pickup_datetime',
        col_map['dropoff']: 'tpep_dropoff_datetime'
    }

    # Apply renames
    new_names = [rename_map.get(name, name) for name in table.column_names]
    return table.rename_columns(new_names)


# ============================================================================
# CONFIGURATION & DEFAULTS
# ============================================================================

DEFAULT_CONFIG = {
    'min_distance_miles': 0.1,      # ~2 NYC blocks minimum
    'max_distance_miles': 800,      # Theoretical max: 10h @ 80mph
    'min_speed_mph': 2.5,           # Below this suggests parked with meter running
    'max_speed_mph': 80,            # Realistic highway speed limit
    'max_trip_hours': 10,           # NYC TLC fatigued driving prevention rules: https://www.nyc.gov/site/tlc/about/fatigued-driving-prevention-frequently-asked-questions.page
    'percentile': 0.90              # Focus on top 10% of trips by distance
}


# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def find_outliers(table: pa.Table, config: Dict) -> Tuple[pa.Table, Dict]:
    """
    Detect outlier trips in the top percentile that violate physics-based constraints.

    Strategy:
    1. Calculate percentile threshold on raw distance data
    2. Filter to trips above percentile (top N%, likely outlier candidates)
    3. Within that subset, find trips that FAIL physics validation (actual outliers)

    Args:
        table: Input table with normalized column names
        config: Configuration dict with threshold values

    Returns:
        tuple: (outlier_table, statistics_dict)
    """
    total_rows = len(table)

    # Step 1: Calculate percentile threshold on raw data
    distance = table['trip_distance']
    percentile_threshold = pc.quantile(distance, q=config['percentile'])[0].as_py()

    # Step 2: Filter to trips above percentile (top N%)
    percentile_mask = pc.greater(distance, percentile_threshold)
    table_top_percentile = pc.filter(table, percentile_mask)
    num_top_percentile = len(table_top_percentile)

    # Step 3: Within top percentile, find outliers (physics validation failures)
    # Extract columns from top percentile subset
    distance = table_top_percentile['trip_distance']
    pickup = table_top_percentile['tpep_pickup_datetime']
    dropoff = table_top_percentile['tpep_dropoff_datetime']

    # Handle datetime columns that might be stored as strings (older parquet files)
    if pa.types.is_string(pickup.type) or pa.types.is_large_string(pickup.type):
        pickup = pc.strptime(pickup, format='%Y-%m-%d %H:%M:%S', unit='us')
    if pa.types.is_string(dropoff.type) or pa.types.is_large_string(dropoff.type):
        dropoff = pc.strptime(dropoff, format='%Y-%m-%d %H:%M:%S', unit='us')

    # Calculate trip duration in hours using Arrow compute
    # 1. Subtract timestamps to get duration
    duration = pc.subtract(dropoff, pickup)

    # 2. Cast duration to int64 to get microseconds
    duration_us = pc.cast(duration, pa.int64())

    # 3. Convert to hours (cast to float, then divide)
    duration_hours = pc.divide(
        pc.cast(duration_us, pa.float64()),
        3600.0 * 1_000_000.0  # microseconds to hours
    )

    # Calculate average speed
    avg_speed_mph = pc.divide(distance, duration_hours)

    # Add computed columns to top percentile table
    table_top_percentile = table_top_percentile.append_column('trip_duration_hours', duration_hours)
    table_top_percentile = table_top_percentile.append_column('avg_speed_mph', avg_speed_mph)

    # Build compound filter mask for VALID trips
    valid_mask = pc.and_(
        pc.and_(
            # Duration filters (valid)
            pc.greater(duration_hours, 0),
            pc.less_equal(duration_hours, config['max_trip_hours'])
        ),
        pc.and_(
            pc.and_(
                # Speed filters (valid)
                pc.greater_equal(avg_speed_mph, config['min_speed_mph']),
                pc.less_equal(avg_speed_mph, config['max_speed_mph'])
            ),
            pc.and_(
                # Distance filters (valid)
                pc.greater_equal(distance, config['min_distance_miles']),
                pc.less_equal(distance, config['max_distance_miles'])
            )
        )
    )

    # INVERT to get OUTLIERS (trips that FAIL validation within top percentile)
    outlier_mask = pc.invert(valid_mask)

    # Filter to get outliers
    outliers = pc.filter(table_top_percentile, outlier_mask)
    num_outliers = len(outliers)

    # Calculate statistics
    stats = {
        'total_rows': total_rows,
        'percentile': config['percentile'],
        'percentile_threshold': percentile_threshold,
        'num_top_percentile': num_top_percentile,
        'pct_top_percentile': (num_top_percentile / total_rows * 100) if total_rows > 0 else 0,
        'num_outliers': num_outliers,
        'pct_outliers_of_top': (num_outliers / num_top_percentile * 100) if num_top_percentile > 0 else 0,
        'pct_outliers_of_total': (num_outliers / total_rows * 100) if total_rows > 0 else 0
    }

    return outliers, stats


# ============================================================================
# OUTPUT
# ============================================================================

def export_results(
    table: pa.Table,
    output_format: str,
    output_path: Path = None
) -> Tuple[Path, float]:
    """
    Export outliers to Parquet or CSV.

    Args:
        table: PyArrow table to export (already sorted)
        output_format: 'parquet' or 'csv'
        output_path: Optional custom output path

    Returns:
        tuple: (output_path, export_time_seconds)
    """
    # Reorder: key columns first, then other columns
    key_cols = ['trip_distance', 'avg_speed_mph', 'trip_duration_hours',
                'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    other_cols = [c for c in table.column_names if c not in key_cols]
    table_reordered = table.select(key_cols + other_cols)

    # Determine output path
    if output_path is None:
        filename = f'outliers.{output_format}'
        output_path = Path(filename)
    else:
        output_path = Path(output_path)

    # Export based on format
    start = time.time()
    if output_format == 'parquet':
        pq.write_table(
            table_reordered,
            output_path,
            compression='snappy',
            use_dictionary=True
        )
    else:  # csv
        pa_csv.write_csv(table_reordered, output_path)
    elapsed = time.time() - start

    return output_path, elapsed


def print_outlier_insights(table: pa.Table):
    """Print analytical insights about detected outliers."""
    print("\n" + "="*60)
    print("OUTLIER INSIGHTS")
    print("="*60)

    if len(table) == 0:
        print("\n✓ No outliers detected - all trips pass validation!")
        return

    # Table is already sorted by distance descending
    print(f"\nTop 10 most extreme outliers (by distance):")

    # Get top 10 rows
    top10_count = min(10, len(table))
    top10_indices = pa.array(range(top10_count))
    top10 = pc.take(table, top10_indices)

    for i in range(len(top10)):
        dist = top10['trip_distance'][i].as_py()
        speed = top10['avg_speed_mph'][i].as_py()
        duration = top10['trip_duration_hours'][i].as_py()
        print(f"  {i+1:2d}. {dist:>8.2f} mi  "
              f"@ {speed:>7.1f} mph  "
              f"({duration:>7.2f} hours)")

    # Distribution statistics
    dist_col = table['trip_distance']
    speed_col = table['avg_speed_mph']
    duration_col = table['trip_duration_hours']

    print(f"\nOutlier distribution:")
    print(f"  Distance:  {pc.min(dist_col).as_py():>8.2f} - {pc.max(dist_col).as_py():>8.2f} miles")
    print(f"  Speed:     {pc.min(speed_col).as_py():>8.1f} - {pc.max(speed_col).as_py():>8.1f} mph")
    print(f"  Duration:  {pc.min(duration_col).as_py():>8.2f} - {pc.max(duration_col).as_py():>8.2f} hours")


# ============================================================================
# CORE API FUNCTION (for benchmarking and imports)
# ============================================================================

def detect_outliers_pyarrow(parquet_path: str, config: Dict) -> DetectionResult:
    """
    Detect outlier trips using PyArrow (main API for benchmarking).

    This function encapsulates the core detection workflow:
    1. Load parquet file and resolve column names
    2. Detect outliers using physics-based validation
    3. Sort by distance (descending)
    4. Return results with timing

    Args:
        parquet_path: Path to input Parquet file
        config: Configuration dict with thresholds (see DEFAULT_CONFIG)

    Returns:
        DetectionResult with outliers_table, processing_time, and stats

    Raises:
        ValueError: If required columns not found
        Exception: For file I/O or processing errors
    """
    # Load and normalize data
    schema = pq.read_schema(parquet_path)
    col_map = resolve_column_names(schema)
    table = pq.read_table(parquet_path, use_threads=True)
    table = normalize_table_columns(table, col_map)

    # Detect outliers (timed)
    start_time = time.time()
    outliers, stats = find_outliers(table, config)
    processing_time = time.time() - start_time

    # Sort by distance descending (most extreme first)
    sorted_indices = pc.sort_indices(outliers, sort_keys=[('trip_distance', 'descending')])
    outliers_sorted = pc.take(outliers, sorted_indices)

    return DetectionResult(
        outliers_table=outliers_sorted,
        processing_time=processing_time,
        stats=stats
    )


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect outlier NYC taxi trips that violate physics-based constraints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Parquet output
  python find_outliers.py yellow_tripdata_2025-01.parquet

  # CSV output
  python find_outliers.py data.parquet --output-format csv

  # Custom thresholds
  python find_outliers.py data.parquet --max-speed 70 --max-trip-hours 8

  # Custom output path
  python find_outliers.py data.parquet --output my_outliers.parquet

Outlier Criteria:
  A trip is an outlier if it violates ANY constraint:
  - Distance < 0.1 mi OR > 800 mi
  - Duration <= 0 hrs OR > 10 hrs
  - Speed < 2.5 mph OR > 80 mph
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input Parquet file'
    )

    parser.add_argument(
        '--min-distance',
        type=float,
        default=DEFAULT_CONFIG['min_distance_miles'],
        help=f'Minimum valid trip distance in miles (default: {DEFAULT_CONFIG["min_distance_miles"]})'
    )

    parser.add_argument(
        '--max-distance',
        type=float,
        default=DEFAULT_CONFIG['max_distance_miles'],
        help=f'Maximum valid trip distance in miles (default: {DEFAULT_CONFIG["max_distance_miles"]})'
    )

    parser.add_argument(
        '--min-speed',
        type=float,
        default=DEFAULT_CONFIG['min_speed_mph'],
        help=f'Minimum valid average speed in mph (default: {DEFAULT_CONFIG["min_speed_mph"]})'
    )

    parser.add_argument(
        '--max-speed',
        type=float,
        default=DEFAULT_CONFIG['max_speed_mph'],
        help=f'Maximum valid average speed in mph (default: {DEFAULT_CONFIG["max_speed_mph"]})'
    )

    parser.add_argument(
        '--max-trip-hours',
        type=float,
        default=DEFAULT_CONFIG['max_trip_hours'],
        help=f'Maximum valid trip duration in hours (default: {DEFAULT_CONFIG["max_trip_hours"]})'
    )

    parser.add_argument(
        '--percentile',
        type=float,
        default=DEFAULT_CONFIG['percentile'],
        help=f'Percentile threshold to focus analysis (default: {DEFAULT_CONFIG["percentile"]}, i.e., top {(1-DEFAULT_CONFIG["percentile"])*100:.0f}%%)'
    )

    parser.add_argument(
        '--output-format',
        type=str,
        choices=['parquet', 'csv'],
        default='parquet',
        help='Output format (default: parquet)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Custom output path (default: outliers.{format})'
    )

    return parser.parse_args()


def main():
    """Main execution function for CLI usage."""
    args = parse_args()

    # Configuration
    config = {
        'min_distance_miles': args.min_distance,
        'max_distance_miles': args.max_distance,
        'min_speed_mph': args.min_speed,
        'max_speed_mph': args.max_speed,
        'max_trip_hours': args.max_trip_hours,
        'percentile': args.percentile,
    }

    print("="*60)
    print("NYC TAXI - OUTLIER DETECTION")
    print("="*60)
    print(f"\nInput: {args.input_file}")
    print(f"Output format: {args.output_format}")
    print(f"\nAnalysis strategy:")
    print(f"  1. Focus on top {(1-config['percentile'])*100:.0f}% trips by distance (above p{int(config['percentile']*100)})")
    print(f"  2. Within that subset, find outliers violating constraints")
    print(f"\nValidation constraints:")
    print(f"  Distance: {config['min_distance_miles']} - {config['max_distance_miles']} miles")
    print(f"  Speed: {config['min_speed_mph']} - {config['max_speed_mph']} mph")
    print(f"  Duration: > 0 and <= {config['max_trip_hours']} hours")

    # Execute detection (timing starts here)
    print(f"\n{'='*60}")
    print("PROCESSING DATA")
    print("="*60)
    start_processing = time.time()

    try:
        # Use core API function
        result = detect_outliers_pyarrow(args.input_file, config)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    processing_time = time.time() - start_processing
    print(f"✓ Outlier detection complete ({processing_time:.2f}s)")

    # Print results
    stats = result.stats
    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60)
    print(f"Total trips: {stats['total_rows']:,}")
    print(f"Percentile threshold (p{int(stats['percentile']*100)}): {stats['percentile_threshold']:.2f} miles")
    print(f"Trips above p{int(stats['percentile']*100)}: {stats['num_top_percentile']:,} ({stats['pct_top_percentile']:.2f}%)")
    print(f"Outliers in top {(1-stats['percentile'])*100:.0f}%: {stats['num_outliers']:,} ({stats['pct_outliers_of_top']:.2f}% of top, {stats['pct_outliers_of_total']:.2f}% of total)")

    if stats['num_outliers'] == 0:
        print("\n✓ No outliers found in top percentile - all long trips pass validation!")
        print("   No output file generated.")
        return 0

    # Use already sorted outliers from result
    outliers_sorted = result.outliers_table

    # Export outliers
    print(f"\n{'='*60}")
    print("EXPORTING OUTLIERS")
    print("="*60)

    output_path, export_time = export_results(
        outliers_sorted,
        args.output_format,
        args.output
    )

    print(f"✓ Exported to: {output_path} ({export_time:.2f}s)")

    # Print insights (using sorted table)
    print_outlier_insights(outliers_sorted)

    # Performance summary
    total_time = processing_time + export_time
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"  Processing:          {processing_time:>7.2f}s")
    print(f"  Export ({args.output_format}):         {export_time:>7.2f}s")
    print(f"  {'─'*30}")
    print(f"  TOTAL:               {total_time:>7.2f}s")
    print("="*60)

    print("\n✅ Analysis complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
