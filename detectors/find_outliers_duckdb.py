#!/usr/bin/env python3
"""
NYC Taxi Trip Analysis - Outlier Detection (DuckDB Implementation)

PURPOSE:
--------
Detect outlier taxi trips using DuckDB's SQL-based analytical engine.
Provides the same outlier detection as PyArrow but using SQL queries.

OUTLIER DETECTION STRATEGY:
---------------------------
Two-phase SQL-based approach:

Phase 1: Percentile-based filtering (default: 90th percentile)
  - Calculate percentile threshold using PERCENTILE_CONT
  - Focus analysis on trips above threshold (top 10% by distance)

Phase 2: Physics-based validation via WHERE clause
  A trip in the top percentile is classified as an outlier if it violates ANY constraint:
  1. Distance bounds: < 0.1 miles OR > 800 miles
  2. Duration bounds: <= 0 hours OR > 10 hours
  3. Speed bounds: < 2.5 mph OR > 80 mph

WHY DUCKDB:
-----------
- SQL interface: Familiar query language for data analysts
- Optimized engine: Vectorized execution for analytical workloads
- Native Parquet: Direct scanning without full data load
- Single query: Percentile calculation and filtering in one pass

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

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pa_csv
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


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
# CONFIGURATION & DEFAULTS
# ============================================================================

DEFAULT_CONFIG = {
    'min_distance_miles': 0.1,      # ~2 NYC blocks minimum
    'max_distance_miles': 800,      # Theoretical max: 10h @ 80mph
    'min_speed_mph': 2.5,           # Below this suggests parked with meter running
    'max_speed_mph': 80,            # Realistic highway speed limit
    'max_trip_hours': 10,           # NYC TLC fatigued driving prevention rules
    'percentile': 0.90              # Focus on top 10% of trips by distance
}


# ============================================================================
# COLUMN NAME RESOLUTION
# ============================================================================

def resolve_column_names(conn: duckdb.DuckDBPyConnection, parquet_path: str) -> Dict[str, str]:
    """
    Resolve column name variations across different years (2009-2025).

    Uses DuckDB's DESCRIBE to introspect parquet schema.

    Args:
        conn: DuckDB connection
        parquet_path: Path to parquet file

    Returns:
        Dict mapping logical names to actual column names

    Raises:
        ValueError: If required columns not found
    """
    # Get column names from parquet file
    result = conn.execute(f"DESCRIBE SELECT * FROM parquet_scan('{parquet_path}')").fetchall()
    col_names = [row[0] for row in result]
    col_names_lower = {col.lower(): col for col in col_names}

    col_map = {}

    # Distance column
    for variant in ['trip_distance', 'Trip_Distance']:
        if variant.lower() in col_names_lower:
            col_map['distance'] = col_names_lower[variant.lower()]
            break
    else:
        # Fallback: any column containing '_distance'
        for col in col_names:
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
        for col in col_names:
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
        for col in col_names:
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
            f"Available columns: {col_names}\n"
            f"Expected: trip_distance, tpep_pickup_datetime, tpep_dropoff_datetime (or variations)"
        )

    return col_map


# ============================================================================
# CORE API FUNCTION (for benchmarking and imports)
# ============================================================================

def detect_outliers_duckdb(parquet_path: str, config: Dict) -> DetectionResult:
    """
    Detect outlier trips using DuckDB (main API for benchmarking).

    This function uses SQL-based detection:
    1. Calculate percentile threshold
    2. Filter to top percentile trips
    3. Apply physics-based validation
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
    # Create DuckDB connection
    conn = duckdb.connect(':memory:')

    # Resolve column names
    col_map = resolve_column_names(conn, parquet_path)
    dist_col = col_map['distance']
    pickup_col = col_map['pickup']
    dropoff_col = col_map['dropoff']

    # Get total row count for statistics
    total_rows = conn.execute(
        f"SELECT COUNT(*) FROM parquet_scan('{parquet_path}')"
    ).fetchone()[0]

    # Calculate percentile threshold
    percentile_threshold = conn.execute(f"""
        SELECT PERCENTILE_CONT({config['percentile']})
        WITHIN GROUP (ORDER BY "{dist_col}")
        FROM parquet_scan('{parquet_path}')
    """).fetchone()[0]

    # Build SQL query for outlier detection
    # Use actual column names in the query
    start_time = time.time()

    query = f"""
    WITH top_percentile AS (
        SELECT *,
            EPOCH("{dropoff_col}"::TIMESTAMP - "{pickup_col}"::TIMESTAMP) / 3600.0 AS trip_duration_hours,
            "{dist_col}" / (EPOCH("{dropoff_col}"::TIMESTAMP - "{pickup_col}"::TIMESTAMP) / 3600.0) AS avg_speed_mph
        FROM parquet_scan('{parquet_path}')
        WHERE "{dist_col}" > {percentile_threshold}
    )
    SELECT * FROM top_percentile
    WHERE trip_duration_hours <= 0
       OR trip_duration_hours > {config['max_trip_hours']}
       OR avg_speed_mph < {config['min_speed_mph']}
       OR avg_speed_mph > {config['max_speed_mph']}
       OR "{dist_col}" < {config['min_distance_miles']}
       OR "{dist_col}" > {config['max_distance_miles']}
    ORDER BY "{dist_col}" DESC
    """

    # Execute query and get results as Arrow table
    result = conn.execute(query)
    # Convert RecordBatchReader to PyArrow Table
    outliers_table = result.fetch_arrow_table()
    processing_time = time.time() - start_time

    # Count trips in top percentile for statistics
    num_top_percentile = conn.execute(f"""
        SELECT COUNT(*)
        FROM parquet_scan('{parquet_path}')
        WHERE "{dist_col}" > {percentile_threshold}
    """).fetchone()[0]

    num_outliers = len(outliers_table)

    # Calculate statistics
    stats = {
        'total_rows': total_rows,
        'percentile': config['percentile'],
        'percentile_threshold': float(percentile_threshold),
        'num_top_percentile': num_top_percentile,
        'pct_top_percentile': (num_top_percentile / total_rows * 100) if total_rows > 0 else 0,
        'num_outliers': num_outliers,
        'pct_outliers_of_top': (num_outliers / num_top_percentile * 100) if num_top_percentile > 0 else 0,
        'pct_outliers_of_total': (num_outliers / total_rows * 100) if total_rows > 0 else 0
    }

    conn.close()

    return DetectionResult(
        outliers_table=outliers_table,
        processing_time=processing_time,
        stats=stats
    )


# ============================================================================
# OUTPUT
# ============================================================================

def export_results(
    table: pa.Table,
    output_format: str,
    output_path: Path = None
) -> tuple:
    """
    Export outliers to Parquet or CSV.

    Args:
        table: PyArrow table to export (already sorted)
        output_format: 'parquet' or 'csv'
        output_path: Optional custom output path

    Returns:
        tuple: (output_path, export_time_seconds)
    """
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
            table,
            output_path,
            compression='snappy',
            use_dictionary=True
        )
    else:  # csv
        pa_csv.write_csv(table, output_path)
    elapsed = time.time() - start

    return output_path, elapsed


def print_outlier_insights(table: pa.Table):
    """Print analytical insights about detected outliers."""
    import pyarrow.compute as pc

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
    for i in range(top10_count):
        # Find distance column (might have original name)
        dist_col = None
        for col_name in table.column_names:
            if 'distance' in col_name.lower():
                dist_col = col_name
                break

        dist = table[dist_col][i].as_py()
        speed = table['avg_speed_mph'][i].as_py()
        duration = table['trip_duration_hours'][i].as_py()
        print(f"  {i+1:2d}. {dist:>8.2f} mi  "
              f"@ {speed:>7.1f} mph  "
              f"({duration:>7.2f} hours)")

    # Distribution statistics
    speed_col = table['avg_speed_mph']
    duration_col = table['trip_duration_hours']

    print(f"\nOutlier distribution:")
    print(f"  Distance:  {pc.min(table[dist_col]).as_py():>8.2f} - {pc.max(table[dist_col]).as_py():>8.2f} miles")
    print(f"  Speed:     {pc.min(speed_col).as_py():>8.1f} - {pc.max(speed_col).as_py():>8.1f} mph")
    print(f"  Duration:  {pc.min(duration_col).as_py():>8.2f} - {pc.max(duration_col).as_py():>8.2f} hours")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Detect outlier NYC taxi trips using DuckDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Parquet output
  python find_outliers_duckdb.py yellow_tripdata_2025-01.parquet

  # CSV output
  python find_outliers_duckdb.py data.parquet --output-format csv

  # Custom thresholds
  python find_outliers_duckdb.py data.parquet --max-speed 70 --max-trip-hours 8

  # Custom output path
  python find_outliers_duckdb.py data.parquet --output my_outliers.parquet

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
    print("NYC TAXI - OUTLIER DETECTION (DuckDB)")
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

    # Execute detection
    print(f"\n{'='*60}")
    print("PROCESSING DATA")
    print("="*60)

    try:
        result = detect_outliers_duckdb(args.input_file, config)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"✓ Outlier detection complete ({result.processing_time:.2f}s)")

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

    # Export outliers (already sorted from SQL ORDER BY)
    print(f"\n{'='*60}")
    print("EXPORTING OUTLIERS")
    print("="*60)

    output_path, export_time = export_results(
        result.outliers_table,
        args.output_format,
        args.output
    )

    print(f"✓ Exported to: {output_path} ({export_time:.2f}s)")

    # Print insights
    print_outlier_insights(result.outliers_table)

    # Performance summary
    total_time = result.processing_time + export_time
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"  Outlier detection:   {result.processing_time:>7.2f}s")
    print(f"  Export ({args.output_format}):         {export_time:>7.2f}s")
    print(f"  {'─'*30}")
    print(f"  TOTAL:               {total_time:>7.2f}s")
    print("="*60)

    print("\n✅ Analysis complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
