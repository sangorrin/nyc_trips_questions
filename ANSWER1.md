# Question 1: PyArrow vs DuckDB Performance Analysis

## Question
Why do you think PyArrow is giving you better performance than DuckDB? What are the main factors?

## Answer

### Performance Summary

Based on benchmarking 50 parquet files (covering 2009-2025 NYC taxi data):

| Implementation | Mean Time (s) | Mean Memory (MB) | Speed Advantage |
|----------------|---------------|------------------|-----------------|
| **PyArrow**    | 0.159         | 16.33            | **6.8x faster** |
| **DuckDB**     | 1.085         | 34.66            | baseline        |

Both implementations produce **identical outlier counts**, confirming analytical correctness.

## Main Performance Factors

### 1. Zero-Copy Operations
**PyArrow's biggest advantage**. The outlier detection workflow is:
1. Read Parquet → Arrow Table (native format, no conversion)
2. Calculate percentile threshold using `pc.quantile()` on Arrow arrays
3. Filter using `pc.filter()` on Arrow arrays (zero-copy masking)
4. Write outliers → Parquet (native format, no conversion)

**DuckDB** must:
1. Read Parquet → Internal columnar format (conversion overhead)
2. Execute SQL query with CTE and subqueries (engine overhead)
3. Convert result → Arrow/Parquet (conversion overhead)

### 2. Task-Specific Efficiency
Our outlier detection is a **simple two-phase operation**:
- Phase 1: Calculate 90th percentile threshold
- Phase 2: Filter rows violating physics constraints

This plays to PyArrow's strengths:
- Direct use of `pc.quantile()` and `pc.filter()` compute functions
- No need for complex query optimization
- Minimal abstraction overhead

DuckDB's SQL engine adds overhead that doesn't provide value here:
- Query parsing and planning
- CTE (Common Table Expression) processing
- Join/aggregation optimization (not needed)
- Vectorization engine initialization

### 3. Columnar Operations Efficiency
The task involves:
- **Sequential column scans** (distance, duration, timestamps)
- **Simple arithmetic** (speed = distance / duration)
- **Comparison filters** (bounds checking)
- **Percentile calculation** (single statistical operation)

PyArrow's Arrow compute functions are **optimized specifically** for these operations:
```python
# PyArrow: Direct vectorized operations
threshold = pc.quantile(table['trip_distance'], 0.90)
mask = pc.greater(table['trip_distance'], threshold)
outliers = table.filter(mask)
```

DuckDB must translate SQL to execution plan:
```sql
-- DuckDB: SQL parsing + optimization + execution
WITH stats AS (
  SELECT PERCENTILE_CONT(0.90) FROM table
)
SELECT * FROM table WHERE trip_distance > (SELECT * FROM stats)
```

### 4. Memory Access Patterns
**PyArrow**: Operates directly on Arrow arrays in memory
- Contiguous memory layout for cache efficiency
- SIMD-friendly operations (vectorized CPU instructions)
- No memory allocation for intermediate results (zero-copy masking)

**DuckDB**: More sophisticated but heavier memory management
- More conservative allocation (higher baseline: 34.66 MB vs 16.33 MB)
- Internal buffer management for SQL execution
- Type system conversions between SQL and Arrow formats

### 5. File Format Integration
Both use Parquet, but differently:

**PyArrow**:
- Native Parquet reader/writer (same underlying C++ library)
- Arrow format = in-memory Parquet representation
- Column projection happens at read time (only load needed columns)

**DuckDB**:
- Parquet scanning optimized for SQL queries
- Must convert between Parquet metadata and SQL catalog
- Additional overhead for query result materialization

## When Performance Gap Is Largest

### Large Files (2009-2010 era: ~450-500 MB)
- PyArrow: **7-8x faster**
- Example: `yellow_tripdata_2009-01.parquet` (448 MB)
  - PyArrow: 0.42s
  - DuckDB: 3.03s

Zero-copy advantage scales with data size. DuckDB's fixed overhead (query planning, engine init) becomes more costly relative to processing time.

### Small Files (2020-2021 COVID era: ~20-50 MB)
- PyArrow: **2-3x faster**
- DuckDB's engine initialization overhead dominates

Gap narrows but PyArrow maintains consistent advantage.

## Memory Trade-offs

**PyArrow**: Lower average (16.33 MB) but occasional spikes
- Notable spikes: 418 MB (2009-01), 85 MB (2023-11)
- Caused by temporary Arrow array allocations during filtering

**DuckDB**: More predictable (34.66 MB average, consistent)
- SQL engine provides better memory governance
- Conservative allocation strategy prevents spikes

## Why This Task Favors PyArrow

The outlier detection produces **<1% output** (few outliers from millions of trips):
1. Must scan all rows (no index optimization possible)
2. Simple statistical calculation (percentile)
3. Simple filtering logic (bounds checking)
4. Minimal output materialization

This is **PyArrow's sweet spot**:
- Full scan → columnar efficiency shines
- Simple operations → no need for query optimization
- Small output → zero-copy filtering highly effective
- No joins/aggregations → DuckDB's SQL sophistication unused

## Bottom Line

For this specific outlier detection task, PyArrow wins because:
1. **Zero-copy operations** eliminate serialization overhead
2. **Direct Arrow compute functions** avoid SQL engine overhead
3. **Columnar efficiency** perfect for sequential scans + filtering
4. **Native Parquet integration** minimizes format conversions
5. **Task simplicity** makes SQL engine sophistication unnecessary overhead

DuckDB would be more competitive for:
- Complex multi-table joins
- Advanced aggregations and window functions
- Interactive SQL analysis where query flexibility matters
- Memory-constrained environments requiring predictable usage

For production outlier detection pipelines prioritizing speed, **PyArrow is the clear winner**.
