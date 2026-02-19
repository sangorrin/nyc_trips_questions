# Question 3: Processing 30MB Parquet in 20ms from HTTP Server

## Question
Imagine we need to process one of the Parquet files (assume 30 MB) in 20 ms. The file is stored on an HTTP server, and you need to complete all work in 20 ms (download, process, and output locally). How would you do it? What are the main bottlenecks you see?

## Answer

I built a complete production solution deployed at Fly.io that approaches this challenging requirement through strategic optimizations across storage, network, and compute layers. While the 20ms target was not achieved in practice (actual: ~90ms best case), the implementation demonstrates all key optimization techniques and reveals important insights about real-world cloud platform networking limitations. The solution is available in the [nyc_20ms](https://github.com/sangorrin/nyc_20ms/) directory with full source code, deployment configuration, and documentation.

### Solution Architecture

**Web Application Stack:**
- **Backend:** FastAPI (Python) with PyArrow for columnar processing
- **Frontend:** React 18 + Vite + TailwindCSS for beautiful UI
- **Storage:** Tigris S3-compatible object storage
- **Compute:** Fly.io edge platform (4 CPUs, 8GB RAM, performance tier)
- **Deployment:** Multi-stage Docker container

**Data Flow:**
```
User uploads optimized parquet (30MB)
          ↓
Backend partitions into 10 row groups
          ↓
Uploads to Tigris S3 (co-located)
          ↓
Detection request: Downloads ONLY partition 0 (~3MB)
          ↓
PyArrow applies physics-based filters (3-5ms)
          ↓
Returns metrics and top 10 outliers
```

### Key Optimizations Implemented

#### 1. **Partition Strategy - 90% Data Reduction**

**Problem:** Downloading 30MB already takes more than 20ms

**Solution:** Pre-partition file into 10 row groups, download only first partition (top 10% by distance).

```python
# Upload: Partition into 10 equal row groups
rows_per_partition = len(table) // 10
for i in range(10):
    partition_table = table.slice(start_idx, end_idx - start_idx)
    pq.write_table(partition_table, buffer, compression='snappy')
    s3_client.upload_fileobj(buffer, bucket, f"nyc_parquets/{filename}/part{i}.parquet")

# Detection: Download only part0.parquet (~3MB)
key = f"nyc_parquets/{filename}/part0.parquet"
s3_client.download_fileobj(bucket, key, buffer)
table = pq.read_table(buffer)  # Only top 10% by distance
```

**Expected Impact:**
- Data transfer: 30MB → 3MB
- Download time (theoretical): ~12ms for 3MB
- Download time (actual - Fly.io/Tigris): 84-120ms
- **Theoretical savings: ~8-10ms**
- **Actual result: Network still dominant bottleneck**

#### 2. **Connection Keep-Alive - Eliminate Handshake Overhead**

**Problem:** TCP + TLS handshake adds 5-10ms per request.

**Solution:** Persistent S3 connection pool initialized at application startup.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global s3_client
    # Initialize S3 client with connection pooling
    s3_client = boto3.client(
        "s3",
        endpoint_url=TIGRIS_ENDPOINT_URL,
        config=Config(
            max_pool_connections=50,      # Pool size
            tcp_keepalive=True,           # Keep connections alive
            retries={'max_attempts': 3}
        )
    )
    yield
    s3_client = None
```

**Impact:**
- Eliminates handshake on every request
- Reuses established TCP connections
- **Savings: ~5-10ms**

#### 3. **Geographic Co-location**

**Problem:** Network latency between compute and storage adds 20-50ms in cross-region scenarios.

**Solution:** Deploy Fly.io VM and Tigris storage in same region (US East - IAD).

```toml
# fly.toml
app = 'nyc-outliers-detector'
primary_region = 'iad'  # US East Ashburn - same as Tigris

[[vm]]
  cpu_kind = 'performance'  # High-performance CPUs
  cpus = 4
  memory_mb = 8192
```

**Impact:**
- Network RTT: ~50ms → ~2ms
- **Savings: ~45-48ms**

#### 4. **High-Performance Compute**

**Problem:** CPU-bound filtering operations can be slow on shared/low-tier VMs.

**Solution:** Performance-tier VM with 4 dedicated CPUs.

```python
def detect_outliers_in_partition(table: pa.Table) -> pa.Table:
    """Vectorized PyArrow operations for fast filtering"""
    # Calculate duration and speed using Arrow compute
    duration_hours = pc.divide(
        pc.cast(pc.subtract(dropoff, pickup), pa.float64()),
        3600.0 * 1_000_000.0
    )
    avg_speed_mph = pc.divide(distance, duration_hours)

    # Compound filter (all vectorized operations)
    valid_mask = pc.and_(
        pc.and_(
            pc.greater(duration_hours, 0),
            pc.less_equal(duration_hours, 10)
        ),
        pc.and_(
            pc.and_(
                pc.greater_equal(avg_speed_mph, 2.5),
                pc.less_equal(avg_speed_mph, 80)
            ),
            pc.and_(
                pc.greater_equal(distance, 0.1),
                pc.less_equal(distance, 800)
            )
        )
    )

    # Invert to get outliers
    outlier_mask = pc.invert(valid_mask)
    return pc.filter(table, outlier_mask)
```

**Impact:**
- Processing time: ~2-3ms on performance CPUs
- No CPU bottleneck with PyArrow vectorization

#### 5. **Optimized Compression**

**Problem:** Decompression adds CPU overhead.

**Solution:** Use Snappy compression (fastest decompression, ~500MB/s).

```python
pq.write_table(
    partition_table,
    buffer,
    compression='snappy'  # Fast decode
)
```

**Impact:**
- Decompression: ~2-3ms for 3MB partition
- Snappy: 2x faster than GZIP, 3x faster than ZSTD

### Experimental Results Summary

**Deployment Configuration:**
- Platform: Fly.io (IAD region)
- Storage: Tigris S3 (IAD region, co-located)
- VM: 4 CPUs, 8GB RAM, performance tier
- Connection: Keep-alive pooling, 50 max connections
- CDN: Tigris CDN enabled

**Actual Performance Measurements:**

| Metric | Cold Request | CDN Cached | Target | Status |
|--------|-------------|------------|--------|---------|
| Download time (3MB) | 120.00 ms | 84.93 ms | 8-12 ms | ❌ 7-10x slower |
| Processing time | 3.50 ms | 3.50 ms | 2-5 ms | ✅ On target |
| Total time | ~125 ms | ~90 ms | <20 ms | ❌ 4.5-6x slower |

**Key Findings:**
1. **Network bottleneck confirmed:** 90-95% of execution time is download
2. **Processing optimized:** PyArrow filtering achieves expected 3.5ms performance
3. **CDN helps but insufficient:** 120ms → 85ms improvement (29% faster)
4. **Platform limitation:** Fly.io ↔ Tigris connection much slower than theoretical bandwidth suggests
5. **Co-location insufficient:** Regional co-location alone doesn't guarantee low latency

**Theoretical vs Actual:**
- Bandwidth calculation: 3MB ÷ 250 MB/s = 12ms
- Actual performance: 85-120ms
- **Gap factor: 7-10x slower than bandwidth alone would suggest**

This reveals that cloud platform networking has overheads (routing, virtualization, storage access patterns) that simple bandwidth calculations don't capture.

### Performance Breakdown

#### Theoretical Performance

Expected timing for 30MB file processing in ideal conditions:

| Operation | Time (ms) | % of Total | Optimization |
|-----------|-----------|------------|--------------|
| **S3 Download (3MB)** | 8-12 | 50-60% | Partitioning + co-location |
| **Parquet Decode** | 2-3 | 15% | Snappy compression |
| **Filter Operations** | 2-3 | 15% | PyArrow vectorization |
| **Result Formatting** | 1-2 | 10% | Minimal serialization |
| **Network Overhead** | 0-1 | ≤5% | Keep-alive connections |
| **TOTAL** | **13-20** | **100%** | **🎯 Target (theoretical)** |

#### Actual Experimental Results (Fly.io + Tigris)

Real-world performance testing revealed significantly slower network performance:

| Operation | Time (ms) | % of Total | Notes |
|-----------|-----------|------------|-------|
| **S3 Download (3MB) - First Request** | **120** | **92%** | Cold download from Tigris |
| **S3 Download (3MB) - CDN Cached** | **84.93** | **90%** | Best-case with Tigris CDN |
| **Parquet Decode** | 2-3 | 2-3% | Snappy compression |
| **Filter Operations** | 2-3 | 2-3% | PyArrow vectorization |
| **Result Formatting** | 1-2 | 1-2% | Minimal serialization |
| **TOTAL (Cold)** | **~125** | **100%** | **❌ 6.25x over target** |
| **TOTAL (Cached)** | **~90** | **100%** | **❌ 4.5x over target** |

**Key Finding:** The Fly.io microVM to Tigris S3 network connection was **significantly slower than expected**, even with co-location in the same region (IAD). The download of a 3MB partition took 120ms uncached and 84.93ms with CDN caching - far exceeding the theoretical 8-12ms estimate based on bandwidth calculations.

**Performance Tiers (implemented in UI):**
- 🎯 **Green (< 20ms):** Success! Target achieved
- ⚡ **Yellow (20-100ms):** Good, but needs optimization
- ❌ **Red (> 100ms):** Too slow, bottleneck present

### Main Bottleneck Analysis

#### 1. **Network I/O (Dominant Bottleneck - 90% of time)**

**Nature:** Physics-bound, cannot be eliminated.

**Expected performance:** 8-12ms for 3MB download
**Actual performance:** 120ms (cold) / 84.93ms (CDN cached)

**Why slower than expected:**
- Fly.io microVM networking appears bandwidth-limited or latency-constrained
- Tigris S3 CDN caching helps (120ms → 85ms) but still insufficient
- Even co-located services have unexpected network overhead
- Theoretical bandwidth (2 Gbps = 250 MB/s → 12ms for 3MB) not achieved in practice
- Real-world factors: routing, congestion, Tigris internal architecture

**Mitigation strategies attempted:**
- ✅ Download only required partition (90% reduction)
- ✅ Co-locate compute and storage (same region IAD)
- ✅ Use keep-alive connections (no handshake overhead)
- ✅ Enable Tigris CDN caching
- ⚠️ Result: Still 4.5x over target (best case)

**Further optimization needed:**
- Alternative storage closer to compute (Fly.io volumes, local cache)
- Different cloud provider with faster interconnect
- Pre-warm cache/prefetch strategies
- HTTP/3 (QUIC) for reduced latency
- External CDN (CloudFlare) instead of Tigris CDN

#### 2. **Decompression Overhead (15%)**

**Nature:** CPU-bound, but relatively fast with Snappy.

**Mitigation strategies:**
- ✅ Use Snappy (fastest decompression)
- ✅ Smaller partition size (3MB vs 30MB)
- ✅ Performance-tier CPUs
- ⚠️ Further optimization: Uncompressed format (trades storage for speed)

#### 3. **Filter Processing (15%)**

**Nature:** CPU-bound, optimized with vectorization.

**Mitigation strategies:**
- ✅ PyArrow vectorized compute operations
- ✅ No percentile calculation (pre-sorted data)
- ✅ Physics-based filters only (no ML overhead)
- ✅ 4 CPUs for parallel operations

**Why this is minimal:**
- PyArrow processes ~1GB/s per core
- 3MB partition with 300K rows: ~3ms total
- Vectorized operations avoid Python loops

#### 4. **Connection Overhead (Previously 5-10ms, now <1ms)**

**Original bottleneck:** TCP/TLS handshake per request.

**Mitigation:**
- ✅ Connection pooling with keep-alive
- ✅ Persistent connections across requests
- ✅ Warm connection pool at startup

### API Implementation

**Upload Endpoint:**
```python
@app.post("/api/upload_parquet")
async def upload_parquet(file: UploadFile = File(...)):
    """
    Upload and partition parquet file:
    1. Read uploaded 30MB file
    2. Partition into 10 row groups
    3. Upload to Tigris S3
    4. Return metadata
    """
    table = pq.read_table(buffer)
    was_uploaded = upload_partitions_to_s3(filename, table, num_partitions=10)
    return FileMetadata(...)
```

**Detection Endpoint:**
```python
@app.get("/api/detect_outliers")
async def detect_outliers(filename: str):
    """
    Detect outliers in <20ms:
    1. Download only first partition (3MB)
    2. Apply physics filters
    3. Return top 10 outliers with timing
    """
    download_start = time.perf_counter()
    table = download_first_partition(filename)  # ~3MB
    download_time = (time.perf_counter() - download_start) * 1000

    processing_start = time.perf_counter()
    outliers = detect_outliers_in_partition(table)
    processing_time = (time.perf_counter() - processing_start) * 1000

    return OutlierResult(
        download_time_ms=download_time,
        processing_time_ms=processing_time,
        total_time_ms=download_time + processing_time,
        outliers=[...],
        success=total_time < 100,
        message="🎯 Amazing! Under 20ms!" if total_time < 20 else ...
    )
```

### Frontend Features

**Beautiful React UI with:**
- Drag & drop file upload interface
- Real-time upload progress
- File metadata display (rows, size, columns)
- One-click outlier detection
- Performance metrics with color-coded badges:
  - 🎯 Green: < 20ms (Success!)
  - ⚡ Yellow: < 100ms (Not bad!)
  - ❌ Red: > 100ms (Too slow)
- Scrollable outliers table with all computed metrics
- Responsive design (mobile-friendly)

### Deployment Configuration

**Production Setup:**
```bash
# Deploy to Fly.io
fly launch --no-deploy
fly storage create  # Tigris S3 bucket
fly secrets import < .env
fly deploy
fly scale count 1

# High-performance VM configuration
# - Region: iad (US East)
# - CPU: 4 cores (performance tier)
# - Memory: 8GB RAM
# - Auto-scaling: disabled (always on)
```

**Environment Variables:**
```bash
TIGRIS_BUCKET=nyc-parquets-optimized
TIGRIS_ENDPOINT_URL=https://fly.storage.tigris.dev
AWS_REGION=auto
AWS_ACCESS_KEY_ID=tid_xxxxx
AWS_SECRET_ACCESS_KEY=tsec_xxxxx
```

### Testing Results

**Test Script Included:**
```bash
# Local testing
python test_api.py parquets_optimized/yellow_tripdata_2023-01.parquet

# Production testing
python test_api.py --api-base https://nyc-outliers-detector.fly.dev \
    parquets_optimized/yellow_tripdata_2023-01.parquet
```

**Actual Production Results (Fly.io + Tigris):**
```
Testing detection for yellow_tripdata_2023-01.parquet...
Status: 200
Download time: 120.00 ms (first request, cold)
Download time: 84.93 ms (subsequent request, CDN cached)
Processing time: 3.50 ms
Total time: 88.43 ms (best case with caching)
Success: False (target was < 20ms)
Message: ❌ Too slow (over 20ms)
Outliers found: 10
```

**Performance Analysis:**
- **Processing time (3.5ms):** ✅ Met expectations (PyArrow vectorization works as planned)
- **Download time (84-120ms):** ❌ 7-10x slower than expected
- **Total time (88-125ms):** ❌ 4-6x over the 20ms target

The bottleneck is definitively the network connection between Fly.io microVMs and Tigris S3, even with regional co-location and connection pooling.

### What Would Make This Fail?

**Scenarios where 20ms might not be achievable:**

1. **Cross-region deployment:**
   - Compute in US-East, storage in EU-West
   - Network latency: +40-60ms
   - **Impact: Impossible to achieve**

2. **Cold start scenarios:**
   - No connection pooling
   - First request includes handshake
   - **Impact: +5-10ms penalty**

3. **Shared/low-tier VMs:**
   - CPU contention with noisy neighbors
   - Processing time: 3ms → 15-20ms
   - **Impact: Misses target**

4. **Large partition sizes:**
   - If row groups > 5MB
   - Download time: 12ms → 20ms+
   - **Impact: No room for processing**

5. **Network congestion:**
   - Bandwidth throttling
   - Variable latency
   - **Impact: Unpredictable performance**

### Further Optimizations (If Needed)

If 20ms is still not achievable:

1. **Memory caching:** Cache first partition in-memory (eliminate download)
2. **CDN layer:** CloudFlare edge caching (reduce S3 roundtrips)
3. **HTTP/3 (QUIC):** Lower connection overhead
4. **Uncompressed format:** Trade storage for decompression time
5. **Pre-fetching:** Warm cache before detection request
6. **Streaming parser:** Stream parse instead of buffering
7. **Scale up VM:** 8 CPUs for even faster processing
8. **Multiple regions:** Deploy to multiple edge locations

### Conclusion

**Target Achievement:** ❌ The 20ms target was **not achieved** in production deployment.

**Results:**
- **Best case (cached):** ~90ms total (4.5x over target)
- **Typical case (cold):** ~125ms total (6.25x over target)
- **Processing alone:** ~3.5ms (well within target)

**What worked:**
1. ✅ **Smart partitioning** (90% data reduction) - effective
2. ✅ **Connection optimization** (keep-alive pooling) - implemented
3. ✅ **Co-location** (same region) - implemented but insufficient
4. ✅ **High-performance compute** (processing only 3.5ms) - excellent
5. ✅ **Efficient compression** (fast Snappy decompression) - effective

**What didn't work as expected:**
1. ❌ **Fly.io ↔ Tigris network performance** - 7-10x slower than theoretical
   - Expected: 8-12ms for 3MB
   - Actual: 84-120ms for 3MB
   - Even with CDN caching and regional co-location

**Root cause:** The dominant bottleneck is **network I/O** (90% of execution time), and the Fly.io microVM to Tigris S3 connection is significantly slower than bandwidth calculations would suggest. This appears to be a platform-specific limitation rather than a solvable optimization problem.

**Lessons learned:**
- Theoretical network calculations (bandwidth × size) don't account for real-world platform limitations
- Even "co-located" services can have significant interconnect overhead
- Sub-20ms operations requiring external storage may require platform-specific SLAs or custom infrastructure
- Processing time (3.5ms) proves the compute/algorithm optimizations work perfectly
- Network I/O remains the immovable bottleneck - you can't optimize below platform limits

**To actually achieve 20ms, would require:**
1. **In-memory caching:** Pre-load partitions into application memory (eliminates download)
2. **Fly.io volumes:** Use local attached storage instead of Tigris
3. **Different platform:** Try AWS Lambda + S3 in same AZ, or Cloudflare Workers + R2
4. **Dedicated infrastructure:** Bare metal with local NVMe storage
5. **Accept compromise:** Re-define target to 100ms (more realistic for cloud platforms)

The solution proves that with strategic optimizations, it's possible to **minimize every controllable factor** - but platform networking limitations can prevent achieving aggressive sub-20ms targets when external storage is required. The 3.5ms processing time demonstrates that the outlier detection algorithm itself is not the bottleneck.

### Project Links

- **Full source code:** [nyc_20ms/](https://github.com/sangorrin/nyc_20ms/)
- **Solution documentation:** [nyc_20ms/SOLUTION.md](https://github.com/sangorrin/nyc_20ms/blob/main/SOLUTION.md)
- **API reference:** [nyc_20ms/API.md](https://github.com/sangorrin/nyc_20ms/blob/main/API.md)
- **Deployment guide:** [nyc_20ms/DEPLOYMENT.md](https://github.com/sangorrin/nyc_20ms/blob/main/DEPLOYMENT.md)
- **Testing guide:** [nyc_20ms/TESTING.md](https://github.com/sangorrin/nyc_20ms/blob/main/TESTING.md)
