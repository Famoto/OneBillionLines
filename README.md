# One Billion Row Challenge — Rust

Aggregates `station;temperature` measurements into per-station minimum, maximum,
sum, and count, with a mean calculation available for formatted output.

## Runtime and hardware

A single release-mode run on **16 October 2026** processed **1,000,000,000 rows**
across **8,843 stations** in **5.1 seconds**, including reading and aggregation.

| Item | Details |
| --- | --- |
| CPU | Intel Core Ultra 7 155H — 16 cores, 22 logical processors |
| RAM | Approximately 64 GiB (63.5 GiB reported by Windows) |
| Mode | Default streaming mode, 21 aggregation workers |

```text
read + scan (21 workers): 5.1s
cities 8843
rows 1000000000
total:  5.1s
```

This is one local measurement; filesystem cache state was not controlled.
The timer starts inside `main`, excludes compilation, and includes reading,
aggregation, and summary output. Per-station statistics output is currently
commented out in `src/main.rs`.

## Techniques

- **SIMD delimiter scanning:** checks 32 bytes at a time for `;` and newline,
  then visits matching positions using a bitmask. Remaining bytes are scanned
  individually.
- **Integer temperature parsing:** converts bytes directly into integer tenths,
  with fast paths for common measurement lengths and an `i64` sum. Floating-point
  conversion is only needed for formatting.
- **Parallel streaming:** one reader feeds 8 MiB chunks to workers through a
  bounded queue. Chunks extend to a newline so records stay intact; reading and
  aggregation can overlap.
- **Reusable buffers:** a pool of `workers + 2` buffers bounds input-buffer memory
  to roughly 184 MiB on this machine, apart from record-boundary extensions and
  per-worker station tables.
- **Worker-local aggregation:** each worker updates its own station table without
  locking each record. Partial counts, sums, minima, and maxima are merged once
  workers finish.
- **Custom hash table:** XXH3 hashes station names into a growing table with linear
  probing and a dense station vector. Full name comparisons handle hash collisions;
  names are copied only when first inserted into a worker's table.

## Run

Requires a nightly Rust toolchain for `portable_simd` and an input file containing
one `station;temperature` record per line, with one fractional digit. LF, CRLF,
and a final record without a newline are supported.

```powershell
cargo +nightly build --release
.\target\release\one-billion-row-challenge.exe measurements.txt

# Optional: load the entire file before scanning
.\target\release\one-billion-row-challenge.exe --read-all measurements.txt
```

The filename defaults to `measurements.txt`. The dataset is excluded from Git.
