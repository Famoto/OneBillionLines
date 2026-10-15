#![feature(portable_simd)]

use colored::Colorize;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::simd::cmp::SimdPartialEq;
use std::simd::Simd;
use std::sync::{mpsc, Arc, Mutex};
use xxhash_rust::xxh3::xxh3_64;

type V = i32;

#[derive(Clone, Copy, Debug)]
struct Record {
    count: u32,
    min: V,
    max: V,
    sum: i64,
}

impl Default for Record {
    fn default() -> Self {
        Self {
            count: 0,
            min: V::MAX,
            max: V::MIN,
            sum: 0,
        }
    }
}

impl Record {
    #[inline(always)]
    fn add(&mut self, value: V) {
        self.count += 1;
        self.sum += i64::from(value);
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.sum += other.sum;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    fn avg(&self) -> V {
        let count = i64::from(self.count);
        let quotient = self.sum.div_euclid(count);
        let remainder = self.sum.rem_euclid(count);
        // Round in tenths, with ties toward positive infinity, including negatives.
        (quotient + i64::from(2 * remainder >= count)) as V
    }
}

fn format_val(value: V) -> String {
    format!("{:.1}", f64::from(value) / 10.0)
}

/// Parse a measurement with one fractional digit into tenths.
#[inline(always)]
fn parse(mut value: &[u8]) -> V {
    if value.ends_with(b"\r") {
        value = &value[..value.len() - 1];
    }
    let negative = value.starts_with(b"-");
    if negative {
        value = &value[1..];
    }
    let result = match value {
        [a @ b'0'..=b'9', b'.', b @ b'0'..=b'9'] => V::from(a - b'0') * 10 + V::from(b - b'0'),
        [a @ b'0'..=b'9', b @ b'0'..=b'9', b'.', c @ b'0'..=b'9'] => {
            V::from(a - b'0') * 100 + V::from(b - b'0') * 10 + V::from(c - b'0')
        }
        _ => {
            let (whole, fraction) =
                value.split_at(value.len().checked_sub(2).expect("empty measurement"));
            assert!(
                matches!(fraction, [b'.', b'0'..=b'9']) && !whole.is_empty(),
                "invalid measurement"
            );
            let mut result = 0;
            for &digit in whole {
                assert!(digit.is_ascii_digit(), "invalid measurement");
                result = result * 10 + V::from(digit - b'0');
            }
            result * 10 + V::from(fraction[1] - b'0')
        }
    };
    if negative {
        -result
    } else {
        result
    }
}

const L: usize = 32;
type S = Simd<u8, L>;

/// Scan bounded, unaligned SIMD blocks, then every remaining byte.
fn iter_lines<'a>(data: &'a [u8], mut callback: impl FnMut(&'a [u8], &'a [u8])) {
    let mut start = 0;
    let mut separator = None;
    let mut delimiter = |position: usize, byte: u8| match byte {
        b';' => {
            assert!(separator.is_none(), "multiple separators in a record");
            separator = Some(position);
        }
        b'\n' => {
            let sep = separator.take().expect("record missing separator");
            callback(&data[start..sep], &data[sep + 1..position]);
            start = position + 1;
        }
        _ => {}
    };

    let mut blocks = data.chunks_exact(L);
    for (index, block) in blocks.by_ref().enumerate() {
        let bytes = S::from_slice(block);
        let mut matches =
            (bytes.simd_eq(S::splat(b';')) | bytes.simd_eq(S::splat(b'\n'))).to_bitmask();
        while matches != 0 {
            let offset = matches.trailing_zeros() as usize;
            delimiter(index * L + offset, block[offset]);
            matches &= matches - 1;
        }
    }
    let remainder = blocks.remainder();
    let base = data.len() - remainder.len();
    for (offset, &byte) in remainder.iter().enumerate() {
        delimiter(base + offset, byte);
    }
    // Accept a final record without a newline too.
    if start < data.len() {
        let sep = separator.expect("record missing separator");
        callback(&data[start..sep], &data[sep + 1..]);
    }
}

struct Station {
    key: u64,
    name: Box<[u8]>,
    record: Record,
}

// Bucket indices remain valid when the dense station vector reallocates.
// Empty buckets are represented independently of hashes, so every u64 key is valid.
struct HashTable {
    buckets: Vec<usize>,
    stations: Vec<Station>,
}

impl HashTable {
    fn new() -> Self {
        Self {
            buckets: vec![usize::MAX; 1024],
            stations: Vec::new(),
        }
    }

    fn grow(&mut self) {
        self.buckets = vec![usize::MAX; self.buckets.len() * 2];
        let mask = self.buckets.len() - 1;
        for (index, station) in self.stations.iter().enumerate() {
            let mut bucket = station.key as usize & mask;
            while self.buckets[bucket] != usize::MAX {
                bucket = (bucket + 1) & mask;
            }
            self.buckets[bucket] = index;
        }
    }

    #[inline(always)]
    fn record_mut(&mut self, key: u64, name: &[u8]) -> &mut Record {
        if self.stations.len() * 4 >= self.buckets.len() * 3 {
            self.grow();
        }
        let mask = self.buckets.len() - 1;
        let mut bucket = key as usize & mask;
        loop {
            let index = self.buckets[bucket];
            if index == usize::MAX {
                let index = self.stations.len();
                self.stations.push(Station {
                    key,
                    name: name.into(),
                    record: Record::default(),
                });
                self.buckets[bucket] = index;
                return &mut self.stations[index].record;
            }
            let station = &self.stations[index];
            // Compare the full name even when two stations share a hash.
            if station.key == key && station.name.as_ref() == name {
                return &mut self.stations[index].record;
            }
            bucket = (bucket + 1) & mask;
        }
    }

    fn len(&self) -> usize {
        self.stations.len()
    }
}

fn aggregate(data: &[u8]) -> HashTable {
    let mut stations = HashTable::new();
    aggregate_into(data, &mut stations);
    stations
}

fn aggregate_into(data: &[u8], stations: &mut HashTable) {
    iter_lines(data, |name, value| {
        stations.record_mut(xxh3_64(name), name).add(parse(value));
    });
}

fn split_chunks(data: &[u8], count: usize) -> Vec<&[u8]> {
    let count = count.max(1).min(data.len().max(1));
    let size = data.len().div_ceil(count);
    let mut chunks = Vec::with_capacity(count);
    let mut start = 0;
    for index in 1..count {
        let target = size * index;
        if target >= data.len() {
            break;
        }
        let end = data[target..]
            .iter()
            .position(|&byte| byte == b'\n')
            .map_or(data.len(), |offset| target + offset + 1);
        if end > start {
            chunks.push(&data[start..end]);
            start = end;
        }
    }
    if start < data.len() {
        chunks.push(&data[start..]);
    }
    chunks
}

fn aggregate_parallel(data: &[u8], workers: usize) -> HashTable {
    if workers <= 1 || data.len() < 1024 * 1024 {
        return aggregate(data);
    }
    let chunks = split_chunks(data, workers);
    std::thread::scope(|scope| {
        let handles = chunks
            .into_iter()
            .map(|chunk| scope.spawn(move || aggregate(chunk)))
            .collect::<Vec<_>>();
        let mut result = HashTable::new();
        for handle in handles {
            for station in handle.join().expect("worker failed").stations {
                result
                    .record_mut(station.key, &station.name)
                    .merge(&station.record);
            }
        }
        result
    })
}

// A read may end in the middle of a record; extend it through the next newline.
fn read_chunk(reader: &mut impl BufRead, buffer: &mut Vec<u8>, size: usize) -> io::Result<bool> {
    buffer.clear();
    reader.take(size as u64).read_to_end(buffer)?;
    if buffer.is_empty() {
        return Ok(false);
    }
    if !buffer.ends_with(b"\n") {
        reader.read_until(b'\n', buffer)?;
    }
    Ok(true)
}

fn aggregate_streaming(
    mut reader: impl BufRead,
    workers: usize,
    chunk_size: usize,
) -> io::Result<HashTable> {
    if chunk_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "chunk size must be positive",
        ));
    }
    let workers = workers.max(1);
    let pool_size = workers + 2;
    let (jobs_tx, jobs_rx) = mpsc::sync_channel::<Vec<u8>>(workers);
    let (free_tx, free_rx) = mpsc::sync_channel::<Vec<u8>>(pool_size);
    for _ in 0..pool_size {
        free_tx
            .send(Vec::with_capacity(chunk_size.saturating_add(1024)))
            .expect("buffer pool disconnected");
    }
    let jobs_rx = Arc::new(Mutex::new(jobs_rx));

    std::thread::scope(|scope| {
        let handles = (0..workers)
            .map(|_| {
                let jobs_rx = Arc::clone(&jobs_rx);
                let free_tx = free_tx.clone();
                scope.spawn(move || {
                    let mut stations = HashTable::new();
                    loop {
                        // Release the queue lock before scanning the buffer.
                        let job = jobs_rx.lock().expect("work queue poisoned").recv();
                        let Ok(buffer) = job else {
                            break;
                        };
                        aggregate_into(&buffer, &mut stations);
                        free_tx.send(buffer).expect("buffer pool disconnected");
                    }
                    stations
                })
            })
            .collect::<Vec<_>>();
        drop(jobs_rx);
        drop(free_tx);

        let read_result = (|| -> io::Result<()> {
            loop {
                let mut buffer = free_rx
                    .recv()
                    .map_err(|_| io::Error::other("all workers stopped"))?;
                if !read_chunk(&mut reader, &mut buffer, chunk_size)? {
                    break;
                }
                jobs_tx
                    .send(buffer)
                    .map_err(|_| io::Error::other("all workers stopped"))?;
            }
            Ok(())
        })();
        drop(jobs_tx);

        let mut result = HashTable::new();
        for handle in handles {
            for station in handle.join().expect("worker failed").stations {
                result
                    .record_mut(station.key, &station.name)
                    .merge(&station.record);
            }
        }
        read_result?;
        Ok(result)
    })
}
/* 
fn write_results(mut output: impl Write, stations: &HashTable) -> io::Result<()> {
    let mut sorted = stations.stations.iter().collect::<Vec<_>>();
    sorted.sort_unstable_by_key(|station| station.name.as_ref());
    write!(output, "{{")?;
    for (index, station) in sorted.into_iter().enumerate() {
        let name = &station.name;
        let record = station.record;
        if index != 0 {
            write!(output, ", ")?;
        }
        write!(
            output,
            "{}={}/{}/{}",
            std::str::from_utf8(name).expect("invalid station name"),
            format_val(record.min),
            format_val(record.avg()),
            format_val(record.max)
        )?;
    }
    writeln!(output, "}}")
}
*/

fn main() {
    let start = std::time::Instant::now();
    let read_all = std::env::args().any(|arg| arg == "--read-all");
    let filename = std::env::args()
        .skip(1)
        .find(|arg| arg != "--read-all")
        .unwrap_or_else(|| "measurements.txt".to_string());
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    let stations = if read_all {
        let data = std::fs::read(filename).expect("failed to read measurements");
        eprintln!("read: {:.1?}", start.elapsed());
        let scan_start = std::time::Instant::now();
        let stations = aggregate_parallel(&data, workers);
        eprintln!("scan ({workers} threads): {:.1?}", scan_start.elapsed());
        stations
    } else {
        let workers = workers.saturating_sub(1).max(1);
        let file = std::fs::File::open(filename).expect("failed to open measurements");
        let stations = aggregate_streaming(BufReader::new(file), workers, 8 * 1024 * 1024)
            .expect("failed to read measurements");
        eprintln!("read + scan ({workers} workers): {:.1?}", start.elapsed());
        stations
    };
   let mut output = io::BufWriter::new(io::stdout().lock());
    //write_results(&mut output, &stations).expect("failed to write statistics");
    output.flush().expect("failed to flush statistics");
    eprintln!("cities {}", stations.len());
    eprintln!(
        "rows {}",
        stations
            .stations
            .iter()
            .map(|station| u64::from(station.record.count))
            .sum::<u64>()
    );
    eprintln!(
        "total: {}",
        format!("{:>5.1?}", start.elapsed()).bold().green()
    );
}
