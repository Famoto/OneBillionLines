#![feature(portable_simd)]

use colored::Colorize;
use fxhash::FxHashMap;
use std::simd::cmp::SimdPartialEq;
use std::simd::Simd;
use std::{env::args, io::Read};
use xxhash_rust::xxh3::xxh3_64;

// ─────────────────────────────────────────────────────────────── Types

type V = i32;

#[derive(Clone, Copy, Debug)]
#[repr(align(64))]
struct Record {
    count: u32,
    min: V,
    max: V,
    sum: V,
}

impl Record {
    const fn default() -> Self {
        Self {
            count: 0,
            min: i32::MAX,
            max: i32::MIN,
            sum: 0,
        }
    }

    #[inline(always)]
    fn add(&mut self, value: V) {
        self.count += 1;
        self.sum += value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    #[inline(always)]
    fn merge(&mut self, rhs: &Record) {
        if rhs.count == 0 {
            return;
        }
        self.count += rhs.count;
        self.sum += rhs.sum;
        self.min = self.min.min(rhs.min);
        self.max = self.max.max(rhs.max);
    }

    #[inline(always)]
    fn avg(&self) -> V {
        self.sum / self.count as V
    }
}

#[inline(always)]
fn probe(keys: &[u64], mask: usize, key: u64) -> usize {
    let mut idx = key as usize & mask;
    loop {
        let k = unsafe { *keys.get_unchecked(idx) };
        if k == key {
            return idx;
        }
        idx = (idx + 1) & mask;
    }
}

fn scan_chunk(chunk: &[u8], keys: &[u64], mask: usize, num_slots: usize) -> Vec<Record> {
    let mut slots = vec![Record::default(); num_slots];
    let mut key_cache: FxHashMap<&[u8], u64> = FxHashMap::default();

    iter_lines(chunk, |name, value| {
        let key = *key_cache.entry(name).or_insert_with(|| to_key(name));
        let idx = probe(keys, mask, key);
        unsafe { slots.get_unchecked_mut(idx) }.add(parse(value));
    });

    slots
}

const L: usize = 32;
#[allow(clippy::upper_case_acronyms)]
type S = Simd<u8, L>;

#[inline(always)]
fn to_key(name: &[u8]) -> u64 {
    xxh3_64(name)
}
// ───────────────────────────────────────────── Custom open‑addressing table

struct HashTable {
    keys: Vec<u64>, // u64::MAX  ==  empty slot
    records: Vec<Record>,
    mask: usize,
}

impl HashTable {
    fn with_capacity(cap: usize) -> Self {
        let cap = cap.next_power_of_two().max(1);
        Self {
            keys: vec![u64::MAX; cap],
            records: vec![Record::default(); cap],
            mask: cap - 1,
        }
    }

    #[inline(always)]
    fn index(&self, key: &u64) -> usize {
        let mut idx = *key as usize & self.mask;
        loop {
            let k = unsafe { *self.keys.get_unchecked(idx) };
            if k == *key || k == u64::MAX {
                return idx;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    /// Insert `key` if it is not present yet.
    #[inline(always)]
    fn insert_key(&mut self, key: u64) {
        let idx = self.index(&key);
        if self.keys[idx] == u64::MAX {
            self.keys[idx] = key;
            // record bucket is already zero‑initialised
        }
    }

    /// Obtain a mutable reference to the `Record` bucket for `key`, creating
    /// it on‑demand.
    #[inline(always)]
    fn record_mut(&mut self, key: u64) -> &mut Record {
        let idx = self.index(&key);
        if self.keys[idx] == u64::MAX {
            self.keys[idx] = key;
        }
        unsafe { self.records.get_unchecked_mut(idx) }
    }
}

// ─────────────────────────────────────────────── Parser helpers

#[inline]
fn to_str(name: &[u8]) -> &str {
    std::str::from_utf8(name).unwrap()
}

#[inline]
fn format_val(v: V) -> String {
    format!("{:.1}", v as f64 / 10.0)
}

/// Parse the fixed‑point value `±abc.d` into *tenths*.
fn parse(mut s: &[u8]) -> V {
    if s.ends_with(b"\r") {
        // ← new
        s = &s[..s.len() - 1];
    }
    let neg = unsafe {
        if *s.get_unchecked(0) == b'-' {
            s = s.get_unchecked(1..);
            true
        } else {
            false
        }
    };

    let (a, b, c, d) = match s {
        [c, b'.', d] => (0, 0, c - b'0', d - b'0'),
        [b, c, b'.', d] => (0, b - b'0', c - b'0', d - b'0'),
        [a, b, c, b'.', d] => (a - b'0', b - b'0', c - b'0', d - b'0'),
        [c] => (0, 0, 0, c - b'0'),
        [b, c] => (0, b - b'0', c - b'0', 0),
        [a, b, c] => (a - b'0', b - b'0', c - b'0', 0),
        _ => panic!("Unknown pattern {:?}", to_str(s)),
    };

    let v = a as V * 1000 + b as V * 100 + c as V * 10 + d as V;
    if neg {
        -v
    } else {
        v
    }
}

// ───────────────────────────────────────────── SIMD line scanner

#[inline(always)]
fn iter_lines<'a>(data: &'a [u8], mut callback: impl FnMut(&'a [u8], &'a [u8])) {
    let simd_data: &[S] = unsafe { data.align_to::<S>().1 };

    let sep = S::splat(b';');
    let end = S::splat(b'\n');
    let mut start_pos = 0;

    let eq_step = |i: &mut usize| {
        *i += 2;
        let lo = simd_data[*i];
        let hi = simd_data[*i + 1];
        (
            ((sep.simd_eq(hi).to_bitmask() as u64) << 32) | sep.simd_eq(lo).to_bitmask() as u64,
            ((end.simd_eq(hi).to_bitmask() as u64) << 32) | end.simd_eq(lo).to_bitmask() as u64,
        )
    };

    let i = &mut usize::MAX.wrapping_sub(1); // = -2 cast to usize
    let (mut eq_sep, mut eq_end) = eq_step(i);

    while *i < simd_data.len() - 3 {
        while eq_sep == 0 {
            (eq_sep, eq_end) = eq_step(i);
        }
        let sep_off = eq_sep.trailing_zeros();
        eq_sep ^= 1 << sep_off;
        let sep_pos = L * *i + sep_off as usize;

        while eq_end == 0 {
            (eq_sep, eq_end) = eq_step(i);
        }
        let end_off = eq_end.trailing_zeros();
        eq_end ^= 1 << end_off;
        let end_pos = L * *i + end_off as usize;

        unsafe {
            let name = data.get_unchecked(start_pos..sep_pos);
            let value = data.get_unchecked(sep_pos + 1..end_pos);
            callback(name, value);
        }

        start_pos = end_pos + 1;
    }
}

// ────────────────────────────────────────────── Build initial table

fn build_hash_table(data: &[u8]) -> (Vec<(u64, &[u8])>, HashTable) {
    let mut cities_map = FxHashMap::default();
    iter_lines(data, |name, _| {
        let key = to_key(name);
        let entry = cities_map.entry(key).or_insert(name);
        debug_assert_eq!(name, *entry);
    });

    let mut cities = cities_map.into_iter().collect::<Vec<_>>();
    cities.sort_unstable_by_key(|&(_k, name)| name);

    let mut table = HashTable::with_capacity(cities.len() * 4);
    for (key, _) in &cities {
        table.insert_key(*key);
    }

    (cities, table)
}

// ────────────────────────────────────────────── entry point

fn main() {
    let start = std::time::Instant::now();
    let filename = args()
        .nth(1)
        .unwrap_or_else(|| "measurements.txt".to_string());

    // ── 1. read file & maintain SIMD alignment ───────────────────────────
    let mut data = Vec::new();
    let offset;
    {
        let stat = std::fs::metadata(&filename).unwrap();
        data.reserve(stat.len() as usize + 2 * L);
        data.resize(4 * L, 0);
        let pre_aligned = unsafe { data.align_to::<S>().0 };
        offset = pre_aligned.len();
        data.resize(offset, 0);
        let mut file = std::fs::File::open(&filename).unwrap();
        eprint!("read  ");
        let t0 = std::time::Instant::now();
        file.read_to_end(&mut data).unwrap();
        eprintln!("{}", format!("{:>5.1?}", t0.elapsed()).bold().green());
    }
    let data = &data[offset..];

    // ── 2. Build hash table  ───────────────
    let t0 = std::time::Instant::now();
    let (cities, mut table) = build_hash_table(&data[..100_000]);
    eprintln!("build {}", format!("{:>5.1?}", t0.elapsed()).bold().green());

    // ── 3. Scan full file & accumulate stats ────────────────────────────
    iter_lines(data, |name, value| {
        let key = to_key(name);
        table.record_mut(key).add(parse(value));
    });

    // ── 4. Optional per‑city dump (disabled by default) ──────────────────
    if false {
        for (key, name) in &cities {
            let idx = table.index(key);
            let r = unsafe { *table.records.get_unchecked(idx) };
            println!(
                "{}: {}/{}/{}",
                to_str(name),
                format_val(r.min),
                format_val(r.avg()),
                format_val(r.max)
            );
        }
        let min_len = cities.iter().map(|x| x.1.len()).min().unwrap();
        let max_len = cities.iter().map(|x| x.1.len()).max().unwrap();
        eprintln!("Min city len: {min_len}");
        eprintln!("Max city len: {max_len}");
    }

    eprintln!("cities {}", cities.len());
    eprintln!(
        "total: {}",
        format!("{:>5.1?}", start.elapsed()).bold().green()
    );
}
