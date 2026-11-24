For variable-length columns this ensures that parquet pages would only ever overflow if a single row was greater than the i32::MAX limit.

This does require computation on a per-row basis, but aligns with what [DuckDB](https://github.com/duckdb/duckdb/blob/d4eea3f208d42ca4ee522aca23f5ea76f902b0ca/extension/parquet/writer/primitive_column_writer.cpp#L107-L166) and [arrow-rs](https://github.com/apache/arrow-rs/blob/5cdc728dcf5937ede19b381fa83ae20a221a7a93/parquet/src/column/writer/byte_budget_chunker.rs) use.

A straightforward per-row pass measured about 28–38 ms per 20M rows (5–9% slower writes).
I tried doing some performance stuff, including:

1. offsets - checking if a full set of rows can be stored on one page immediately ([which arrow-rs also does](https://github.com/apache/arrow-rs/blob/5cdc728dcf5937ede19b381fa83ae20a221a7a93/parquet/src/arrow/arrow_writer/byte_array.rs#L1006-L1045))
2. offsets - using a binary search over the leaf offsets.
3. views - using a single linear pass over lengths.

The performance calculation is now 0.06–0.14 ms (offsets) and ~6.5 ms (views) per 20M rows, against 0.04 / 4.9 ms on main.

| Column | Pages main → fix | Max page main → fix | Range calc main → fix | zstd write vs main |
|---|---|---|---|---|
| offset short (~8 B) | 326 → 326 | 723 → 1024 KiB | 0.04 → 0.06 ms | +0.4% |
| offset medium (~32 B) | 814 → 814 | 899 → 1024 KiB | 0.04 → 0.13 ms | −2.7% |
| offset skewed | 526 → 405 | 4337 → 4096 KiB | 0.04 → 0.06 ms | +3.4% |
| view short | 325 → 326 | 1365 → 1024 KiB | 4.9 → 6.6 ms | −4.1% |
| view medium | 651 → 814 | 1117 → 1024 KiB | 4.9 → 6.6 ms | −1.5% |
| view skewed | 525 → 405 | 4382 → 4096 KiB | 4.9 → 6.6 ms | +1.1% |
| int64 (control) | 325 → 325 | — | — | +0.0% |

<sub>Population: one column of 20M rows per case, written in row groups of 122,880 rows (the Python sink default); dictionary encoding attempted, then plain; V1 pages; zstd. Single-threaded encode to memory, best of 15 runs over 2 interleaved rounds, on an Apple M4 Pro. "Skewed" means 8 B values plus 50 × 4 MiB values. Run-to-run noise is about ±1.5% with zstd. Source: the harness below; main = `f4b7a81dc5`, fix = this PR.</sub>

- **Offset skewed (+3.4%):** the extra time comes from the new page layout (each 4 MiB value on its own page).
- **View columns:** the seem to be faster than main because main overshot the 1 MiB target (up to 1365 KiB pages), not because the new code is faster.

<details>
<summary>Benchmark harness (<code>bench_pages.rs</code>)</summary>

Copy into `crates/polars-parquet/examples/` and run:

```sh
cargo build --profile fast-release -p polars-parquet --example bench_pages --features zstd
target/fast-release/examples/bench_pages <label> [repeats]
```

```rust
//! Benchmark harness for parquet page splitting (#30312). Untracked; copied into
//! crates/polars-parquet/examples/ for each revision.
//!
//! Usage: bench_pages <label> [repeats]
//! Env: BENCH_ROWS (default 20_000_000), BENCH_RG (row group rows, default 122_880 = the
//! Python sink morsel size), BENCH_COLS (comma list), BENCH_COMP (comma list: zstd,none).
//!
//! Output: TSV lines on stdout.
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

use polars_arrow::array::{Array, Int64Array, MutableBinaryViewArray, Utf8Array};
use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field};
use polars_arrow::offset::{Offsets, OffsetsBuffer};
use polars_arrow::record_batch::RecordBatchT;
use polars_buffer::Buffer;
use polars_parquet::parquet::page::CompressedPage;
use polars_parquet::write::{
    CompressionOptions, Encoding, FileWriter, Page, RowGroupIterator, StatisticsOptions, Version,
    WriteOptions, array_to_columns, compress, to_parquet_type,
};

const BIG_EVERY: usize = 400_000;
const BIG_LEN: usize = 4 << 20; // 4 MiB

fn gen_value(col: &str, i: usize, buf: &mut String, rng: &mut u64) {
    use std::fmt::Write;
    buf.clear();
    match col {
        "short" => write!(buf, "{i:08}").unwrap(),
        "medium" => write!(buf, "customer-name-{i:018}").unwrap(),
        "skewed" => {
            if i % BIG_EVERY == BIG_EVERY / 2 {
                buf.reserve(BIG_LEN);
                for _ in 0..BIG_LEN {
                    *rng ^= *rng << 13;
                    *rng ^= *rng >> 7;
                    *rng ^= *rng << 17;
                    buf.push((b'a' + (*rng % 26) as u8) as char);
                }
            } else {
                write!(buf, "{i:08}").unwrap()
            }
        },
        _ => unreachable!(),
    }
}

fn build(col: &str, n: usize, view: bool) -> Box<dyn Array> {
    if col == "int64" {
        return Int64Array::from_vec((0..n as i64).collect()).boxed();
    }
    let mut buf = String::new();
    let mut rng = 0x9E3779B97F4A7C15u64;
    if view {
        let mut m = MutableBinaryViewArray::<str>::with_capacity(n);
        for i in 0..n {
            gen_value(col, i, &mut buf, &mut rng);
            m.push_value(buf.as_str());
        }
        m.freeze().boxed()
    } else {
        let mut offsets = Vec::with_capacity(n + 1);
        let mut values = Vec::<u8>::new();
        offsets.push(0i64);
        for i in 0..n {
            gen_value(col, i, &mut buf, &mut rng);
            values.extend_from_slice(buf.as_bytes());
            offsets.push(values.len() as i64);
        }
        let offsets: OffsetsBuffer<i64> = unsafe { Offsets::new_unchecked(offsets) }.into();
        Utf8Array::<i64>::new(ArrowDataType::LargeUtf8, offsets, Buffer::from(values), None)
            .boxed()
    }
}

fn options(comp: &str) -> WriteOptions {
    WriteOptions {
        statistics: StatisticsOptions::default(),
        version: Version::V1,
        compression: match comp {
            "zstd" => CompressionOptions::Zstd(None),
            _ => CompressionOptions::Uncompressed,
        },
        data_page_size: None,
    }
}

fn row_groups(array: &dyn Array, rg: usize) -> Vec<Box<dyn Array>> {
    (0..array.len())
        .step_by(rg)
        .map(|s| array.sliced(s, rg.min(array.len() - s)))
        .collect()
}

/// Full file write into memory, like polars-io: RleDictionary attempted, falls back to plain.
fn write_file(field: &Field, rgs: &[Box<dyn Array>], opts: WriteOptions) -> usize {
    let schema = ArrowSchema::from_iter([field.clone()]);
    let schema_ref = Arc::new(schema.clone());
    let batches = rgs
        .iter()
        .map(|a| RecordBatchT::try_new(a.len(), schema_ref.clone(), vec![a.clone()]));
    let iter = RowGroupIterator::try_new(
        batches,
        &schema,
        opts,
        Buffer::from_iter([vec![Encoding::RleDictionary]]),
    )
    .unwrap();
    let mut w = FileWriter::try_new(Cursor::new(Vec::new()), schema, opts, None).unwrap();
    for (rg, a) in iter.zip(rgs) {
        w.write(a.len() as u64, rg.unwrap()).unwrap();
    }
    w.end(None).unwrap() as usize
}

#[derive(Default)]
struct PageStats {
    data_pages: usize,
    dict_pages: usize,
    unc_sum: usize,
    unc_max: usize,
    comp_sum: usize,
    comp_max: usize,
}

fn page_stats(field: &Field, rgs: &[Box<dyn Array>], opts: WriteOptions) -> PageStats {
    let type_ = to_parquet_type(field).unwrap();
    let mut st = PageStats::default();
    for a in rgs {
        let cols =
            array_to_columns(a.clone(), type_.clone(), opts, &[Encoding::RleDictionary]).unwrap();
        for col in cols {
            for page in col {
                let page = page.unwrap();
                let is_dict = matches!(page, Page::Dict(_));
                match compress(page, vec![], opts.compression).unwrap() {
                    CompressedPage::Data(p) => {
                        st.data_pages += 1;
                        st.unc_sum += p.uncompressed_size();
                        st.unc_max = st.unc_max.max(p.uncompressed_size());
                        st.comp_sum += p.compressed_size();
                        st.comp_max = st.comp_max.max(p.compressed_size());
                    },
                    CompressedPage::Dict(_) => {
                        assert!(is_dict);
                        st.dict_pages += 1
                    },
                }
            }
        }
    }
    st
}

/// Only the work done before the first page is produced: nesting info and (on the fix
/// branches) the eager page-range computation. Plain encoding, so no dictionary attempt.
fn ranges_only(field: &Field, rgs: &[Box<dyn Array>], opts: WriteOptions) -> usize {
    let type_ = to_parquet_type(field).unwrap();
    let mut n = 0;
    for a in rgs {
        let cols = array_to_columns(a.clone(), type_.clone(), opts, &[Encoding::Plain]).unwrap();
        n += cols.len();
    }
    n
}

fn best<T>(repeats: usize, mut f: impl FnMut() -> T) -> (f64, f64, T) {
    let mut times = Vec::new();
    let mut out = None;
    for _ in 0..repeats {
        let t = Instant::now();
        out = Some(std::hint::black_box(f()));
        times.push(t.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[0], times[times.len() / 2], out.unwrap())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let label = &args[1];
    let repeats: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(15);
    let n: usize = std::env::var("BENCH_ROWS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(20_000_000);
    let rg: usize = std::env::var("BENCH_RG")
        .map(|s| s.parse().unwrap())
        .unwrap_or(122_880);
    let cols = std::env::var("BENCH_COLS").unwrap_or("short,medium,skewed,int64".into());
    let comps = std::env::var("BENCH_COMP").unwrap_or("zstd,none".into());

    for col in cols.split(',') {
        for view in [false, true] {
            if col == "int64" && view {
                continue;
            }
            let path = if col == "int64" {
                "prim"
            } else if view {
                "view"
            } else {
                "offset"
            };
            let array = build(col, n, view);
            let field = Field::new("a".into(), array.dtype().clone(), true);
            let rgs = row_groups(array.as_ref(), rg);
            for comp in comps.split(',') {
                let opts = options(comp);
                let st = page_stats(&field, &rgs, opts);
                let (wmin, wmed, size) = best(repeats, || write_file(&field, &rgs, opts));
                let (rmin, rmed, _) = best(repeats, || ranges_only(&field, &rgs, opts));
                println!(
                    "{label}\t{path}\t{col}\t{comp}\twrite_min={wmin:.4}\twrite_med={wmed:.4}\tsetup_min={rmin:.5}\tsetup_med={rmed:.5}\tfile_mib={:.2}\tdata_pages={}\tdict_pages={}\tavg_unc_kib={:.1}\tmax_unc_kib={:.1}\tavg_comp_kib={:.1}\tmax_comp_kib={:.1}",
                    size as f64 / (1 << 20) as f64,
                    st.data_pages,
                    st.dict_pages,
                    st.unc_sum as f64 / st.data_pages as f64 / 1024.0,
                    st.unc_max as f64 / 1024.0,
                    st.comp_sum as f64 / st.data_pages as f64 / 1024.0,
                    st.comp_max as f64 / 1024.0,
                );
            }
        }
    }
}
```

</details>

happy though to switch back to the less-complex version if it's not an issue. Performance and filesize also seems to vary marginally based on zstd stuff
(the current implementation is hewing closer to the 1MB default)

Not addressed:
- the half-written file thing
- #29624

AI Disclosure: I did use Claude Opus 5.5 heavily in diagnosing, building,
testing, and especially benchmarking the fix, but I guided it heavily, read everything
several times, and sought out / read the comparable fixes myself. 
