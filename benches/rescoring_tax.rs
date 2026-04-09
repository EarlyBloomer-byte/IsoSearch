#![allow(missing_docs, clippy::pedantic, clippy::nursery, clippy::unwrap_used)]
//! Rescoring tax analysis benchmark suite.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use isosearch::graph::{HNSWGraph, Node};
use isosearch::hashing::{BinaryQuantizer, LocalitySensitiveHasher, Quantizer, SimHasher};
use isosearch::indexing::{BucketIndex, VectorStore};
use isosearch::normalization::{Normalizer, WhiteningNormalizer};
use isosearch::projection::{PoincareProjector, Projector, RandomProjector};
use isosearch::types::{Embedding, ID};
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashSet;
use std::fmt::Write as FmtWrite;
use std::time::{Duration, Instant};

const DIM: usize = 384;
const TARGET_DIM: usize = 128;
const HASH_BITS: usize = 256;
const N_NODES: usize = 10_000;
const DEGREE: usize = 32;
const SAMPLE_SIZE: usize = 128;
const MAX_K: usize = 1000;
const K_VALUES: [usize; 7] = [10, 25, 50, 100, 200, 500, 1000];
const RNG_SEED: u64 = 0x5EED_5EED_5EED_1234;

struct BenchContext {
    corpus: Vec<Embedding>,
    vector_store: VectorStore,
    index: BucketIndex,
    graph: HNSWGraph,
    normalizer: WhiteningNormalizer,
    poincare: PoincareProjector,
    projector: RandomProjector,
    hasher: SimHasher,
    quantizer: BinaryQuantizer,
    query_raw: Embedding,
    use_poincare: bool,
    exact_top_ids: Vec<ID>,
}

#[derive(Clone, Default)]
struct TimingTotals {
    total_iters: u64,
    candidates_ns: u128,
    rescore_ns: u128,
    total_ns: u128,
}

#[derive(Clone, Copy)]
struct CsvRow {
    k: usize,
    recall_at_k: f32,
    t_candidates_ns: u64,
    t_rescore_ns: u64,
    t_total_ns: u64,
}

fn bench_rescoring_tax(c: &mut Criterion) {
    let ctx = build_context();
    let recall_map = compute_recall_map(&ctx);

    let mut group = c.benchmark_group("rescoring_tax");
    let mut timing_totals = vec![TimingTotals::default(); K_VALUES.len()];

    for (idx, &k) in K_VALUES.iter().enumerate() {
        let totals = &mut timing_totals[idx];
        group.bench_function(format!("rescore_top_{k}"), |b| {
            let mut candidate_ids: Vec<ID> = Vec::with_capacity(k);
            let mut rescore_buf: Vec<(ID, f32)> = Vec::with_capacity(k);

            b.iter_custom(|iters| {
                let mut candidates_total = Duration::ZERO;
                let mut rescore_total = Duration::ZERO;
                let mut total_total = Duration::ZERO;

                for _ in 0..iters {
                    let total_start = Instant::now();

                    let query_hash = hash_embedding(
                        &ctx.normalizer,
                        &ctx.poincare,
                        &ctx.projector,
                        &ctx.hasher,
                        &ctx.quantizer,
                        &ctx.query_raw,
                        ctx.use_poincare,
                    );

                    let candidates_start = Instant::now();
                    let bucket_candidates = ctx.index.intersect(black_box(&query_hash));
                    let graph_results = ctx.graph.search(black_box(&query_hash), k);
                    let candidates_elapsed = candidates_start.elapsed();

                    candidate_ids.clear();
                    candidate_ids.extend(graph_results.iter().map(|(id, _)| *id));

                    let rescore_start = Instant::now();
                    rescore_candidates(
                        &ctx.vector_store,
                        &ctx.query_raw,
                        &candidate_ids,
                        &mut rescore_buf,
                    );
                    let rescore_elapsed = rescore_start.elapsed();

                    let total_elapsed = total_start.elapsed();

                    black_box(&bucket_candidates);
                    black_box(&rescore_buf);

                    candidates_total += candidates_elapsed;
                    rescore_total += rescore_elapsed;
                    total_total += total_elapsed;
                }

                totals.total_iters += iters;
                totals.candidates_ns += candidates_total.as_nanos();
                totals.rescore_ns += rescore_total.as_nanos();
                totals.total_ns += total_total.as_nanos();

                total_total
            });
        });
    }

    group.finish();

    let mut rows = Vec::with_capacity(K_VALUES.len());
    for (idx, &k) in K_VALUES.iter().enumerate() {
        let totals = &timing_totals[idx];
        let recall_at_k = recall_map[idx].1;
        let iters = totals.total_iters.max(1) as u128;

        rows.push(CsvRow {
            k,
            recall_at_k,
            t_candidates_ns: (totals.candidates_ns / iters) as u64,
            t_rescore_ns: (totals.rescore_ns / iters) as u64,
            t_total_ns: (totals.total_ns / iters) as u64,
        });
    }

    if let Err(err) = write_csv(&rows) {
        eprintln!("rescoring_tax: failed to write CSV: {err}");
    }
}

fn build_context() -> BenchContext {
    let use_poincare = env_flag("ISOSEARCH_USE_POINCARE");
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let dist = Normal::new(0.0, 1.0).unwrap();

    let mut corpus = Vec::with_capacity(N_NODES);
    for _ in 0..N_NODES {
        corpus.push(sample_embedding(&mut rng, &dist, DIM));
    }

    let query_raw = sample_embedding(&mut rng, &dist, DIM);
    let normalizer = WhiteningNormalizer::fit(&corpus).expect("normalizer fit failed");

    let projector = seeded_random_projector(RNG_SEED ^ 0xA5A5_A5A5_A5A5_A5A5, DIM, TARGET_DIM);
    let hasher = seeded_simhasher(RNG_SEED ^ 0xC3C3_C3C3_C3C3_C3C3, TARGET_DIM, HASH_BITS);
    let quantizer = BinaryQuantizer::new();
    let poincare = PoincareProjector::new();

    let mut index = BucketIndex::new();
    let mut vector_store = VectorStore::new();
    let mut hashes: Vec<Vec<u64>> = Vec::with_capacity(N_NODES);

    for (i, embedding) in corpus.iter().enumerate() {
        let id = i as ID;
        vector_store.insert(id, embedding.clone());

        let hash = hash_embedding(
            &normalizer,
            &poincare,
            &projector,
            &hasher,
            &quantizer,
            embedding,
            use_poincare,
        );

        for &word in &hash {
            index.insert(word, id);
        }

        hashes.push(hash);
    }

    let graph = build_graph(&hashes);
    let exact_top_ids = exact_top_k(&corpus, &query_raw, MAX_K);

    BenchContext {
        corpus,
        vector_store,
        index,
        graph,
        normalizer,
        poincare,
        projector,
        hasher,
        quantizer,
        query_raw,
        use_poincare,
        exact_top_ids,
    }
}

fn build_graph(hashes: &[Vec<u64>]) -> HNSWGraph {
    let mut graph = HNSWGraph::new();
    let mut rng = StdRng::seed_from_u64(RNG_SEED ^ 0xDEAD_BEEF_1234_5678);

    for id in 0..hashes.len() {
        let mut sample_ids: HashSet<ID> = HashSet::with_capacity(SAMPLE_SIZE);
        while sample_ids.len() < SAMPLE_SIZE {
            let cand = rng.gen_range(0..hashes.len());
            if cand != id {
                sample_ids.insert(cand as ID);
            }
        }

        let mut scored: Vec<(ID, u32)> = Vec::with_capacity(SAMPLE_SIZE);
        for cand in sample_ids {
            let dist = HNSWGraph::fast_hamming_distance(&hashes[id], &hashes[cand as usize]);
            scored.push((cand, dist));
        }

        scored.sort_unstable_by(|a, b| a.1.cmp(&b.1));
        let neighbors: Vec<ID> = scored.into_iter().take(DEGREE).map(|(cand, _)| cand).collect();

        graph.nodes.insert(
            id as ID,
            Node {
                id: id as ID,
                hash: hashes[id].clone(),
                neighbors: vec![neighbors],
            },
        );
    }

    graph.entry_point = Some(0);
    graph.max_level = 0;
    graph
}

fn hash_embedding(
    normalizer: &WhiteningNormalizer,
    poincare: &PoincareProjector,
    projector: &RandomProjector,
    hasher: &SimHasher,
    quantizer: &BinaryQuantizer,
    embedding: &Embedding,
    use_poincare: bool,
) -> Vec<u64> {
    let normalized = normalizer.normalize(embedding);
    let projected = if use_poincare {
        poincare.project(&normalized)
    } else {
        normalized
    };
    let reduced = projector.project(&projected);
    let hash_bits = hasher.hash(&reduced);
    quantizer.quantize(&hash_bits)
}

fn rescore_candidates(
    store: &VectorStore,
    query: &Embedding,
    candidates: &[ID],
    out: &mut Vec<(ID, f32)>,
) {
    out.clear();
    out.reserve(candidates.len());

    for &id in candidates {
        if let Some(vec) = store.storage.get(&id) {
            let dist_sq: f32 = query
                .iter()
                .zip(vec.iter())
                .map(|(q, v)| {
                    let d = q - v;
                    d * d
                })
                .sum();
            out.push((id, dist_sq));
        }
    }

    out.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
}

fn exact_top_k(corpus: &[Embedding], query: &Embedding, k: usize) -> Vec<ID> {
    let mut scored: Vec<(ID, f32)> = Vec::with_capacity(corpus.len());
    for (idx, vec) in corpus.iter().enumerate() {
        let dist_sq: f32 = query
            .iter()
            .zip(vec.iter())
            .map(|(q, v)| {
                let d = q - v;
                d * d
            })
            .sum();
        scored.push((idx as ID, dist_sq));
    }

    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

fn compute_recall_map(ctx: &BenchContext) -> Vec<(usize, f32)> {
    let mut rows = Vec::with_capacity(K_VALUES.len());
    let mut rescore_buf: Vec<(ID, f32)> = Vec::with_capacity(MAX_K);

    for &k in &K_VALUES {
        let query_hash = hash_embedding(
            &ctx.normalizer,
            &ctx.poincare,
            &ctx.projector,
            &ctx.hasher,
            &ctx.quantizer,
            &ctx.query_raw,
            ctx.use_poincare,
        );

        let _bucket_candidates = ctx.index.intersect(&query_hash);
        let graph_results = ctx.graph.search(&query_hash, k);

        let mut candidate_ids: Vec<ID> = Vec::with_capacity(graph_results.len());
        candidate_ids.extend(graph_results.iter().map(|(id, _)| *id));

        rescore_candidates(
            &ctx.vector_store,
            &ctx.query_raw,
            &candidate_ids,
            &mut rescore_buf,
        );
        rescore_buf.truncate(k);

        let exact_set: HashSet<ID> = ctx.exact_top_ids[..k].iter().copied().collect();
        let hits = rescore_buf
            .iter()
            .filter(|(id, _)| exact_set.contains(id))
            .count();
        let recall_at_k = hits as f32 / k as f32;

        rows.push((k, recall_at_k));
    }

    rows
}

fn sample_embedding(rng: &mut StdRng, dist: &Normal<f32>, dim: usize) -> Embedding {
    let mut values = Vec::with_capacity(dim);
    for _ in 0..dim {
        values.push(dist.sample(rng));
    }
    Array1::from_vec(values)
}

fn seeded_random_projector(seed: u64, source_dim: usize, target_dim: usize) -> RandomProjector {
    let mut rng = StdRng::seed_from_u64(seed);
    #[allow(clippy::cast_precision_loss)]
    let std_dev = 1.0 / (target_dim as f32).sqrt();
    let dist = Normal::new(0.0, std_dev).unwrap();

    let matrix = ndarray::Array2::from_shape_fn((target_dim, source_dim), |_| dist.sample(&mut rng));
    RandomProjector { matrix }
}

fn seeded_simhasher(seed: u64, dim: usize, bits: usize) -> SimHasher {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Normal::new(0.0, 1.0).unwrap();

    let hyperplanes = ndarray::Array2::from_shape_fn((bits, dim), |_| dist.sample(&mut rng));
    SimHasher { hyperplanes }
}

fn env_flag(key: &str) -> bool {
    std::env::var(key)
        .map(|value| {
            let value = value.to_ascii_lowercase();
            value == "1" || value == "true" || value == "yes"
        })
        .unwrap_or(false)
}

fn write_csv(rows: &[CsvRow]) -> std::io::Result<()> {
    let output_dir = std::path::Path::new("bench-results");
    std::fs::create_dir_all(output_dir)?;
    let output_path = output_dir.join("rescoring_tax.csv");

    let mut csv = String::new();
    csv.push_str("k,recall_at_k,t_candidates_ns,t_rescore_ns,t_total_ns\n");
    for row in rows {
        writeln!(
            &mut csv,
            "{},{:.6},{},{},{}",
            row.k, row.recall_at_k, row.t_candidates_ns, row.t_rescore_ns, row.t_total_ns
        )
        .expect("csv write failed");
    }

    std::fs::write(output_path, csv)
}

criterion_group!(benches, bench_rescoring_tax);
criterion_main!(benches);
