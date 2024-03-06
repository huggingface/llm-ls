use criterion::criterion_main;

mod benchmarks;

criterion_main! {
    benchmarks::retrieval_speed::retrieval_speed,
}
