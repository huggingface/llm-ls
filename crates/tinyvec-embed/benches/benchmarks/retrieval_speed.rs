use std::collections::HashMap;

use criterion::{criterion_group, Criterion};
use tinyvec_embed::{
    db::{Collection, Embedding},
    similarity::Distance,
};
use tokio::runtime::Runtime;
use uuid::Uuid;

pub fn get_collection(dimension: usize, embeddings_count: usize) -> Collection {
    let embeddings = (0..embeddings_count)
        .map(|i| Embedding {
            id: Uuid::new_v4(),
            metadata: Some(HashMap::from([(i.to_string(), i.into())])),
            vector: vec![i as f32; dimension],
        })
        .collect::<Vec<Embedding>>();

    Collection {
        dimension,
        distance: Distance::Cosine,
        embeddings,
    }
}

pub fn bench_retrieval(c: &mut Criterion) {
    let dimension = 768;
    let embeddings_count = 50_000;
    let rt = Runtime::new().unwrap();
    c.bench_function("get top 5 k", |b| {
        let collection = get_collection(dimension, embeddings_count);
        let query = vec![42.; dimension];
        b.to_async(&rt)
            .iter(|| async { collection.get(&query, 5, None).await.unwrap() });
    });
}

criterion_group! {
    name = retrieval_speed;
    config = Criterion::default();
    targets = bench_retrieval
}
