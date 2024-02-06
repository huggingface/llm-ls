use serde::{Deserialize, Serialize};
use std::{
    collections::{BinaryHeap, HashMap},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{sync::Semaphore, task::JoinSet};
use tracing::debug;
use uuid::Uuid;

use crate::{
    error::{Collection as Error, Result},
    similarity::{Distance, ScoreIndex},
};

#[derive(Debug, Serialize, Deserialize)]
pub struct Db {
    pub collections: HashMap<String, Collection>,
    pub location: PathBuf,
}

impl Db {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            debug!("Creating database store");
            fs::create_dir_all(
                path.parent()
                    .ok_or(Error::InvalidPath(path.to_path_buf()))?,
            )
            .map_err(Into::<Error>::into)?;

            return Ok(Self {
                collections: HashMap::new(),
                location: path.to_path_buf(),
            });
        }
        debug!("Loading database from store");
        let db = fs::read(path).map_err(Into::<Error>::into)?;
        Ok(bincode::deserialize(&db[..]).map_err(Into::<Error>::into)?)
    }

    pub fn create_collection(
        &mut self,
        name: String,
        dimension: usize,
        distance: Distance,
    ) -> Result<Collection> {
        if self.collections.contains_key(&name) {
            return Err(Error::UniqueViolation.into());
        }

        let collection = Collection {
            dimension,
            distance,
            embeddings: Vec::new(),
        };

        self.collections.insert(name, collection.clone());

        Ok(collection)
    }

    pub fn delete_collection(&mut self, name: &str) {
        self.collections.remove(name);
    }

    pub fn get_collection(&self, name: &str) -> Result<&Collection> {
        self.collections.get(name).ok_or(Error::NotFound.into())
    }

    fn save_to_store(&self) -> Result<()> {
        let db = bincode::serialize(self).map_err(Into::<Error>::into)?;

        fs::write(self.location.as_path(), db).map_err(Into::<Error>::into)?;

        Ok(())
    }
}

impl Drop for Db {
    fn drop(&mut self) {
        debug!("Saving database to store");
        let _ = self.save_to_store();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    score: f32,
    embedding: Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    /// Dimension of the vectors in the collection
    pub dimension: usize,
    /// Distance metric used for querying
    pub distance: Distance,
    /// Embeddings in the collection
    #[serde(default)]
    pub embeddings: Vec<Embedding>,
}

impl Collection {
    pub fn filter(&self) -> FilterBuilder {
        FilterBuilder::new()
    }

    pub async fn get(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<impl FnMut(&&Embedding) -> bool>,
    ) -> Result<Vec<SimilarityResult>> {
        let embeddings = if let Some(filter) = filter {
            self.embeddings.iter().filter(filter).collect::<Vec<_>>()
        } else {
            self.embeddings.iter().collect::<Vec<_>>()
        };
        get_similarity(self.distance, &embeddings, query, k).await
    }

    pub fn insert(&mut self, embedding: Embedding) -> Result<()> {
        if embedding.vector.len() != self.dimension {
            return Err(Error::DimensionMismatch.into());
        }

        self.embeddings.push(embedding);

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    pub id: Uuid,
    pub metadata: Option<HashMap<String, String>>,
    pub vector: Vec<f32>,
}

impl Embedding {
    pub fn new(vector: Vec<f32>, metadata: Option<HashMap<String, String>>) -> Self {
        Self {
            id: Uuid::new_v4(),
            metadata,
            vector,
        }
    }
}

pub enum Compare {
    Eq,
    Neq,
    Gt,
    Lt,
}

#[derive(Clone)]
enum Chain {
    And,
    Or,
}

pub struct FilterBuilder {
    filter: Vec<(String, Compare, String, Option<Chain>)>,
}

impl FilterBuilder {
    pub fn new() -> Self {
        Self { filter: Vec::new() }
    }

    pub fn and(mut self) -> Self {
        self.filter
            .last_mut()
            .map(|c| c.3.as_mut().map(|c| *c = Chain::And));
        self
    }

    pub fn or(mut self) -> Self {
        self.filter
            .last_mut()
            .map(|c| c.3.as_mut().map(|c| *c = Chain::Or));
        self
    }

    pub fn condtion(mut self, lhs: String, op: Compare, rhs: String) -> Self {
        self.filter.push((lhs, op, rhs, None));
        self
    }

    pub fn build(self) -> impl Fn(&&Embedding) -> bool {
        move |e| {
            let mut ret = true;
            let mut prev = None;
            for condition in &self.filter {
                let cond_res = match condition.1 {
                    Compare::Eq => e
                        .metadata
                        .as_ref()
                        .map(|f| f.get(&condition.0) == Some(&condition.2))
                        .unwrap_or(false),
                    Compare::Neq => e
                        .metadata
                        .as_ref()
                        .map(|f| f.get(&condition.0) != Some(&condition.2))
                        .unwrap_or(false),
                    Compare::Gt => e
                        .metadata
                        .as_ref()
                        .map(|f| f.get(&condition.0) > Some(&condition.2))
                        .unwrap_or(false),
                    Compare::Lt => e
                        .metadata
                        .as_ref()
                        .map(|f| f.get(&condition.0) < Some(&condition.2))
                        .unwrap_or(false),
                };
                if let Some(prev) = prev {
                    match prev {
                        Chain::And => ret = ret && cond_res,
                        Chain::Or => ret = ret || cond_res,
                    }
                }
                prev = condition.3.clone();
            }
            ret
        }
    }
}

async fn get_similarity(
    distance: Distance,
    embeddings: &[&Embedding],
    query: &[f32],
    k: usize,
) -> Result<Vec<SimilarityResult>> {
    let semaphore = Arc::new(Semaphore::new(8));
    let mut set = JoinSet::new();
    for (index, embedding) in embeddings.into_iter().enumerate() {
        let embedding = (*embedding).clone();
        let query = query.to_owned();
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        set.spawn_blocking(move || {
            let score = distance.compute(&embedding.vector, &query);
            drop(permit);
            ScoreIndex { score, index }
        });
    }

    let mut heap = BinaryHeap::new();
    while let Some(res) = set.join_next().await {
        let score_index = res.map_err(Into::<Error>::into)?;
        if heap.len() < k || score_index < *heap.peek().unwrap() {
            heap.push(score_index);

            if heap.len() > k {
                heap.pop();
            }
        }
    }
    Ok(heap
        .into_sorted_vec()
        .into_iter()
        .map(|ScoreIndex { score, index }| SimilarityResult {
            score,
            embedding: embeddings[index].clone(),
        })
        .collect())
}
