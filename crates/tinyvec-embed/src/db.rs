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
        filter: Option<FilterBuilder>,
    ) -> Result<Vec<SimilarityResult>> {
        let embeddings = if let Some(filter) = filter {
            self.embeddings
                .iter()
                .filter(filter.build())
                .collect::<Vec<_>>()
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
    pub metadata: Option<HashMap<String, Value>>,
    pub vector: Vec<f32>,
}

impl Embedding {
    pub fn new(vector: Vec<f32>, metadata: Option<HashMap<String, Value>>) -> Self {
        Self {
            id: Uuid::new_v4(),
            metadata,
            vector,
        }
    }
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Embedding {}

#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Value {
    String(String),
    Number(f32),
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Self::Number(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

#[derive(Debug)]
pub enum Compare {
    Eq,
    Neq,
    Gt,
    Lt,
}

#[derive(Clone, Debug)]
enum Chain {
    And,
    Or,
}

pub struct FilterBuilder {
    filter: Vec<(String, Compare, Value, Option<Chain>)>,
}

impl FilterBuilder {
    pub fn new() -> Self {
        Self { filter: Vec::new() }
    }

    pub fn and(mut self) -> Self {
        if let Some(c) = self.filter.last_mut() {
            c.3 = Some(Chain::And);
        };
        self
    }

    pub fn or(mut self) -> Self {
        if let Some(c) = self.filter.last_mut() {
            c.3 = Some(Chain::Or);
        }
        self
    }

    pub fn comparison(mut self, key: String, op: Compare, value: Value) -> Self {
        assert!(
            self.filter.last().map(|c| c.3.is_some()).unwrap_or(true),
            "Missing chain operator in filter"
        );
        self.filter.push((key, op, value, None));
        self
    }

    // XXX: we assume the user will chain filters correctly
    fn build(self) -> impl Fn(&&Embedding) -> bool {
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
                } else {
                    ret = cond_res;
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
    for (index, embedding) in embeddings.iter().enumerate() {
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

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[tokio::test]
    async fn simple_similarity() {
        let temp_dir = TempDir::new().expect("failed to create tempt dir");
        let db_path = temp_dir.path().join("embeddings.db");
        let mut db = match Db::open(db_path) {
            Ok(db) => db,
            Err(err) => panic!("{}", err.to_string()),
        };
        let mut col = match db.create_collection("test".to_owned(), 5, Distance::Cosine) {
            Ok(col) => col,
            Err(err) => panic!("{}", err.to_string()),
        };
        let embedding = Embedding::new(
            vec![0.9999695, 0.76456239, 0.86767905, 0.17577756, 0.9949882],
            None,
        );
        col.insert(embedding.clone())
            .expect("faield to insert embedding");

        let expected = SimilarityResult {
            score: 0.7449362,
            embedding,
        };
        let results = col
            .get(
                &[0.5902804, 0.516834, 0.12403694, 0.8444756, 0.4672038],
                1,
                None,
            )
            .await
            .expect("failed to get most similar embeddings");
        let actual = results
            .first()
            .expect("missing embedding in similarity result");
        assert!((expected.score - actual.score).abs() <= f32::EPSILON);
        assert_eq!(expected.embedding.id, actual.embedding.id);
    }

    #[tokio::test]
    async fn filter() {
        let temp_dir = TempDir::new().expect("failed to create tempt dir");
        let db_path = temp_dir.path().join("embeddings.db");
        let mut db = match Db::open(db_path) {
            Ok(db) => db,
            Err(err) => panic!("{}", err.to_string()),
        };
        let mut col = match db.create_collection("test".to_owned(), 5, Distance::Cosine) {
            Ok(col) => col,
            Err(err) => panic!("{}", err.to_string()),
        };
        let embedding1 = Embedding::new(
            vec![0.5880849, 0.25781349, 0.32253786, 0.80958734, 0.8591076],
            Some(HashMap::from([
                ("i".to_owned(), 32.0.into()),
                ("j".to_owned(), 10.0.into()),
            ])),
        );
        col.insert(embedding1.clone())
            .expect("faield to insert embedding");
        let embedding2 = Embedding::new(
            vec![0.43717385, 0.21100248, 0.5068433, 0.9626808, 0.6763327],
            Some(HashMap::from([
                ("i".to_owned(), 7.0.into()),
                ("j".to_owned(), 100.0.into()),
            ])),
        );
        col.insert(embedding2.clone())
            .expect("faield to insert embedding");
        let embedding3 = Embedding::new(
            vec![0.2630481, 0.24888718, 0.3375401, 0.92770165, 0.44944693],
            Some(HashMap::from([
                ("i".to_owned(), 29.0.into()),
                ("j".to_owned(), 16.0.into()),
            ])),
        );
        col.insert(embedding3.clone())
            .expect("faield to insert embedding");
        let embedding4 = Embedding::new(
            vec![0.7642892, 0.47043378, 0.9035855, 0.31120034, 0.5757918],
            Some(HashMap::from([
                ("i".to_owned(), 3.0.into()),
                ("j".to_owned(), 110.0.into()),
            ])),
        );
        col.insert(embedding4.clone())
            .expect("faield to insert embedding");

        let results = col
            .get(
                &[0.09537213, 0.5104327, 0.69980987, 0.13146928, 0.30541683],
                4,
                Some(
                    col.filter()
                        .comparison("i".to_owned(), Compare::Lt, 25.0.into())
                        .and()
                        .comparison("j".to_owned(), Compare::Gt, 50.0.into()),
                ),
            )
            .await
            .expect("failed to get most similar embeddings");
        let actual_scores: Vec<f32> = results.iter().map(|r| r.score).collect();
        let expected_scores: Vec<f32> = vec![0.8701641, 0.6552329];
        assert!(expected_scores
            .iter()
            .zip(actual_scores.iter())
            .all(|(e, a)| { (e - a).abs() <= f32::EPSILON }));
        let expected_embeddings = vec![embedding4, embedding2];
        let actual_embeddings: Vec<Embedding> = results.into_iter().map(|r| r.embedding).collect();
        assert_eq!(expected_embeddings, actual_embeddings);
    }
}
