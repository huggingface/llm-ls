use serde::{Deserialize, Serialize};
use std::{
    collections::{BinaryHeap, HashMap},
    fmt::Display,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::{
    fs,
    sync::{RwLock, Semaphore},
    task::JoinSet,
};
use tracing::debug;
use uuid::Uuid;

use crate::{
    error::{Collection as CollectionError, Error, Result},
    similarity::{Distance, ScoreIndex},
};

#[derive(Clone, Debug)]
pub struct Db {
    inner: Arc<RwLock<DbInner>>,
}

#[derive(Clone, Debug)]
pub struct DbInner {
    collections: HashMap<String, Arc<RwLock<Collection>>>,
    location: PathBuf,
}

impl Db {
    /// Opens a database from disk or creates a new one if it doesn't exist
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut inner = DbInner {
            collections: HashMap::new(),
            location: path.to_path_buf(),
        };
        if !path.exists() {
            debug!("Creating database store");
            fs::create_dir_all(path).await?;

            return Ok(Self {
                inner: Arc::new(RwLock::new(inner)),
            });
        }
        debug!("Loading database from store");

        let mut entries = fs::read_dir(path).await?;
        while let Some(entry) = entries.next_entry().await? {
            let entry_type = entry.file_type().await?;
            if entry_type.is_file() {
                let col = fs::read(entry.path()).await?;
                let col = bincode::deserialize(&col[..])?;
                let name = entry
                    .file_name()
                    .to_str()
                    .ok_or(Error::InvalidFileName)?
                    .to_owned();
                inner.collections.insert(name, Arc::new(RwLock::new(col)));
            } else {
                // warning?
            }
        }
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    pub async fn create_collection(
        &mut self,
        name: String,
        dimension: usize,
        distance: Distance,
    ) -> Result<Arc<RwLock<Collection>>> {
        if self.inner.read().await.collections.contains_key(&name) {
            return Err(CollectionError::UniqueViolation.into());
        }

        let collection = Arc::new(RwLock::new(Collection {
            dimension,
            distance,
            embeddings: Vec::new(),
        }));

        self.inner
            .write()
            .await
            .collections
            .insert(name, collection.clone());

        Ok(collection)
    }

    /// Removes a collection from [`Db`].
    ///
    /// The [`Collection`] will still exist in memory for as long as you hold a copy, given it is
    /// wrapped in an `Arc`.
    pub async fn delete_collection(&mut self, name: &str) {
        self.inner.write().await.collections.remove(name);
    }

    pub async fn get_collection(&self, name: &str) -> Result<Arc<RwLock<Collection>>> {
        self.inner
            .read()
            .await
            .collections
            .get(name)
            .ok_or(CollectionError::NotFound.into())
            .cloned()
    }

    /// Save database to disk
    pub async fn save(&self) -> Result<()> {
        let inner = self.inner.read().await;
        for (name, collection) in inner.collections.iter() {
            let db = bincode::serialize(&*collection.read().await)?;

            fs::write(inner.location.as_path().join(name), db).await?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimilarityResult {
    pub score: f32,
    pub embedding: Embedding,
}

#[derive(Debug, Serialize, Deserialize)]
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
    pub fn filter() -> FilterBuilder {
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
                .filter(filter.fn_ref_closure())
                .collect::<Vec<_>>()
        } else {
            self.embeddings.iter().collect::<Vec<_>>()
        };
        get_similarity(self.distance, &embeddings, query, k).await
    }

    pub fn insert(&mut self, embedding: Embedding) -> Result<()> {
        if embedding.vector.len() != self.dimension {
            return Err(CollectionError::DimensionMismatch.into());
        }

        self.embeddings.push(embedding);

        Ok(())
    }

    pub fn batch_insert(&mut self, embeddings: Vec<Embedding>) -> Result<()> {
        if embeddings
            .iter()
            .any(|embedding| embedding.vector.len() != self.dimension)
        {
            return Err(CollectionError::DimensionMismatch.into());
        }
        self.embeddings.extend(embeddings);
        Ok(())
    }

    /// Remove values matching filter.
    ///
    /// Empties the collection when `filter` is `None`.
    pub fn remove(&mut self, filter: Option<FilterBuilder>) -> Result<()> {
        if let Some(filter) = filter {
            let mut closure = filter.fn_mut_closure();
            self.embeddings.retain(|e| !closure(e));
        } else {
            self.embeddings.clear();
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Value {
    String(String),
    Number(f32),
    Usize(usize),
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => write!(f, "{s}"),
            Self::Number(n) => write!(f, "{n}"),
            Self::Usize(u) => write!(f, "{u}"),
        }
    }
}

impl Value {
    pub fn inner_string(&self) -> Result<String> {
        match self {
            Self::String(s) => Ok(s.to_owned()),
            _ => Err(Error::ValueNotString(self.to_owned())),
        }
    }

    pub fn inner_value(&self) -> Result<usize> {
        match self {
            Self::Usize(s) => Ok(s.to_owned()),
            _ => Err(Error::ValueNotString(self.to_owned())),
        }
    }
}

impl TryInto<usize> for &Value {
    type Error = Error;

    fn try_into(self) -> Result<usize> {
        if let Value::Number(n) = self {
            Ok(n.clone() as usize)
        } else {
            Err(Error::ValueNotNumber(self.to_owned()))
        }
    }
}

impl From<usize> for Value {
    fn from(value: usize) -> Self {
        Self::Number(value as f32)
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Self::Number(value as f32)
    }
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
    GtEq,
    Lt,
    LtEq,
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

    fn fn_mut_closure(self) -> impl FnMut(&Embedding) -> bool {
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
                    Compare::GtEq => e
                        .metadata
                        .as_ref()
                        .map(|f| f.get(&condition.0) >= Some(&condition.2))
                        .unwrap_or(false),
                    Compare::Lt => e
                        .metadata
                        .as_ref()
                        .map(|f| f.get(&condition.0) < Some(&condition.2))
                        .unwrap_or(false),
                    Compare::LtEq => e
                        .metadata
                        .as_ref()
                        .map(|f| f.get(&condition.0) <= Some(&condition.2))
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

    // XXX: we assume the user will chain filters correctly
    fn fn_ref_closure(self) -> impl Fn(&&Embedding) -> bool {
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
                    Compare::GtEq => e
                        .metadata
                        .as_ref()
                        .map(|f| f.get(&condition.0) >= Some(&condition.2))
                        .unwrap_or(false),
                    Compare::Lt => e
                        .metadata
                        .as_ref()
                        .map(|f| f.get(&condition.0) < Some(&condition.2))
                        .unwrap_or(false),
                    Compare::LtEq => e
                        .metadata
                        .as_ref()
                        .map(|f| f.get(&condition.0) <= Some(&condition.2))
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

impl Default for FilterBuilder {
    fn default() -> Self {
        Self::new()
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
        let permit = semaphore.clone().acquire_owned().await?;
        set.spawn_blocking(move || {
            let score = distance.compute(&embedding.vector, &query);
            drop(permit);
            ScoreIndex { score, index }
        });
    }

    let mut heap = BinaryHeap::new();
    while let Some(res) = set.join_next().await {
        let score_index = res.map_err(Into::<CollectionError>::into)?;
        if heap.len() < k || score_index < *heap.peek().ok_or(CollectionError::EmptyBinaryHeap)? {
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
        let db_path = temp_dir.path().join("embeddings");
        let mut db = match Db::open(db_path).await {
            Ok(db) => db,
            Err(err) => panic!("{}", err.to_string()),
        };
        let col = match db
            .create_collection("test".to_owned(), 5, Distance::Cosine)
            .await
        {
            Ok(col) => col,
            Err(err) => panic!("{}", err.to_string()),
        };
        let embedding = Embedding::new(
            vec![0.9999695, 0.76456239, 0.86767905, 0.17577756, 0.9949882],
            None,
        );
        col.write()
            .await
            .insert(embedding.clone())
            .expect("faield to insert embedding");

        let expected = SimilarityResult {
            score: 0.7449362,
            embedding,
        };
        let results = col
            .read()
            .await
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
        let db_path = temp_dir.path().join("embeddings");
        let mut db = match Db::open(db_path).await {
            Ok(db) => db,
            Err(err) => panic!("{}", err.to_string()),
        };
        let col = match db
            .create_collection("test".to_owned(), 5, Distance::Cosine)
            .await
        {
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
        col.write()
            .await
            .insert(embedding1.clone())
            .expect("faield to insert embedding");
        let embedding2 = Embedding::new(
            vec![0.43717385, 0.21100248, 0.5068433, 0.9626808, 0.6763327],
            Some(HashMap::from([
                ("i".to_owned(), 7.0.into()),
                ("j".to_owned(), 100.0.into()),
            ])),
        );
        col.write()
            .await
            .insert(embedding2.clone())
            .expect("faield to insert embedding");
        let embedding3 = Embedding::new(
            vec![0.2630481, 0.24888718, 0.3375401, 0.92770165, 0.44944693],
            Some(HashMap::from([
                ("i".to_owned(), 29.0.into()),
                ("j".to_owned(), 16.0.into()),
            ])),
        );
        col.write()
            .await
            .insert(embedding3.clone())
            .expect("faield to insert embedding");
        let embedding4 = Embedding::new(
            vec![0.7642892, 0.47043378, 0.9035855, 0.31120034, 0.5757918],
            Some(HashMap::from([
                ("i".to_owned(), 3.0.into()),
                ("j".to_owned(), 110.0.into()),
            ])),
        );
        col.write()
            .await
            .insert(embedding4.clone())
            .expect("faield to insert embedding");

        let results = col
            .read()
            .await
            .get(
                &[0.09537213, 0.5104327, 0.69980987, 0.13146928, 0.30541683],
                4,
                Some(
                    Collection::filter()
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

    #[tokio::test]
    async fn storage() {
        let temp_dir = TempDir::new().expect("failed to create tempt dir");
        let db_path = temp_dir.path().join("embeddings");

        let mut db = match Db::open(db_path.as_path()).await {
            Ok(db) => db,
            Err(err) => panic!("{}", err.to_string()),
        };
        assert!(db.inner.read().await.collections.is_empty());
        assert_eq!(db.inner.read().await.location, db_path);

        let col = match db
            .create_collection("test".to_owned(), 5, Distance::Cosine)
            .await
        {
            Ok(col) => col,
            Err(err) => panic!("{}", err.to_string()),
        };
        let embedding = Embedding::new(
            vec![0.9999695, 0.76456239, 0.86767905, 0.17577756, 0.9949882],
            None,
        );
        col.write()
            .await
            .insert(embedding.clone())
            .expect("faield to insert embedding");

        db.save().await.expect("failed to save to disk");
        let db = match Db::open(db_path).await {
            Ok(db) => db,
            Err(err) => panic!("{}", err.to_string()),
        };
        assert_eq!(db.inner.read().await.collections.len(), 1);
        let col = db
            .get_collection("test")
            .await
            .expect("failed to get collection");
        assert_eq!(col.read().await.len(), 1);
        assert_eq!(col.read().await.distance, Distance::Cosine);
        assert_eq!(col.read().await.dimension, 5);
    }
}
