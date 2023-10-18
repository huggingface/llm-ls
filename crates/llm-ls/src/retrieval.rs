use arrow_array::{RecordBatch, RecordBatchIterator, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use std::{path::PathBuf, sync::Arc};
use vectordb::{Database, Table};

async fn initialse_database(cache_path: PathBuf) -> Table {
    let uri = cache_path.join("database");
    let db = Database::connect(uri.to_str().expect("path should be utf8"))
        .await
        .expect("failed to open database");
    let table = match db.open_table("code-slices").await {
        Ok(table) => table,
        Err(vectordb::error::Error::TableNotFound { .. }) => {
            let schema = Schema::new(vec![
                Field::new("workspace_root", DataType::Utf8, false),
                Field::new("file_url", DataType::Utf8, false),
                Field::new("start_line_no", DataType::UInt32, false),
                Field::new("end_line_no", DataType::UInt32, false),
                Field::new("window_size", DataType::UInt32, false),
            ]);
            let batch = RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(StringArray::from(Vec::<&str>::new())),
                    Arc::new(StringArray::from(Vec::<&str>::new())),
                    Arc::new(UInt32Array::from(Vec::<u32>::new())),
                    Arc::new(UInt32Array::from(Vec::<u32>::new())),
                    Arc::new(UInt32Array::from(Vec::<u32>::new())),
                ],
            )
            .expect("failure while defining schema");
            db.create_table(
                "code-slices",
                RecordBatchIterator::new(vec![batch.clone()].into_iter().map(Ok), batch.schema()),
                None,
            )
            .await
            .expect("failed to create table")
        }
        Err(err) => panic!("error while opening table: {}", err),
    };
    table
}

pub(crate) struct SnippetRetriever {
    db: Table,
}

impl SnippetRetriever {
    /// # Panics
    ///
    /// Panics if the database cannot be initialised.
    pub(crate) async fn new(cache_path: PathBuf) -> Self {
        Self {
            db: initialse_database(cache_path).await,
        }
    }

    pub(crate) async fn build_workspace_snippets(workspace_root: String) {}

    pub(crate) async fn update_document(file_url: String) {}
}
