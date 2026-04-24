use std::{any::Any, sync::Arc};

use arrow::{
    array::{Int64Array, RecordBatch, StringArray},
    datatypes::SchemaRef,
};
use async_stream::try_stream;
use datafusion::{
    catalog::{Session, TableProvider},
    common::exec_datafusion_err,
    datasource::TableType,
    logical_expr::{Expr, TableProviderFilterPushDown},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
        ExecutionPlan, PlanProperties,
    },
};

use crate::schemas;

/// User-facing chain proof metadata exposed through Flight SQL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainProof {
    /// The CAIP-2 chain ID of the proof.
    pub chain_id: String,
    /// The blockchain transaction hash.
    pub transaction_hash: String,
    /// The transaction input that carried the anchored root.
    pub transaction_input: String,
    /// The block hash that included the transaction.
    pub block_hash: String,
    /// The block timestamp as a unix timestamp.
    pub timestamp: i64,
}

/// Access to chain proof metadata.
#[async_trait::async_trait]
pub trait ChainProofFeed: std::fmt::Debug + Send + Sync {
    /// Return all persisted chain proofs available to the node.
    async fn chain_proofs(&self) -> anyhow::Result<Vec<ChainProof>>;
}

#[async_trait::async_trait]
impl<T: ChainProofFeed> ChainProofFeed for Arc<T> {
    async fn chain_proofs(&self) -> anyhow::Result<Vec<ChainProof>> {
        self.as_ref().chain_proofs().await
    }
}

fn chain_proofs_to_record_batch(proofs: &[ChainProof]) -> datafusion::common::Result<RecordBatch> {
    let chain_ids =
        StringArray::from_iter_values(proofs.iter().map(|proof| proof.chain_id.as_str()));
    let transaction_hashes =
        StringArray::from_iter_values(proofs.iter().map(|proof| proof.transaction_hash.as_str()));
    let transaction_inputs =
        StringArray::from_iter_values(proofs.iter().map(|proof| proof.transaction_input.as_str()));
    let block_hashes =
        StringArray::from_iter_values(proofs.iter().map(|proof| proof.block_hash.as_str()));
    let timestamps = Int64Array::from_iter_values(proofs.iter().map(|proof| proof.timestamp));

    Ok(RecordBatch::try_new(
        schemas::chain_proofs(),
        vec![
            Arc::new(chain_ids),
            Arc::new(transaction_hashes),
            Arc::new(transaction_inputs),
            Arc::new(block_hashes),
            Arc::new(timestamps),
        ],
    )?)
}

/// A TableProvider for chain proof metadata.
#[derive(Debug)]
pub struct ChainProofTable<T> {
    feed: Arc<T>,
    schema: SchemaRef,
}

impl<T> ChainProofTable<T> {
    /// Create a new chain proof table.
    pub fn new(feed: Arc<T>) -> Self {
        Self {
            feed,
            schema: schemas::chain_proofs(),
        }
    }
}

#[async_trait::async_trait]
impl<T: ChainProofFeed + std::fmt::Debug + 'static> TableProvider for ChainProofTable<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let schema = projection
            .map(|projection| self.schema.project(projection))
            .transpose()?
            .map(Arc::new)
            .unwrap_or_else(|| self.schema.clone());

        Ok(Arc::new(ChainProofExec {
            feed: Arc::clone(&self.feed),
            schema: schema.clone(),
            projection: projection.cloned(),
            properties: PlanProperties::new(
                EquivalenceProperties::new(schema),
                datafusion::physical_plan::Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            ),
        }))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Unsupported)
            .collect())
    }
}

#[derive(Debug)]
struct ChainProofExec<T> {
    feed: Arc<T>,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    properties: PlanProperties,
}

impl<T> datafusion::physical_plan::DisplayAs for ChainProofExec<T> {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        write!(f, "ChainProofExec")
    }
}

impl<T: ChainProofFeed + std::fmt::Debug + 'static> ExecutionPlan for ChainProofExec<T> {
    fn name(&self) -> &str {
        "ChainProofExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<datafusion::execution::context::TaskContext>,
    ) -> datafusion::common::Result<datafusion::execution::SendableRecordBatchStream> {
        if partition != 0 {
            return Err(exec_datafusion_err!(
                "ChainProofExec only supports a single partition"
            ));
        }

        let feed = Arc::clone(&self.feed);
        let projection = self.projection.clone();
        let schema = self.schema.clone();

        let stream = try_stream! {
            let proofs = feed
                .chain_proofs()
                .await
                .map_err(|err| exec_datafusion_err!("{err}"))?;
            let batch = chain_proofs_to_record_batch(&proofs)?;
            let batch = if let Some(projection) = projection {
                batch.project(&projection)?
            } else {
                batch
            };
            yield batch;
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::util::pretty::pretty_format_batches;
    use datafusion::prelude::SessionContext;
    use expect_test::expect;
    use test_log::test;

    use crate::{
        chain_proof::{ChainProof, ChainProofTable},
        tests::MockConclusionFeed,
    };

    #[test(tokio::test)]
    async fn can_query_chain_proofs_table() {
        let mut mock_feed = MockConclusionFeed::new();
        mock_feed.expect_chain_proofs().once().return_once(|| {
            Ok(vec![ChainProof {
                chain_id: "eip155:11155111".to_owned(),
                transaction_hash: "0xabc".to_owned(),
                transaction_input: "0xdef".to_owned(),
                block_hash: "0x123".to_owned(),
                timestamp: 42,
            }])
        });

        let table = ChainProofTable::new(Arc::new(mock_feed));
        let ctx = SessionContext::new();
        let data = pretty_format_batches(
            &ctx.read_table(Arc::new(table))
                .unwrap()
                .select_columns(&["chain_id", "transaction_hash", "timestamp"])
                .unwrap()
                .collect()
                .await
                .unwrap(),
        )
        .unwrap();

        expect![[r#"
            +-----------------+------------------+-----------+
            | chain_id        | transaction_hash | timestamp |
            +-----------------+------------------+-----------+
            | eip155:11155111 | 0xabc            | 42        |
            +-----------------+------------------+-----------+"#]]
        .assert_eq(&data.to_string());
    }
}
