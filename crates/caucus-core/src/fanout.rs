//! Bounded concurrent fan-out.
//!
//! [`bounded_fanout`] runs named tasks with a hard concurrency bound and
//! returns successes and failures without losing either, in input order.
//! The provider-specific fan-out used by the engine lives in
//! [`crate::provider::fanout`].

use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

/// A normalized per-task failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureRecord {
    pub name: String,
    pub error: String,
}

/// Aggregate outcome of a [`bounded_fanout`] run.
#[derive(Debug, Clone)]
pub struct FanoutBatch<T> {
    /// Successful results, in input order.
    pub successes: Vec<(String, T)>,
    /// Failed tasks, in input order.
    pub failures: Vec<FailureRecord>,
}

// Manual impl: `#[derive(Default)]` would require `T: Default`.
impl<T> Default for FanoutBatch<T> {
    fn default() -> Self {
        Self { successes: Vec::new(), failures: Vec::new() }
    }
}

impl<T> FanoutBatch<T> {
    pub fn success_count(&self) -> usize {
        self.successes.len()
    }

    pub fn quorum_met(&self, quorum: usize) -> bool {
        self.successes.len() >= quorum
    }
}

/// Run named tasks concurrently, bounded by `max_concurrency`.
/// Results are returned in input order regardless of completion order.
pub async fn bounded_fanout<T, F, Fut>(
    tasks: Vec<(String, F)>,
    max_concurrency: usize,
) -> FanoutBatch<T>
where
    T: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<T>> + Send + 'static,
{
    let semaphore = Arc::new(Semaphore::new(max_concurrency.max(1)));
    let mut set: JoinSet<(usize, String, Result<T>)> = JoinSet::new();

    for (index, (name, task)) in tasks.into_iter().enumerate() {
        let permit = Arc::clone(&semaphore);
        set.spawn(async move {
            let _permit = permit.acquire_owned().await.expect("semaphore open");
            (index, name, task().await)
        });
    }

    let mut ordered: Vec<(usize, String, Result<T>)> = Vec::new();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(outcome) => ordered.push(outcome),
            Err(err) => tracing::warn!("fan-out task failed to join: {err}"),
        }
    }
    ordered.sort_by_key(|(index, _, _)| *index);

    let mut batch = FanoutBatch::default();
    for (_, name, result) in ordered {
        match result {
            Ok(value) => batch.successes.push((name, value)),
            Err(err) => batch.failures.push(FailureRecord { name, error: format!("{err:#}") }),
        }
    }
    batch
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[tokio::test]
    async fn bounded_fanout_collects_in_order() {
        let tasks: Vec<(String, _)> = (0..4)
            .map(|i| {
                (format!("t{i}"), move || async move {
                    tokio::time::sleep(Duration::from_millis((4 - i) * 10)).await;
                    Ok(i)
                })
            })
            .collect();
        let batch = bounded_fanout(tasks, 2).await;
        let names: Vec<&str> = batch.successes.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["t0", "t1", "t2", "t3"]);
        assert!(batch.quorum_met(4));
    }

    #[tokio::test]
    async fn bounded_fanout_keeps_failures() {
        use std::future::Future;
        use std::pin::Pin;

        type BoxedTask =
            Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = Result<i32>> + Send>> + Send>;
        let tasks: Vec<(String, BoxedTask)> = vec![
            ("ok".to_string(), Box::new(|| Box::pin(async { Ok(1) }))),
            ("bad".to_string(), Box::new(|| Box::pin(async { Err(anyhow::anyhow!("nope")) }))),
        ];
        let batch = bounded_fanout(tasks, 2).await;
        assert_eq!(batch.success_count(), 1);
        assert_eq!(batch.failures.len(), 1);
        assert_eq!(batch.failures[0].name, "bad");
        assert!(batch.failures[0].error.contains("nope"));
    }

    #[tokio::test]
    async fn bounded_fanout_bounds_concurrency() {
        let current = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let tasks: Vec<(String, _)> = (0..6)
            .map(|i| {
                let current = Arc::clone(&current);
                let peak = Arc::clone(&peak);
                (format!("t{i}"), move || async move {
                    let now = current.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    current.fetch_sub(1, Ordering::SeqCst);
                    Ok(i)
                })
            })
            .collect();
        let batch = bounded_fanout(tasks, 2).await;
        assert_eq!(batch.success_count(), 6);
        assert!(peak.load(Ordering::SeqCst) <= 2);
    }
}
