//! Shared whole-run wall-clock deadline.

use std::future::Future;
use std::time::Duration;

use anyhow::Result;
use tokio::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct RunDeadline {
    started: Instant,
    limit: Option<Duration>,
}

impl RunDeadline {
    pub fn new(seconds: Option<u64>) -> Self {
        Self { started: Instant::now(), limit: seconds.map(Duration::from_secs) }
    }

    /// Clamp a request/process timeout to the time remaining in the run.
    pub fn turn_timeout(&self, requested: Duration) -> Result<Duration> {
        match self.remaining()? {
            Some(remaining) => Ok(requested.min(remaining)),
            None => Ok(requested),
        }
    }

    /// Run a phase within the remaining whole-run wall-clock budget.
    pub async fn wait<F: Future>(&self, future: F) -> Result<F::Output> {
        match self.remaining()? {
            Some(remaining) => tokio::time::timeout(remaining, future)
                .await
                .map_err(|_| anyhow::anyhow!("whole-run deadline exceeded")),
            None => Ok(future.await),
        }
    }

    fn remaining(&self) -> Result<Option<Duration>> {
        let Some(limit) = self.limit else { return Ok(None) };
        let remaining = limit
            .checked_sub(self.started.elapsed())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| anyhow::anyhow!("whole-run deadline exceeded"))?;
        Ok(Some(remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn deadline_bounds_the_whole_run_and_each_turn() {
        let deadline = RunDeadline::new(Some(10));
        let initial = deadline.turn_timeout(Duration::from_secs(30)).unwrap();
        assert!(initial <= Duration::from_secs(10) && initial > Duration::from_secs(9));
        tokio::time::advance(Duration::from_secs(4)).await;
        let remaining = deadline.turn_timeout(Duration::from_secs(30)).unwrap();
        assert!(remaining <= Duration::from_secs(6) && remaining > Duration::from_secs(5));
        let result = deadline
            .wait(async {
                tokio::time::sleep(Duration::from_secs(7)).await;
            })
            .await;
        assert!(result.is_err());
    }
}
