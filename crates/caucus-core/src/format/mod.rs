pub mod plain;
pub mod structured;
pub mod supreme_court;
pub mod detailed;

use anyhow::Result;
use crate::types::ConsensusResult;

/// Output format for rendering consensus results.
pub enum OutputFormat {
    Plain,
    Json,
    SupremeCourt,
    Detailed,
}

impl OutputFormat {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "plain" | "text" => Ok(Self::Plain),
            "json" => Ok(Self::Json),
            "supreme-court" | "supreme_court" => Ok(Self::SupremeCourt),
            "detailed" | "debug" => Ok(Self::Detailed),
            other => anyhow::bail!("Unknown format: {other}"),
        }
    }

    pub fn render(&self, result: &ConsensusResult) -> String {
        match self {
            Self::Plain => plain::render(result),
            Self::Json => structured::render(result),
            Self::SupremeCourt => supreme_court::render(result),
            Self::Detailed => detailed::render(result),
        }
    }
}
