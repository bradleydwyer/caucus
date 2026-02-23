use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// A candidate response submitted for consensus.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct Candidate {
    #[pyo3(get, set)]
    content: String,
    #[pyo3(get, set)]
    model: Option<String>,
    #[pyo3(get, set)]
    confidence: Option<f64>,
}

#[pymethods]
impl Candidate {
    #[new]
    #[pyo3(signature = (content, model=None, confidence=None))]
    fn new(content: String, model: Option<String>, confidence: Option<f64>) -> Self {
        Self { content, model, confidence }
    }

    fn __repr__(&self) -> String {
        format!(
            "Candidate(content={:?}, model={:?}, confidence={:?})",
            self.content, self.model, self.confidence
        )
    }
}

impl From<&Candidate> for caucus_core::Candidate {
    fn from(py_candidate: &Candidate) -> Self {
        let mut c = caucus_core::Candidate::new(&py_candidate.content);
        if let Some(model) = &py_candidate.model {
            c = c.with_model(model.clone());
        }
        if let Some(conf) = py_candidate.confidence {
            c = c.with_confidence(conf);
        }
        c
    }
}

/// The result of a consensus operation.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct ConsensusResult {
    #[pyo3(get)]
    content: String,
    #[pyo3(get)]
    strategy: String,
    #[pyo3(get)]
    agreement_score: f64,
    #[pyo3(get)]
    reasoning: Option<String>,
    #[pyo3(get)]
    dissent_count: usize,
}

#[pymethods]
impl ConsensusResult {
    fn __repr__(&self) -> String {
        format!(
            "ConsensusResult(strategy={:?}, agreement={:.0}%, dissents={})",
            self.strategy,
            self.agreement_score * 100.0,
            self.dissent_count,
        )
    }

    /// Serialize to a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("content", &self.content)?;
        dict.set_item("strategy", &self.strategy)?;
        dict.set_item("agreement_score", self.agreement_score)?;
        dict.set_item("dissent_count", self.dissent_count)?;
        if let Some(r) = &self.reasoning {
            dict.set_item("reasoning", r)?;
        }
        Ok(dict.into_any().unbind())
    }
}

impl From<caucus_core::ConsensusResult> for ConsensusResult {
    fn from(r: caucus_core::ConsensusResult) -> Self {
        Self {
            content: r.content,
            strategy: r.strategy,
            agreement_score: r.agreement_score,
            reasoning: r.reasoning,
            dissent_count: r.dissents.len(),
        }
    }
}

/// Run consensus on a list of candidates using the specified strategy.
///
/// Args:
///     candidates: List of Candidate objects or strings
///     strategy: Strategy name ("majority_vote", "weighted_vote", "judge", "debate", etc.)
///
/// Returns:
///     ConsensusResult
#[pyfunction]
#[pyo3(signature = (candidates, strategy="majority_vote"))]
fn consensus(candidates: Vec<Candidate>, strategy: &str) -> PyResult<ConsensusResult> {
    let core_candidates: Vec<caucus_core::Candidate> =
        candidates.iter().map(|c| c.into()).collect();

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {e}")))?;

    let result =
        rt.block_on(async { caucus_core::consensus(&core_candidates, strategy, None).await });

    match result {
        Ok(r) => Ok(r.into()),
        Err(e) => Err(PyValueError::new_err(format!("Consensus error: {e}"))),
    }
}

/// caucus: Multi-LLM consensus engine
#[pymodule]
fn caucus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Candidate>()?;
    m.add_class::<ConsensusResult>()?;
    m.add_function(wrap_pyfunction!(consensus, m)?)?;
    Ok(())
}
