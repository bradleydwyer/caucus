use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use clap::Args;
use colored::Colorize;
use serde::{Deserialize, Serialize};

use caucus_core::{Candidate, ConsensusResult, OutputFormat, consensus};

use super::build_single_provider;

#[derive(Args)]
pub struct ServeArgs {
    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    pub port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Run as MCP server (stdio transport) instead of HTTP
    #[arg(long)]
    pub mcp: bool,
}

#[derive(Clone)]
struct AppState {}

pub async fn run(args: ServeArgs) -> anyhow::Result<()> {
    if args.mcp {
        return run_mcp().await;
    }

    let state = Arc::new(AppState {});

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/consensus", post(consensus_endpoint))
        .route("/v1/pipeline", post(pipeline_endpoint))
        .with_state(state)
        .layer(tower_http::cors::CorsLayer::permissive());

    let addr = format!("{}:{}", args.host, args.port);
    eprintln!("{} caucus API server listening on {}", "▶".green(), addr.cyan(),);
    eprintln!("  POST /v1/consensus  — Run consensus on candidates");
    eprintln!("  POST /v1/pipeline   — Run a multi-step pipeline");
    eprintln!("  GET  /health        — Health check");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Deserialize)]
struct ConsensusRequest {
    candidates: Vec<CandidateInput>,
    #[serde(default = "default_strategy")]
    strategy: String,
    #[serde(default = "default_format")]
    format: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum CandidateInput {
    Text(String),
    Full {
        content: String,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        confidence: Option<f64>,
    },
}

fn default_strategy() -> String {
    "majority_vote".to_string()
}
fn default_format() -> String {
    "json".to_string()
}

#[derive(Serialize)]
struct ConsensusResponse {
    content: String,
    strategy: String,
    agreement_score: f64,
    reasoning: Option<String>,
    dissent_count: usize,
    formatted: String,
}

impl From<(ConsensusResult, &OutputFormat)> for ConsensusResponse {
    fn from((result, format): (ConsensusResult, &OutputFormat)) -> Self {
        let formatted = format.render(&result);
        Self {
            content: result.content,
            strategy: result.strategy,
            agreement_score: result.agreement_score,
            reasoning: result.reasoning,
            dissent_count: result.dissents.len(),
            formatted,
        }
    }
}

async fn consensus_endpoint(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ConsensusRequest>,
) -> Result<Json<ConsensusResponse>, (StatusCode, String)> {
    let candidates: Vec<Candidate> = req
        .candidates
        .into_iter()
        .map(|input| match input {
            CandidateInput::Text(text) => Candidate::new(text),
            CandidateInput::Full { content, model, confidence } => {
                let mut c = Candidate::new(content);
                if let Some(m) = model {
                    c = c.with_model(m);
                }
                if let Some(conf) = confidence {
                    c = c.with_confidence(conf);
                }
                c
            }
        })
        .collect();

    let format: OutputFormat = req
        .format
        .parse()
        .map_err(|e: anyhow::Error| (StatusCode::BAD_REQUEST, format!("Invalid format: {e}")))?;

    // Build judge LLM if needed
    let judge_llm: Option<Box<dyn caucus_core::LlmProvider>> = if strategy_needs_llm(&req.strategy)
    {
        // Try to use the first candidate's model, or fall back to env-configured default
        let model_name = candidates
            .first()
            .and_then(|c| c.model.clone())
            .unwrap_or_else(|| "gpt-4o".to_string());
        build_single_provider(&model_name).ok()
    } else {
        None
    };

    let result = consensus(&candidates, &req.strategy, judge_llm.as_deref())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Consensus error: {e}")))?;

    Ok(Json(ConsensusResponse::from((result, &format))))
}

#[derive(Deserialize)]
struct PipelineRequest {
    prompt: String,
    #[serde(default)]
    models: Vec<String>,
    #[serde(default)]
    pipeline: Vec<PipelineStepInput>,
    #[serde(default = "default_format")]
    format: String,
}

#[derive(Deserialize)]
struct PipelineStepInput {
    step: String,
    #[serde(default)]
    rounds: Option<usize>,
    #[serde(default)]
    method: Option<String>,
}

async fn pipeline_endpoint(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<PipelineRequest>,
) -> Result<Json<ConsensusResponse>, (StatusCode, String)> {
    use caucus_core::{Pipeline, VoteMethod};

    let format: OutputFormat = req
        .format
        .parse()
        .map_err(|e: anyhow::Error| (StatusCode::BAD_REQUEST, format!("Invalid format: {e}")))?;

    let mut pipeline = Pipeline::new();

    for step in &req.pipeline {
        pipeline = match step.step.as_str() {
            "generate" => pipeline.generate(req.models.clone()),
            "debate" => pipeline.debate(step.rounds.unwrap_or(3)),
            "vote" => {
                let method = match step.method.as_deref() {
                    Some("weighted") => VoteMethod::Weighted,
                    _ => VoteMethod::Majority,
                };
                pipeline.vote(method)
            }
            "judge" | "synthesize" => pipeline.judge(),
            other => {
                return Err((StatusCode::BAD_REQUEST, format!("Unknown pipeline step: {other}")));
            }
        };
    }

    let provider = super::build_provider(&req.models)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Provider error: {e}")))?;

    // Use first model as judge
    let judge_llm: Option<Box<dyn caucus_core::LlmProvider>> =
        req.models.first().and_then(|m| build_single_provider(m).ok());

    let result = pipeline
        .run(&req.prompt, &provider, judge_llm.as_deref())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Pipeline error: {e}")))?;

    Ok(Json(ConsensusResponse::from((result, &format))))
}

fn strategy_needs_llm(name: &str) -> bool {
    matches!(
        name,
        "judge"
            | "judge_synthesis"
            | "judge-synthesis"
            | "debate"
            | "multi_round_debate"
            | "multi-round-debate"
            | "debate_then_vote"
            | "debate-then-vote"
    )
}

async fn run_mcp() -> anyhow::Result<()> {
    // MCP server using stdio transport
    // Reads JSON-RPC requests from stdin, writes responses to stdout
    use tokio::io::{AsyncBufReadExt, BufReader};

    eprintln!("{} caucus MCP server started (stdio)", "▶".green());

    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    // Send server capabilities
    let capabilities = serde_json::json!({
        "jsonrpc": "2.0",
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "caucus",
                "version": env!("CARGO_PKG_VERSION")
            }
        }
    });

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let request: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let method = request["method"].as_str().unwrap_or("");
        let id = &request["id"];

        let response = match method {
            "initialize" => {
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": capabilities["result"]
                })
            }
            "tools/list" => {
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "tools": [
                            {
                                "name": "consensus",
                                "description": "Run consensus across multiple LLM responses",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "candidates": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "List of response texts to evaluate"
                                        },
                                        "strategy": {
                                            "type": "string",
                                            "enum": ["majority_vote", "weighted_vote", "judge", "debate"],
                                            "default": "majority_vote"
                                        }
                                    },
                                    "required": ["candidates"]
                                }
                            }
                        ]
                    }
                })
            }
            "tools/call" => {
                let tool_name = request["params"]["name"].as_str().unwrap_or("");
                match tool_name {
                    "consensus" => {
                        let args = &request["params"]["arguments"];
                        let candidate_texts: Vec<String> = args["candidates"]
                            .as_array()
                            .map(|arr| {
                                arr.iter().filter_map(|v| v.as_str().map(String::from)).collect()
                            })
                            .unwrap_or_default();

                        let strategy = args["strategy"].as_str().unwrap_or("majority_vote");

                        let candidates: Vec<Candidate> =
                            candidate_texts.into_iter().map(Candidate::new).collect();

                        match consensus(&candidates, strategy, None).await {
                            Ok(result) => {
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "result": {
                                        "content": [{
                                            "type": "text",
                                            "text": serde_json::to_string_pretty(&result).unwrap()
                                        }]
                                    }
                                })
                            }
                            Err(e) => {
                                serde_json::json!({
                                    "jsonrpc": "2.0",
                                    "id": id,
                                    "result": {
                                        "content": [{
                                            "type": "text",
                                            "text": format!("Error: {}", e)
                                        }],
                                        "isError": true
                                    }
                                })
                            }
                        }
                    }
                    _ => {
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32601,
                                "message": format!("Unknown tool: {}", tool_name)
                            }
                        })
                    }
                }
            }
            _ => {
                serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32601,
                        "message": format!("Unknown method: {}", method)
                    }
                })
            }
        };

        println!("{}", serde_json::to_string(&response)?);
    }

    Ok(())
}
