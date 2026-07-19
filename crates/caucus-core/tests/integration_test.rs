use caucus_core::*;

#[tokio::test]
async fn test_majority_vote_with_agreement() {
    let candidates = vec![
        Candidate::new("Paris is the capital of France"),
        Candidate::new("Paris is the capital of France"),
        Candidate::new("I think the answer involves quantum mechanics and string theory"),
    ];

    let result = consensus(&candidates, "majority_vote", None).await.unwrap();
    assert!(result.agreement_score > 0.5);
    assert_eq!(result.dissents.len(), 1);
    assert!(result.content.contains("Paris"));
}

#[tokio::test]
async fn test_weighted_vote_with_confidence() {
    let candidates = vec![
        Candidate::new("The answer is 42").with_confidence(0.95).with_model("expert-model"),
        Candidate::new("Something completely unrelated to anything")
            .with_confidence(0.1)
            .with_model("bad-model"),
        Candidate::new("The answer is 42").with_confidence(0.85).with_model("good-model"),
    ];

    let result = consensus(&candidates, "weighted_vote", None).await.unwrap();
    assert!(result.agreement_score > 0.5);
    assert!(result.content.contains("42"));
}

#[tokio::test]
async fn test_judge_synthesis_with_mock() {
    let judge_response = serde_json::json!({
        "synthesis": "Paris is the capital of France, located in the Île-de-France region.",
        "reasoning": "Both responses agree on Paris; Response 1 is more concise, Response 2 adds geographic context.",
        "agreement_score": 0.85,
        "dissent_indices": []
    });

    let provider = MockProvider::fixed(judge_response.to_string());
    let candidates = vec![
        Candidate::new("Paris is the capital of France"),
        Candidate::new("The capital of France is Paris, in the Île-de-France region"),
    ];

    let result = consensus(&candidates, "judge", Some(&provider)).await.unwrap();
    assert!(result.content.contains("Paris"));
    assert_eq!(result.agreement_score, 0.85);
}

#[tokio::test]
async fn test_pipeline_generate_vote() {
    let provider = MultiProvider::new()
        .add("a", MockProvider::fixed("The answer is 42"))
        .add("b", MockProvider::fixed("The answer is 42"))
        .add("c", MockProvider::fixed("The answer is 7"));

    let pipeline = Pipeline::new()
        .generate(vec!["a".into(), "b".into(), "c".into()])
        .vote(VoteMethod::Majority);

    let result = pipeline.run("What is the answer?", &provider, None).await.unwrap();
    assert_eq!(result.content, "The answer is 42");
    assert!(result.agreement_score > 0.5);
}

#[tokio::test]
async fn test_output_formats() {
    let candidates = vec![
        Candidate::new("Test response").with_model("model-a"),
        Candidate::new("Different response").with_model("model-b"),
    ];
    let result = consensus(&candidates, "majority_vote", None).await.unwrap();

    // Plain format
    let plain = OutputFormat::Plain.render(&result);
    assert!(!plain.is_empty());

    // JSON format
    let json = OutputFormat::Json.render(&result);
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed["content"].is_string());

    // Supreme Court format
    let sc = OutputFormat::SupremeCourt.render(&result);
    assert!(sc.contains("MAJORITY OPINION"));
    assert!(sc.contains("VOTE SUMMARY"));

    // Detailed format
    let detailed = OutputFormat::Detailed.render(&result);
    assert!(detailed.contains("CANDIDATES"));
}

#[tokio::test]
async fn test_all_strategies_from_name() {
    let names = [
        "majority_vote",
        "majority-vote",
        "weighted_vote",
        "weighted-vote",
        "judge",
        "judge_synthesis",
        "debate",
        "multi_round_debate",
        "debate_then_vote",
    ];

    for name in names {
        let strategy = strategy_from_name(name);
        assert!(strategy.is_ok(), "Failed to create strategy: {name}");
    }
}
