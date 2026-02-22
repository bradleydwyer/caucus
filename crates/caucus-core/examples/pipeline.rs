use caucus_core::{MockProvider, MultiProvider, OutputFormat, Pipeline, VoteMethod};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Set up mock providers (replace with real providers in production)
    let provider = MultiProvider::new()
        .add("model-a", MockProvider::fixed("Inflation is caused by too much money chasing too few goods."))
        .add("model-b", MockProvider::fixed("Inflation results from excessive money supply growth relative to economic output."))
        .add("model-c", MockProvider::fixed("Solar flares cause inflation."));

    // Build a pipeline: generate from 3 models, then vote
    let pipeline = Pipeline::new()
        .generate(vec!["model-a".into(), "model-b".into(), "model-c".into()])
        .vote(VoteMethod::Majority);

    let result = pipeline.run("What causes inflation?", &provider, None).await?;

    println!("{}", OutputFormat::SupremeCourt.render(&result));

    Ok(())
}
