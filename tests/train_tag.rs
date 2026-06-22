use nerrs::{NerTagger, crfrs::TrainConfig};

// Helpers ----------------------------------------------------------------

/// Simple 5-token BIO-tagged corpus repeated `n` times.
/// "علی در تهران زندگی می‌کند ."
///   PER  O   LOC    O      O   O
fn corpus(n: usize) -> Vec<Vec<(String, String)>> {
    let sent = vec![
        ("علی".to_string(),      "B-PER".to_string()),
        ("در".to_string(),       "O".to_string()),
        ("تهران".to_string(),    "B-LOC".to_string()),
        ("زندگی".to_string(),    "O".to_string()),
        ("می‌کند".to_string(),  "O".to_string()),
        (".".to_string(),        "O".to_string()),
    ];
    vec![sent; n]
}

fn tmp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

// Tests ------------------------------------------------------------------

#[test]
fn train_save_load_tag_cycle() {
    let path = tmp_path("nerrs_train_tag.json");
    NerTagger::train_and_save(&corpus(50), &path, TrainConfig { epochs: 10 }).unwrap();

    let mut tagger = NerTagger::new();
    tagger.load_model(&path).unwrap();

    let result = tagger
        .tag(&["علی", "در", "تهران", "زندگی", "می‌کند", "."])
        .unwrap();

    assert_eq!(result.len(), 6);
    assert_eq!(result[0].1, "B-PER", "علی should be B-PER");
    assert_eq!(result[2].1, "B-LOC", "تهران should be B-LOC");

    std::fs::remove_file(&path).ok();
}

#[test]
fn evaluate_accuracy_on_training_data() {
    let data = corpus(50);
    let path = tmp_path("nerrs_eval.json");
    NerTagger::train_and_save(&data, &path, TrainConfig { epochs: 10 }).unwrap();

    let mut tagger = NerTagger::new();
    tagger.load_model(&path).unwrap();

    let acc = tagger.evaluate(&data).unwrap();
    assert_eq!(acc, 1.0, "expected 1.0 accuracy on training data, got {acc}");

    std::fs::remove_file(&path).ok();
}

#[test]
fn tag_empty_returns_empty() {
    let path = tmp_path("nerrs_empty.json");
    NerTagger::train_and_save(&corpus(10), &path, TrainConfig { epochs: 3 }).unwrap();

    let mut tagger = NerTagger::new();
    tagger.load_model(&path).unwrap();

    assert!(tagger.tag(&[]).unwrap().is_empty());
    std::fs::remove_file(&path).ok();
}

#[test]
fn model_not_loaded_returns_error() {
    let tagger = NerTagger::new();
    assert!(tagger.tag(&["سلام"]).is_err());
}

#[test]
fn fit_loads_model_immediately() {
    let path = tmp_path("nerrs_fit.json");
    let mut tagger = NerTagger::new();
    tagger
        .fit(&corpus(30), &path, TrainConfig { epochs: 10 })
        .unwrap();

    // fit() should have loaded the model — tag must succeed without a separate load_model call.
    assert!(tagger.is_loaded());
    let result = tagger.tag(&["علی", "در", "تهران"]).unwrap();
    assert_eq!(result.len(), 3);

    std::fs::remove_file(&path).ok();
}

#[test]
fn tag_sents_matches_individual_tag_calls() {
    let path = tmp_path("nerrs_tag_sents.json");
    NerTagger::train_and_save(&corpus(30), &path, TrainConfig { epochs: 10 }).unwrap();

    let mut tagger = NerTagger::new();
    tagger.load_model(&path).unwrap();

    let sents: Vec<Vec<&str>> = vec![
        vec!["علی", "در", "تهران"],
        vec!["در", "."],
    ];
    let batch = tagger.tag_sents(&sents).unwrap();
    let singles: Vec<_> = sents.iter().map(|s| tagger.tag(s).unwrap()).collect();

    assert_eq!(batch, singles);
    std::fs::remove_file(&path).ok();
}
