# nerrs

CRF-based Named Entity Recognition for Persian text, written in Rust.

Uses the averaged structured perceptron from [`crfrs`](https://crates.io/crates/crfrs) with Viterbi decoding. You supply a BIO-tagged corpus; the crate handles feature extraction, training, and inference.

## Features

- BIO-encoded NER with 9 entity types (PER, ORG, LOC, DAT, TIM, MON, PCT, EVE)
- Persian-specific feature extraction (character n-grams, orthographic flags, context window)
- Train from any IOB-tagged corpus
- Save/load models as JSON
- Token-level evaluation

## Usage

```toml
[dependencies]
nerrs = "0.1"
```

```rust
use nerrs::{NerTagger, crfrs::TrainConfig};

// Train
let corpus: Vec<Vec<(String, String)>> = vec![/* (word, BIO-tag) pairs */];
NerTagger::train_and_save(&corpus, "ner.model", TrainConfig::default()).unwrap();

// Tag
let mut tagger = NerTagger::new();
tagger.load_model("ner.model").unwrap();
let tags = tagger.tag(&["علی", "به", "تهران", "رفت", "."]).unwrap();
// → [("علی","B-PER"), ("به","O"), ("تهران","B-LOC"), ("رفت","O"), (".","O")]
```

## API

| Item | Description |
|---|---|
| `NerTagger::new()` | Create a tagger with no loaded model |
| `NerTagger::train_and_save(corpus, path, config)` | Train and save model to disk |
| `NerTagger::fit(&mut self, corpus, path, config)` | Train, save, and load in one step |
| `NerTagger::load_model(path)` | Load a previously saved model |
| `NerTagger::tag(words)` | Tag a single sentence |
| `NerTagger::tag_sents(sents)` | Tag multiple sentences |
| `NerTagger::evaluate(test)` | Token-level accuracy on a labelled corpus |

## Entity types

| Tag | Meaning |
|---|---|
| `O` | Outside any entity |
| `B-PER` / `I-PER` | Person name |
| `B-ORG` / `I-ORG` | Organisation |
| `B-LOC` / `I-LOC` | Location |
| `B-DAT` / `I-DAT` | Date |
| `B-TIM` / `I-TIM` | Time |
| `B-MON` / `I-MON` | Money / currency amount |
| `B-PCT` / `I-PCT` | Percentage |
| `B-EVE` / `I-EVE` | Event |

## License

MIT
