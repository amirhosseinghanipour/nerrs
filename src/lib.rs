//! # nerrs
//!
//! CRF-based Named Entity Recognition for Persian text.
//!
//! Uses [`crfrs`] for model training and Viterbi inference.
//!
//! ## Entity types
//!
//! The default tagset uses BIO encoding over these entity classes:
//!
//! | Tag | Meaning |
//! |-----|---------|
//! | `O` | Outside any entity |
//! | `B-PER` / `I-PER` | Person name |
//! | `B-ORG` / `I-ORG` | Organisation |
//! | `B-LOC` / `I-LOC` | Location |
//! | `B-DAT` / `I-DAT` | Date |
//! | `B-TIM` / `I-TIM` | Time |
//! | `B-MON` / `I-MON` | Money / currency amount |
//! | `B-PCT` / `I-PCT` | Percentage |
//! | `B-EVE` / `I-EVE` | Event |
//!
//! ## Quick start
//!
//! ```no_run
//! use nerrs::NerTagger;
//!
//! let mut tagger = NerTagger::new();
//! tagger.load_model("ner.model").unwrap();
//! let entities = tagger.tag(&["علی", "به", "تهران", "رفت", "."]).unwrap();
//! // → [("علی","B-PER"), ("به","O"), ("تهران","B-LOC"), ("رفت","O"), (".","O")]
//! ```
//!
//! ## Training
//!
//! ```no_run
//! use nerrs::{NerTagger, crfrs::TrainConfig};
//!
//! // IOB-tagged corpus: Vec<Vec<(word, ner_tag)>>
//! let corpus: Vec<Vec<(String, String)>> = vec![];
//! NerTagger::train_and_save(&corpus, "ner.model", TrainConfig::default()).unwrap();
//! ```

#![warn(missing_docs)]

/// Error types for nerrs.
pub mod error;
pub mod features;
pub mod tagger;

pub use crfrs;
pub use error::{Error, Result};
pub use tagger::NerTagger;

/// Default NER labels in BIO encoding.
pub const DEFAULT_LABELS: &[&str] = &[
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-DAT", "I-DAT",
    "B-TIM", "I-TIM",
    "B-MON", "I-MON",
    "B-PCT", "I-PCT",
    "B-EVE", "I-EVE",
];
