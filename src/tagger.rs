//! CRF-based Persian NER tagger.

use std::path::Path;
use crfrs::{train, PerceptronModel, TrainConfig};
use crate::{
    DEFAULT_LABELS,
    error::{Error, Result},
    features::sentence_ner_features,
};

/// A `(word, ner_tag)` pair.
pub type TaggedToken = (String, String);
/// A fully NER-tagged sentence.
pub type TaggedSentence = Vec<TaggedToken>;

/// CRF-based Named Entity Recognition tagger for Persian text.
///
/// # Lifecycle
///
/// 1. Call [`NerTagger::new`] to create an instance.
/// 2. Call [`load_model`](Self::load_model) to attach a trained model, **or**
///    call [`train_and_save`](Self::train_and_save) to train and immediately load.
/// 3. Call [`tag`](Self::tag) / [`tag_sents`](Self::tag_sents) to annotate text.
pub struct NerTagger {
    model: Option<PerceptronModel>,
}

impl NerTagger {
    /// Creates a tagger with no loaded model.
    pub fn new() -> Self {
        Self { model: None }
    }

    /// Returns `true` if a model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    /// Loads a previously saved model.
    pub fn load_model(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.model = Some(PerceptronModel::load(path.as_ref())?);
        Ok(())
    }

    /// Tags a single sentence.
    ///
    /// `words` is a slice of tokenised words in reading order.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ModelNotLoaded`] if no model has been loaded.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use nerrs::NerTagger;
    ///
    /// let mut tagger = NerTagger::new();
    /// tagger.load_model("ner.model").unwrap();
    /// let tags = tagger.tag(&["علی", "به", "تهران", "رفت", "."]).unwrap();
    /// ```
    pub fn tag(&self, words: &[&str]) -> Result<TaggedSentence> {
        let model = self.model.as_ref().ok_or(Error::ModelNotLoaded)?;
        if words.is_empty() {
            return Ok(vec![]);
        }
        let features = sentence_ner_features(words);
        let tags = model.tag(&features);
        Ok(words.iter().zip(tags).map(|(w, t)| (w.to_string(), t)).collect())
    }

    /// Tags multiple sentences.
    pub fn tag_sents(&self, sents: &[Vec<&str>]) -> Result<Vec<TaggedSentence>> {
        sents.iter().map(|s| self.tag(s)).collect()
    }

    /// Trains a NER model from an IOB-tagged corpus and saves it to `model_path`.
    ///
    /// After training the model is **not** automatically loaded into `self` — use
    /// [`load_model`](Self::load_model) afterwards if needed.
    ///
    /// # Arguments
    ///
    /// * `corpus` — training corpus: a list of sentences, each being a list of
    ///   `(word, ner_tag)` pairs using BIO encoding.
    /// * `model_path` — destination path for the saved model JSON.
    /// * `config` — training hyper-parameters (epochs, etc.).
    pub fn train_and_save(
        corpus: &[Vec<(String, String)>],
        model_path: impl AsRef<Path>,
        config: TrainConfig,
    ) -> Result<()> {
        let label_set = collect_labels(corpus);
        let examples: Vec<(Vec<Vec<String>>, Vec<String>)> = corpus
            .iter()
            .map(|sent| {
                let words: Vec<&str> = sent.iter().map(|(w, _)| w.as_str()).collect();
                let gold: Vec<String> = sent.iter().map(|(_, t)| t.clone()).collect();
                (sentence_ner_features(&words), gold)
            })
            .collect();
        let model = train(&examples, label_set, config);
        model.save(model_path.as_ref())?;
        Ok(())
    }

    /// Trains from corpus and immediately loads the model.
    ///
    /// Equivalent to calling [`train_and_save`](Self::train_and_save) then
    /// [`load_model`](Self::load_model).
    pub fn fit(
        &mut self,
        corpus: &[Vec<(String, String)>],
        model_path: impl AsRef<Path>,
        config: TrainConfig,
    ) -> Result<()> {
        let path = model_path.as_ref();
        Self::train_and_save(corpus, path, config)?;
        self.load_model(path)
    }

    /// Evaluates token-level NER accuracy on a labelled test corpus.
    ///
    /// Returns accuracy in `[0.0, 1.0]`.
    pub fn evaluate(&self, test: &[Vec<(String, String)>]) -> Result<f64> {
        let model = self.model.as_ref().ok_or(Error::ModelNotLoaded)?;
        let (mut correct, mut total) = (0usize, 0usize);
        for sent in test {
            let words: Vec<&str> = sent.iter().map(|(w, _)| w.as_str()).collect();
            let gold: Vec<&str> = sent.iter().map(|(_, t)| t.as_str()).collect();
            let pred = model.tag(&sentence_ner_features(&words));
            for (p, g) in pred.iter().zip(gold.iter()) {
                if p == g {
                    correct += 1;
                }
                total += 1;
            }
        }
        Ok(if total == 0 { 1.0 } else { correct as f64 / total as f64 })
    }
}

impl Default for NerTagger {
    fn default() -> Self {
        Self::new()
    }
}

fn collect_labels(corpus: &[Vec<(String, String)>]) -> Vec<String> {
    let mut seen: std::collections::BTreeSet<String> = DEFAULT_LABELS
        .iter()
        .map(|s| s.to_string())
        .collect();
    for sent in corpus {
        for (_, tag) in sent {
            seen.insert(tag.clone());
        }
    }
    seen.into_iter().collect()
}
