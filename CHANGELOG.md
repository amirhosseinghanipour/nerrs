# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-06-23

### Added

- `extract_entities` — converts a BIO-tagged sentence into `(span, entity_type)` pairs,
  merging consecutive `B-X`/`I-X` tokens into a single space-joined span.
- Optional `hf-hub` feature: enables `NerTagger::load_from_hub(repo_id, filename)` to
  download a model file directly from the Hugging Face Hub at runtime.
- `Error::Hub` variant (behind `hf-hub` feature) wrapping Hub download failures.

## [0.1.0] - 2026-06-22

### Added

- Initial release of `nerrs`: CRF-based Named Entity Recognition for Persian text.
- `NerTagger` — train on annotated corpora, save/load models, and tag token sequences.
- `NerTagger::tag` / `tag_sents` for single and batch sentence tagging.
- `NerTagger::train` with configurable CRF training parameters.
- Feature extraction pipeline (`src/features.rs`) covering prefix/suffix n-grams,
  character-level patterns, and context window features.
- `DEFAULT_LABELS` constant with standard Persian NER BIO label set.
- `Error` / `Result` types with `thiserror`-derived variants.

[0.1.1]: https://github.com/amirhosseinghanipour/nerrs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/amirhosseinghanipour/nerrs/releases/tag/v0.1.0
