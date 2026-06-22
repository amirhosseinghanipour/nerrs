use thiserror::Error;

/// Errors from nerrs.
#[derive(Debug, Error)]
pub enum Error {
    /// No model is loaded.
    #[error("model is not loaded — call load_model() first")]
    ModelNotLoaded,
    /// Error from the underlying CRF layer.
    #[error(transparent)]
    Crf(#[from] crfrs::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
