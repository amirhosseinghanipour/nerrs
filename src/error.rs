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
    /// Hugging Face Hub download failed.
    #[cfg(feature = "hf-hub")]
    #[error("Hugging Face Hub error: {0}")]
    Hub(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
