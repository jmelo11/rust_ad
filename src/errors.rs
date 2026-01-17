//! Error types for the automatic differentiation crate.

use thiserror::Error;

#[derive(Error, Debug)]
/// Error variants for tape and AD operations.
pub enum ADError {
    #[error("Node error: {0}")]
    /// A generic node error.
    NodeError(String),
    #[error("Tape error: {0}")]
    /// A generic tape error.
    TapeError(String),
    #[error("AD Number error: {0}")]
    /// A generic AD number error.
    ADNumberError(String),
    #[error("Node not indexed in tape")]
    /// Attempted to access a node that is not recorded on the tape.
    NodeNotIndexedInTapeErr,
}

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, ADError>;
