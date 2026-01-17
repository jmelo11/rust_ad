use thiserror::Error;

#[derive(Error, Debug)]
pub enum ADError {
    #[error("Node error: {0}")]
    NodeError(String),
    #[error("Tape error: {0}")]
    TapeError(String),
    #[error("AD Number error: {0}")]
    ADNumberError(String),
    #[error("Node not indexed in tape")]
    NodeNotIndexedInTapeErr,
}

pub type Result<T> = std::result::Result<T, ADError>;
