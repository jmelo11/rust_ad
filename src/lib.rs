//! Reverse-mode automatic differentiation primitives.
//!
//! This crate provides a tape-based implementation for recording operations and
//! propagating adjoints to compute gradients.

pub mod adreal;
pub mod errors;
pub mod node;
pub mod prelude;
pub mod tape;
