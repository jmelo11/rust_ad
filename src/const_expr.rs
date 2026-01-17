//! Constant expression type.

/// A constant expression wrapper for interoperability.
#[derive(Clone, Copy)]
pub struct Const(pub f64);

#[path = "const_expr_impl.rs"]
mod const_expr_impl;
