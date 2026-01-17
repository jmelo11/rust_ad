use crate::adnumber::Expr;
use crate::node::TapeNode;

use super::Const;

impl From<f64> for Const {
    /// Converts a `f64` into a constant expression.
    fn from(v: f64) -> Self {
        Const(v)
    }
}
impl From<f32> for Const {
    /// Converts a `f32` into a constant expression.
    fn from(v: f32) -> Self {
        Const(v as f64)
    }
}
impl From<i32> for Const {
    /// Converts an `i32` into a constant expression.
    fn from(v: i32) -> Self {
        Const(v as f64)
    }
}
impl From<u32> for Const {
    /// Converts a `u32` into a constant expression.
    fn from(v: u32) -> Self {
        Const(v as f64)
    }
}
impl From<i64> for Const {
    /// Converts an `i64` into a constant expression.
    fn from(v: i64) -> Self {
        Const(v as f64)
    }
}
impl From<u64> for Const {
    /// Converts a `u64` into a constant expression.
    fn from(v: u64) -> Self {
        Const(v as f64)
    }
}
impl From<Const> for f64 {
    /// Extracts the underlying `f64` from a constant expression.
    fn from(c: Const) -> Self {
        c.0
    }
}

impl Expr for Const {
    /// Returns the scalar value of the constant expression.
    fn inner_value(&self) -> f64 {
        self.0
    }
    /// Constants do not contribute adjoints to the tape.
    fn push_adj(&self, _: &mut TapeNode, _: f64) {}
}
