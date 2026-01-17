//! Automatic differentiation number types and expression building blocks.

use crate::node::TapeNode;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Conversion helpers for numeric types used by this crate.
pub trait ToNumeric<T> {
    /// Creates a new numeric value from the given scalar.
    fn new(v: T) -> Self;
    /// Returns the underlying scalar value.
    fn value(&self) -> T;
    /// Returns the multiplicative identity.
    fn one() -> Self;
    /// Returns the additive identity.
    fn zero() -> Self;
}

/// A differentiable expression that can record its contribution to the tape.
pub trait Expr: Clone {
    /// Returns the scalar value of the expression.
    fn inner_value(&self) -> f64;
    /// Pushes this expression's adjoint contribution into the tape node.
    fn push_adj(&self, parent: &mut TapeNode, adj: f64);
}

/// A scalar value tracked on the automatic differentiation tape.
#[derive(Clone, Copy, Default)]
pub struct ADNumber {
    pub(crate) val: f64,
    pub(crate) node: Option<NonNull<TapeNode>>,
}

/// A constant expression wrapper for interoperability.
pub use crate::const_expr::Const;

/// A binary operation definition for the expression system.
pub trait BinOp {
    /// Evaluates the operator on the input values.
    fn eval(l: f64, r: f64) -> f64;
    /// Computes the derivative with respect to the left operand.
    fn d_left(l: f64, r: f64) -> f64;
    /// Computes the derivative with respect to the right operand.
    fn d_right(l: f64, r: f64) -> f64;
}

/// Binary addition operator.
#[derive(Clone, Copy, Debug)]
pub struct AddOp;
/// Binary subtraction operator.
#[derive(Clone, Copy, Debug)]
pub struct SubOp;
/// Binary multiplication operator.
#[derive(Clone, Copy, Debug)]
pub struct MulOp;
/// Binary division operator.
#[derive(Clone, Copy, Debug)]
pub struct DivOp;
/// Binary power operator.
#[derive(Clone, Copy, Debug)]
pub struct PowOp;
/// Binary maximum operator.
#[derive(Clone, Copy, Debug)]
pub struct MaxOp;
/// Binary minimum operator.
#[derive(Clone, Copy, Debug)]
pub struct MinOp;

/// A binary expression over two child expressions.
#[derive(Clone)]
pub struct BinExpr<L, R, O> {
    pub(crate) l: L,
    pub(crate) r: R,
    pub(crate) val: f64,
    pub(crate) _ph: PhantomData<O>,
}

/// A unary operation definition for the expression system.
pub trait UnOp {
    /// Evaluates the operator on the input value.
    fn eval(x: f64) -> f64;
    /// Computes the derivative with respect to the input.
    fn deriv(x: f64, v: f64) -> f64;
}

/// Unary exponential operator.
#[derive(Clone, Copy, Debug)]
pub struct ExpOp;
/// Unary natural logarithm operator.
#[derive(Clone, Copy, Debug)]
pub struct LogOp;
/// Unary square root operator.
#[derive(Clone, Copy, Debug)]
pub struct SqrtOp;
/// Unary absolute value operator (alias).
#[derive(Clone, Copy, Debug)]
pub struct FabsOp;
/// Unary sine operator.
#[derive(Clone, Copy, Debug)]
pub struct SinOp;
/// Unary cosine operator.
#[derive(Clone, Copy, Debug)]
pub struct CosOp;
/// Unary absolute value operator.
#[derive(Clone, Copy, Debug)]
pub struct AbsOp;

/// A unary expression over a child expression.
#[derive(Clone)]
pub struct UnExpr<A, O> {
    pub(crate) a: A,
    pub(crate) val: f64,
    pub(crate) _ph: PhantomData<O>,
}

/// Convenience methods for common floating-point operations on expressions.
pub trait FloatExt: Expr + Clone + Sized {
    /// Returns `e^x` for the expression.
    fn exp(self) -> UnExpr<Self, ExpOp>;
    /// Returns the natural logarithm of the expression.
    fn ln(self) -> UnExpr<Self, LogOp>;
    /// Returns the sine of the expression.
    fn sin(self) -> UnExpr<Self, SinOp>;
    /// Returns the cosine of the expression.
    fn cos(self) -> UnExpr<Self, CosOp>;
    /// Returns the absolute value of the expression.
    fn abs(self) -> UnExpr<Self, AbsOp>;
    /// Raises the expression to a constant power.
    fn powf(self, p: f64) -> BinExpr<Self, Const, PowOp>;
    /// Returns the square root of the expression.
    fn sqrt(self) -> UnExpr<Self, SqrtOp>;
    /// Raises the expression to the power of another expression.
    fn pow_expr<R: Expr + Clone>(self, p: R) -> BinExpr<Self, R, PowOp>;
    /// Returns the minimum of two expressions.
    fn min<R: Expr + Clone>(self, r: R) -> BinExpr<Self, R, MinOp>;
    /// Returns the maximum of two expressions.
    fn max<R: Expr + Clone>(self, r: R) -> BinExpr<Self, R, MaxOp>;
}

#[path = "adnumber_impl.rs"]
mod adnumber_impl;
pub(crate) use adnumber_impl::flatten;
pub use adnumber_impl::{abs, cos, exp, fabs, log, max, min, pow, sin, sqrt};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    #[test]
    fn compare_and_flatten() {
        let x = ADNumber::new(5.0);
        let y = abs(x - 2.0);
        assert!(y > 2.0); // value-based comparison
        let z: ADNumber = (y + 1.0).into();
        assert_eq!(z.value(), 4.0);
    }

    #[test]
    fn backprop_basic() {
        Tape::start_recording();
        let a = ADNumber::new(3.0);
        let b = ADNumber::new(4.0);
        let expr = (a * b).sin();
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn test_late_tape_recording() {
        let mut a = ADNumber::new(3.0);
        println!("a: {:?}", a);
        Tape::start_recording(); // start recording
        a.put_on_tape();
        let expr = a * a;
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(a.adjoint().unwrap(), 6.0);
    }

    #[test]
    fn backprop_with_const() {
        Tape::start_recording();
        let a = ADNumber::new(3.0);
        let b = Const(4.0);
        let expr = (a * b).sin();
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn tape_reset() {
        Tape::start_recording();
        let a = ADNumber::new(3.0);
        let b = ADNumber::new(4.0);
        let expr = (a * b).sin();
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(out.adjoint().unwrap(), 1.0);

        Tape::reset_adjoints(); // reset adjoints
        assert_eq!(out.adjoint().unwrap(), 0.0); // should be zero now
    }

    #[test]
    fn tape_propagate_mark() {
        Tape::start_recording();
        let a = ADNumber::new(3.0);
        let b = ADNumber::new(4.0);
        let expr = (a * b).sin();
        let out: ADNumber = expr.into();
        out.backward_to_mark().unwrap(); // propagate to the current mark
        assert_eq!(out.adjoint().unwrap(), 1.0); // should be 1.0
    }

    #[test]
    fn tape_backward_to_mark() {
        Tape::start_recording();
        let a = ADNumber::new(3.0);
        let b = ADNumber::new(4.0);
        let expr = (a * b).sin();
        let out: ADNumber = expr.into();
        out.backward_to_mark().unwrap(); // propagate to the current mark
        assert_eq!(out.adjoint().unwrap(), 1.0); // should be 1.0

        out.backward().unwrap(); // propagate from mark to start
        assert_eq!(out.adjoint().unwrap(), 1.0); // should still be 1.0
    }

    #[test]
    fn check_exp_derivate() {
        Tape::start_recording();
        let x = ADNumber::new(2.0);
        let expr = exp(x);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), f64::exp(2.0)); // derivative of exp(x) wrt x
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn check_log_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(2.0);
        let expr = log(x);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 1.0 / 2.0); // derivative of log(x) wrt x
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }
    #[test]
    fn check_sqrt_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(4.0);
        let expr = sqrt(x);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 0.5 / 2.0); // derivative of sqrt(x) wrt x
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }
    #[test]
    fn check_sin_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(0.0);
        let expr = sin(x);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 1.0); // derivative of sin(x) wrt x
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }
    #[test]
    fn check_cos_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(0.0);
        let expr = cos(x);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 0.0); // derivative of cos(x) wrt x
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }
    #[test]
    fn check_abs_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(-3.0);
        let expr = abs(x);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), -1.0); // derivative of abs(x) wrt x
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }
    #[test]
    fn check_pow_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(2.0);
        let expr = pow(x, Const(3.0));
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 3.0 * 2.0f64.powf(2.0)); // derivative of x^3 wrt x at x=2
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }
    #[test]
    fn check_max_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(2.0);
        let y = ADNumber::new(3.0);
        let expr = max(x, y);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 0.0); // derivative wrt x
        assert_eq!(y.adjoint().unwrap(), 1.0); // derivative wrt y
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }
    #[test]
    fn check_min_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(2.0);
        let y = ADNumber::new(3.0);
        let expr = min(x, y);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 1.0); // derivative wrt x
        assert_eq!(y.adjoint().unwrap(), 0.0); // derivative wrt y
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }
    #[test]
    fn check_flattening() {
        Tape::start_recording();
        let x = ADNumber::new(5.0);
        let y = ADNumber::new(3.0);
        let expr = (x + y) * 2.0;
        let out: ADNumber = expr.into();
        assert_eq!(out.value(), 16.0); // (5 + 3) * 2 = 16
        out.backward().unwrap();
        assert_eq!(out.adjoint().unwrap(), 1.0); // should be 1.0 after propagation
    }

    #[test]
    fn check_add_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(2.0);
        let y = ADNumber::new(3.0);
        let expr = x + y;
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 1.0);
        assert_eq!(y.adjoint().unwrap(), 1.0);
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn check_sub_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(5.0);
        let y = ADNumber::new(2.0);
        let expr = x - y;
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 1.0);
        assert_eq!(y.adjoint().unwrap(), -1.0);
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn check_mul_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(4.0);
        let y = ADNumber::new(2.0);
        let expr = x * y;
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 2.0);
        assert_eq!(y.adjoint().unwrap(), 4.0);
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn check_div_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(6.0);
        let y = ADNumber::new(3.0);
        let expr = x / y;
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert!((x.adjoint().unwrap() - (1.0 / 3.0)).abs() < 1e-12);
        assert!((y.adjoint().unwrap() + (6.0 / 9.0)).abs() < 1e-12);
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn check_fabs_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(-2.0);
        let expr = fabs(x);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), -1.0);
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn check_pow_variable_exponent() {
        Tape::start_recording();
        let x = ADNumber::new(2.0);
        let y = ADNumber::new(3.0);
        let expr = pow(x, y);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 3.0 * 2.0f64.powf(2.0));
        assert!((y.adjoint().unwrap() - (8.0 * 2.0f64.ln())).abs() < 1e-12);
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn check_max_derivative_x_greater() {
        Tape::start_recording();
        let x = ADNumber::new(5.0);
        let y = ADNumber::new(3.0);
        let expr = max(x, y);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 1.0);
        assert_eq!(y.adjoint().unwrap(), 0.0);
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn check_min_derivative_y_less() {
        Tape::start_recording();
        let x = ADNumber::new(5.0);
        let y = ADNumber::new(3.0);
        let expr = min(x, y);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 0.0);
        assert_eq!(y.adjoint().unwrap(), 1.0);
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn check_abs_positive_derivative() {
        Tape::start_recording();
        let x = ADNumber::new(3.0);
        let expr = abs(x);
        let out: ADNumber = expr.into();
        out.backward().unwrap();
        assert_eq!(x.adjoint().unwrap(), 1.0);
        assert_eq!(out.adjoint().unwrap(), 1.0);
    }

    #[test]
    fn test_reassigning() {
        Tape::start_recording();

        let a0 = ADNumber::new(5.0); // keep an immutable handle to the leaf
        let b = ADNumber::new(3.0);
        let mut a = a0; // mutable alias
        a *= b; // 5 * 3 = 15
        let c = a;
        assert_eq!(c.value(), 15.0); // value is correct

        c.backward().unwrap();

        assert_eq!(a0.adjoint().unwrap(), 3.0); // ∂c/∂a0  = b = 3
        assert_eq!(b.adjoint().unwrap(), 5.0); // ∂c/∂b   = a0 = 5
        assert_eq!(c.adjoint().unwrap(), 1.0); // seed stays 1
    }
}
