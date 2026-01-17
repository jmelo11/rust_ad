use crate::const_expr::Const;
use crate::errors::{ADError, Result};
use crate::node::TapeNode;
use crate::tape::Tape;
use crate::tape::TAPE;
use core::fmt;
use std::cell::Cell;
use std::cmp::Ordering;
use std::ptr::NonNull;

use super::{
    AbsOp, AddOp, ADNumber, BinExpr, BinOp, CosOp, DivOp, ExpOp, Expr, FabsOp, FloatExt, LogOp,
    MaxOp, MinOp, MulOp, PowOp, SinOp, SqrtOp, SubOp, ToNumeric, UnExpr, UnOp,
};

impl ToNumeric<f64> for f64 {
    #[inline]
    fn new(v: f64) -> Self {
        v
    }
    #[inline]
    fn value(&self) -> f64 {
        *self
    }

    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn zero() -> Self {
        0.0
    }
}

impl ToNumeric<f64> for ADNumber {
    #[inline]
    fn new(v: f64) -> Self {
        let node = ADNumber::TAPE_PTR.with(|t| {
            if !t.get().is_null() {
                unsafe { t.get().as_mut().unwrap().new_leaf() }
            } else {
                panic!("ADNumber::new called without a tape set");
            }
        });
        Self {
            val: v,
            node: Some(node),
        }
    }
    #[inline]
    fn value(&self) -> f64 {
        self.val
    }
    #[inline]
    fn one() -> Self {
        ADNumber::new(1.0)
    }
    #[inline]
    fn zero() -> Self {
        ADNumber::new(0.0)
    }
}

impl fmt::Debug for ADNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ADNumber({}, Node: {:?})", self.val, self.node)
    }
}

impl fmt::Display for ADNumber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ADNumber({})", self.val)
    }
}

unsafe impl Sync for ADNumber {}
unsafe impl Send for ADNumber {}

impl ADNumber {
    thread_local! {
        static TAPE_PTR: Cell<*mut Tape> = Cell::new(unsafe {
            &mut *TAPE.with(|t| t.as_ptr()) as *mut Tape
        });
    }

    /// Sets the active tape for the current thread.
    pub fn set_tape(t: &mut Tape) {
        Self::TAPE_PTR.with(|c| c.set(t));
    }

    /// Returns whether a tape pointer has been configured for this thread.
    pub fn has_tape() -> bool {
        Self::TAPE_PTR.with(|c| !c.get().is_null())
    }

    /// Returns the raw pointer address of the current tape.
    pub fn tape_addr() -> *mut Tape {
        Self::TAPE_PTR.with(|c| c.get())
    }

    /// Creates a new scalar, recording it on the tape if recording is active.
    pub fn new(v: f64) -> Self {
        Self::TAPE_PTR.with(|t| {
            let ptr = t.get();
            if !ptr.is_null() {
                let active = unsafe { (*ptr).active };
                if active {
                    let node = unsafe { (*ptr).new_leaf() };
                    ADNumber {
                        val: v,
                        node: Some(node),
                    }
                } else {
                    ADNumber { val: v, node: None }
                }
            } else {
                ADNumber {
                    val: v,
                    node: None, // no tape active
                }
            }
        })
    }

    /// Returns the stored scalar value.
    pub fn value(&self) -> f64 {
        self.val
    }

    /// Returns the adjoint for this value if it is on the tape.
    pub fn adjoint(&self) -> Result<f64> {
        self.node
            .map(|p| unsafe { p.as_ref().adj })
            .ok_or(ADError::NodeNotIndexedInTapeErr)
    }

    /// Runs a full backward pass from this node to the start of the tape.
    pub fn backward(&self) -> Result<()> {
        let root = self.node.ok_or(ADError::NodeNotIndexedInTapeErr)?;

        Self::TAPE_PTR.with(|t| {
            let tape = unsafe { &mut *t.get() };
            tape.mut_node(root).unwrap().adj = 1.0;
            tape.propagate_from(root)
        })?;
        Ok(())
    }

    /// Runs a backward pass from the current mark down to the start.
    pub fn backward_mark_to_start(&self) -> Result<()> {
        let root: NonNull<TapeNode> = self.node.ok_or(ADError::NodeNotIndexedInTapeErr)?;

        Self::TAPE_PTR.with(|t| {
            let tape = unsafe { &mut *t.get() };
            tape.mut_node(root).unwrap().adj = 1.0;
            tape.propagate_mark_to_start()
        });
        Ok(())
    }

    /// Runs a backward pass from the end of the tape down to the current mark.
    pub fn backward_to_mark(&self) -> Result<()> {
        let root: NonNull<TapeNode> = self.node.ok_or(ADError::NodeNotIndexedInTapeErr)?;
        Self::TAPE_PTR.with(|t| {
            let tape = unsafe { &mut *t.get() };
            tape.mut_node(root).unwrap().adj = 1.0;
            tape.propagate_to_mark()
        });
        Ok(())
    }

    /// Attaches this value to the current tape if it is not already recorded.
    pub fn put_on_tape(&mut self) {
        if self.node.is_some() {
            return; // already on a tape
        }

        Self::TAPE_PTR.with(|t| {
            let ptr = t.get();
            if !ptr.is_null() {
                let node = unsafe { (*ptr).new_leaf() };
                self.node = Some(node);
            } else {
                panic!("ADNumber::put_on_tape called without a tape set");
            }
        });
    }
}

impl Expr for ADNumber {
    /// Returns the scalar value for use in tape recording.
    fn inner_value(&self) -> f64 {
        self.val
    }

    /// Pushes this value into the parent tape node with the given derivative.
    fn push_adj(&self, parent: &mut TapeNode, deriv: f64) {
        if let Some(p) = self.node {
            parent.childs.push(p);
            parent.derivs.push(deriv);
        }
    }
}

/// Records an expression into the tape, returning the resulting `ADNumber`.
pub(crate) fn flatten<E: Expr + Clone>(e: &E) -> ADNumber {
    let mut node = TapeNode::default();
    e.push_adj(&mut node, 1.0);

    let ptr_opt = ADNumber::TAPE_PTR.with(|t| {
        let tape = unsafe { &mut *t.get() };
        tape.record(node)
    });

    ADNumber {
        val: e.inner_value(),
        node: ptr_opt,
    }
}

impl PartialEq for ADNumber {
    fn eq(&self, o: &Self) -> bool {
        self.val == o.val
    }
}
impl PartialOrd for ADNumber {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&o.val)
    }
}

impl BinOp for AddOp {
    /// Evaluates the operator on the input values.
    fn eval(l: f64, r: f64) -> f64 {
        l + r
    }
    /// Returns the derivative with respect to the left operand.
    fn d_left(_: f64, _: f64) -> f64 {
        1.0
    }
    /// Returns the derivative with respect to the right operand.
    fn d_right(_: f64, _: f64) -> f64 {
        1.0
    }
}
impl BinOp for SubOp {
    /// Evaluates the operator on the input values.
    fn eval(l: f64, r: f64) -> f64 {
        l - r
    }
    /// Returns the derivative with respect to the left operand.
    fn d_left(_: f64, _: f64) -> f64 {
        1.0
    }
    /// Returns the derivative with respect to the right operand.
    fn d_right(_: f64, _: f64) -> f64 {
        -1.0
    }
}
impl BinOp for MulOp {
    /// Evaluates the operator on the input values.
    fn eval(l: f64, r: f64) -> f64 {
        l * r
    }
    /// Returns the derivative with respect to the left operand.
    fn d_left(_: f64, r: f64) -> f64 {
        r
    }
    /// Returns the derivative with respect to the right operand.
    fn d_right(l: f64, _: f64) -> f64 {
        l
    }
}
impl BinOp for DivOp {
    /// Evaluates the operator on the input values.
    fn eval(l: f64, r: f64) -> f64 {
        l / r
    }
    /// Returns the derivative with respect to the left operand.
    fn d_left(_: f64, r: f64) -> f64 {
        1.0 / r
    }
    /// Returns the derivative with respect to the right operand.
    fn d_right(l: f64, r: f64) -> f64 {
        -l / (r * r)
    }
}
impl BinOp for PowOp {
    /// Evaluates the operator on the input values.
    fn eval(l: f64, r: f64) -> f64 {
        l.powf(r)
    }
    /// Returns the derivative with respect to the left operand.
    fn d_left(l: f64, r: f64) -> f64 {
        r * l.powf(r - 1.0)
    }
    /// Returns the derivative with respect to the right operand.
    fn d_right(l: f64, r: f64) -> f64 {
        l.powf(r) * l.ln()
    }
}
impl BinOp for MaxOp {
    /// Evaluates the operator on the input values.
    fn eval(l: f64, r: f64) -> f64 {
        l.max(r)
    }
    /// Returns the derivative with respect to the left operand.
    fn d_left(l: f64, r: f64) -> f64 {
        if l > r { 1.0 } else { 0.0 }
    }
    /// Returns the derivative with respect to the right operand.
    fn d_right(l: f64, r: f64) -> f64 {
        if r > l { 1.0 } else { 0.0 }
    }
}
impl BinOp for MinOp {
    /// Evaluates the operator on the input values.
    fn eval(l: f64, r: f64) -> f64 {
        l.min(r)
    }
    /// Returns the derivative with respect to the left operand.
    fn d_left(l: f64, r: f64) -> f64 {
        if l < r { 1.0 } else { 0.0 }
    }
    /// Returns the derivative with respect to the right operand.
    fn d_right(l: f64, r: f64) -> f64 {
        if r < l { 1.0 } else { 0.0 }
    }
}

impl<L: Expr, R: Expr, O: BinOp> BinExpr<L, R, O> {
    /// Constructs a new binary expression and caches its value.
    pub(crate) fn new(l: L, r: R) -> Self {
        let val = O::eval(l.inner_value(), r.inner_value());
        Self {
            l,
            r,
            val,
            _ph: std::marker::PhantomData,
        }
    }
}
impl<L: Expr, R: Expr, O: BinOp + Clone> Expr for BinExpr<L, R, O> {
    /// Returns the cached scalar value of the expression.
    fn inner_value(&self) -> f64 {
        self.val
    }
    /// Pushes adjoint contributions for the left and right child expressions.
    fn push_adj(&self, parent: &mut TapeNode, adj: f64) {
        self.l.push_adj(
            parent,
            adj * O::d_left(self.l.inner_value(), self.r.inner_value()),
        );
        self.r.push_adj(
            parent,
            adj * O::d_right(self.l.inner_value(), self.r.inner_value()),
        );
    }
}

impl UnOp for ExpOp {
    /// Evaluates the unary operator.
    fn eval(x: f64) -> f64 {
        f64::exp(x)
    }
    /// Returns the derivative of the unary operator.
    fn deriv(_x: f64, v: f64) -> f64 {
        v
    }
}
impl UnOp for LogOp {
    /// Evaluates the unary operator.
    fn eval(x: f64) -> f64 {
        f64::ln(x)
    }
    /// Returns the derivative of the unary operator.
    fn deriv(x: f64, _v: f64) -> f64 {
        1.0 / x
    }
}
impl UnOp for SqrtOp {
    /// Evaluates the unary operator.
    fn eval(x: f64) -> f64 {
        f64::sqrt(x)
    }
    /// Returns the derivative of the unary operator.
    fn deriv(_x: f64, v: f64) -> f64 {
        0.5 / v
    }
}
impl UnOp for FabsOp {
    /// Evaluates the unary operator.
    fn eval(x: f64) -> f64 {
        f64::abs(x)
    }
    /// Returns the derivative of the unary operator.
    fn deriv(x: f64, _v: f64) -> f64 {
        if x >= 0.0 { 1.0 } else { -1.0 }
    }
}
impl UnOp for SinOp {
    /// Evaluates the unary operator.
    fn eval(x: f64) -> f64 {
        f64::sin(x)
    }
    /// Returns the derivative of the unary operator.
    fn deriv(x: f64, _v: f64) -> f64 {
        f64::cos(x)
    }
}
impl UnOp for CosOp {
    /// Evaluates the unary operator.
    fn eval(x: f64) -> f64 {
        f64::cos(x)
    }
    /// Returns the derivative of the unary operator.
    fn deriv(x: f64, _v: f64) -> f64 {
        -f64::sin(x)
    }
}
impl UnOp for AbsOp {
    /// Evaluates the unary operator.
    fn eval(x: f64) -> f64 {
        f64::abs(x)
    }
    /// Returns the derivative of the unary operator.
    fn deriv(x: f64, _v: f64) -> f64 {
        if x >= 0.0 { 1.0 } else { -1.0 }
    }
}

impl<A: Expr, O: UnOp> UnExpr<A, O> {
    /// Constructs a new unary expression and caches its value.
    pub(crate) fn new(a: A) -> Self {
        let val = O::eval(a.inner_value());
        Self {
            a,
            val,
            _ph: std::marker::PhantomData,
        }
    }
}
impl<A: Expr, O: UnOp + Clone> Expr for UnExpr<A, O> {
    /// Returns the cached scalar value of the expression.
    fn inner_value(&self) -> f64 {
        self.val
    }
    /// Pushes adjoint contributions for the child expression.
    fn push_adj(&self, parent: &mut TapeNode, adj: f64) {
        self.a
            .push_adj(parent, adj * O::deriv(self.a.inner_value(), self.val));
    }
}

impl<A, O> PartialEq for UnExpr<A, O>
where
    A: Expr,
    O: UnOp + Clone,
{
    fn eq(&self, rhs: &Self) -> bool {
        self.inner_value() == rhs.inner_value()
    }
}
impl<A, O> PartialOrd for UnExpr<A, O>
where
    A: Expr,
    O: UnOp + Clone,
{
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        self.inner_value().partial_cmp(&rhs.inner_value())
    }
}
impl<A, O> PartialEq<f64> for UnExpr<A, O>
where
    A: Expr,
    O: UnOp + Clone,
{
    fn eq(&self, rhs: &f64) -> bool {
        self.inner_value() == *rhs
    }
}
impl<A, O> PartialOrd<f64> for UnExpr<A, O>
where
    A: Expr,
    O: UnOp + Clone,
{
    fn partial_cmp(&self, rhs: &f64) -> Option<Ordering> {
        self.inner_value().partial_cmp(rhs)
    }
}

impl<L, R, O> PartialEq for BinExpr<L, R, O>
where
    L: Expr,
    R: Expr,
    O: BinOp + Clone,
{
    fn eq(&self, rhs: &Self) -> bool {
        self.inner_value() == rhs.inner_value()
    }
}
impl<L, R, O> PartialOrd for BinExpr<L, R, O>
where
    L: Expr,
    R: Expr,
    O: BinOp + Clone,
{
    fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
        self.inner_value().partial_cmp(&rhs.inner_value())
    }
}
impl<L, R, O> PartialEq<f64> for BinExpr<L, R, O>
where
    L: Expr,
    R: Expr,
    O: BinOp + Clone,
{
    fn eq(&self, rhs: &f64) -> bool {
        self.inner_value() == *rhs
    }
}
impl<L, R, O> PartialOrd<f64> for BinExpr<L, R, O>
where
    L: Expr,
    R: Expr,
    O: BinOp + Clone,
{
    fn partial_cmp(&self, rhs: &f64) -> Option<Ordering> {
        self.inner_value().partial_cmp(rhs)
    }
}

/// Returns the exponential of an expression.
pub fn exp<A: Expr + Clone>(a: A) -> UnExpr<A, ExpOp> {
    UnExpr::new(a)
}
/// Returns the natural logarithm of an expression.
pub fn log<A: Expr + Clone>(a: A) -> UnExpr<A, LogOp> {
    UnExpr::new(a)
}
/// Returns the square root of an expression.
pub fn sqrt<A: Expr + Clone>(a: A) -> UnExpr<A, SqrtOp> {
    UnExpr::new(a)
}
/// Returns the absolute value of an expression.
pub fn fabs<A: Expr + Clone>(a: A) -> UnExpr<A, FabsOp> {
    UnExpr::new(a)
}
/// Returns the sine of an expression.
pub fn sin<A: Expr + Clone>(a: A) -> UnExpr<A, SinOp> {
    UnExpr::new(a)
}
/// Returns the cosine of an expression.
pub fn cos<A: Expr + Clone>(a: A) -> UnExpr<A, CosOp> {
    UnExpr::new(a)
}
/// Returns the absolute value of an expression.
pub fn abs<A: Expr + Clone>(a: A) -> UnExpr<A, AbsOp> {
    UnExpr::new(a)
}

/// Raises one expression to the power of another.
pub fn pow<L: Expr + Clone, R: Expr + Clone>(l: L, r: R) -> BinExpr<L, R, PowOp> {
    BinExpr::new(l, r)
}
/// Returns the maximum of two expressions.
pub fn max<L: Expr + Clone, R: Expr + Clone>(l: L, r: R) -> BinExpr<L, R, MaxOp> {
    BinExpr::new(l, r)
}
/// Returns the minimum of two expressions.
pub fn min<L: Expr + Clone, R: Expr + Clone>(l: L, r: R) -> BinExpr<L, R, MinOp> {
    BinExpr::new(l, r)
}

impl<L, R, O> From<BinExpr<L, R, O>> for ADNumber
where
    L: Expr + Clone,
    R: Expr + Clone,
    O: BinOp + Clone,
{
    /// Flattens a binary expression into an `ADNumber` and records it.
    fn from(e: BinExpr<L, R, O>) -> Self {
        flatten(&e)
    }
}
impl<A, O> From<UnExpr<A, O>> for ADNumber
where
    A: Expr + Clone,
    O: UnOp + Clone,
{
    /// Flattens a unary expression into an `ADNumber` and records it.
    fn from(e: UnExpr<A, O>) -> Self {
        flatten(&e)
    }
}
impl From<f64> for ADNumber {
    /// Converts a `f64` into an `ADNumber`, recording if the tape is active.
    fn from(v: f64) -> Self {
        ADNumber::new(v)
    }
}
impl From<f32> for ADNumber {
    /// Converts a `f32` into an `ADNumber`, recording if the tape is active.
    fn from(v: f32) -> Self {
        ADNumber::new(v as f64)
    }
}
impl From<i32> for ADNumber {
    /// Converts an `i32` into an `ADNumber`, recording if the tape is active.
    fn from(v: i32) -> Self {
        ADNumber::new(v as f64)
    }
}

impl<T: Expr + Clone> FloatExt for T {
    fn exp(self) -> UnExpr<Self, ExpOp> {
        UnExpr::new(self)
    }

    fn ln(self) -> UnExpr<Self, LogOp> {
        UnExpr::new(self)
    }

    fn sin(self) -> UnExpr<Self, SinOp> {
        UnExpr::new(self)
    }

    fn cos(self) -> UnExpr<Self, CosOp> {
        UnExpr::new(self)
    }

    fn abs(self) -> UnExpr<Self, AbsOp> {
        UnExpr::new(self)
    }

    fn powf(self, p: f64) -> BinExpr<Self, Const, PowOp> {
        BinExpr::new(self, Const(p))
    }

    fn sqrt(self) -> UnExpr<Self, SqrtOp> {
        UnExpr::new(self)
    }

    fn pow_expr<R: Expr + Clone>(self, p: R) -> BinExpr<Self, R, PowOp> {
        BinExpr::new(self, p)
    }

    fn min<R: Expr + Clone>(self, r: R) -> BinExpr<Self, R, MinOp> {
        BinExpr::new(self, r)
    }

    fn max<R: Expr + Clone>(self, r: R) -> BinExpr<Self, R, MaxOp> {
        BinExpr::new(self, r)
    }
}

impl PartialEq<f64> for ADNumber {
    fn eq(&self, rhs: &f64) -> bool {
        self.value() == *rhs
    }
}

impl PartialOrd<f64> for ADNumber {
    fn partial_cmp(&self, rhs: &f64) -> Option<Ordering> {
        self.value().partial_cmp(rhs)
    }
}
