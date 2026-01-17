use crate::adnumber::{flatten, AddOp, ADNumber, BinExpr, DivOp, Expr, MulOp, SubOp, UnExpr};
use crate::const_expr::Const;
use std::ops::*;

macro_rules! impl_bin_ops_local {
    ($Self:ty) => {
        impl<Rhs> Add<Rhs> for $Self
        where
            Rhs: Expr + Clone,
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Rhs, AddOp>;
            fn add(self, rhs: Rhs) -> Self::Output {
                BinExpr::new(self, rhs)
            }
        }
        impl Add<f64> for $Self
        where
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Const, AddOp>;
            fn add(self, rhs: f64) -> Self::Output {
                BinExpr::new(self, Const(rhs))
            }
        }

        impl<Rhs> Sub<Rhs> for $Self
        where
            Rhs: Expr + Clone,
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Rhs, SubOp>;
            fn sub(self, rhs: Rhs) -> Self::Output {
                BinExpr::new(self, rhs)
            }
        }
        impl Sub<f64> for $Self
        where
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Const, SubOp>;
            fn sub(self, rhs: f64) -> Self::Output {
                BinExpr::new(self, Const(rhs))
            }
        }

        impl<Rhs> Mul<Rhs> for $Self
        where
            Rhs: Expr + Clone,
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Rhs, MulOp>;
            fn mul(self, rhs: Rhs) -> Self::Output {
                BinExpr::new(self, rhs)
            }
        }
        impl Mul<f64> for $Self
        where
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Const, MulOp>;
            fn mul(self, rhs: f64) -> Self::Output {
                BinExpr::new(self, Const(rhs))
            }
        }

        impl<Rhs> Div<Rhs> for $Self
        where
            Rhs: Expr + Clone,
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Rhs, DivOp>;
            fn div(self, rhs: Rhs) -> Self::Output {
                BinExpr::new(self, rhs)
            }
        }
        impl Div<f64> for $Self
        where
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Const, DivOp>;
            fn div(self, rhs: f64) -> Self::Output {
                BinExpr::new(self, Const(rhs))
            }
        }

        impl Neg for $Self
        where
            Self: Expr + Clone,
        {
            type Output = BinExpr<Const, Self, SubOp>;
            fn neg(self) -> Self::Output {
                BinExpr::new(Const(0.0), self)
            }
        }
    };
}

impl_bin_ops_local!(ADNumber);
impl_bin_ops_local!(Const);

macro_rules! impl_bin_ops_expr {
    ($Expr:ident) => {
        impl<L, R, O, Rhs> Add<Rhs> for $Expr<L, R, O>
        where
            Rhs: Expr + Clone,
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Rhs, AddOp>;
            fn add(self, rhs: Rhs) -> Self::Output {
                BinExpr::new(self, rhs)
            }
        }
        impl<L, R, O> Add<f64> for $Expr<L, R, O>
        where
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Const, AddOp>;
            fn add(self, rhs: f64) -> Self::Output {
                BinExpr::new(self, Const(rhs))
            }
        }

        /* Sub ------------------------------------------------------- */
        impl<L, R, O, Rhs> Sub<Rhs> for $Expr<L, R, O>
        where
            Rhs: Expr + Clone,
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Rhs, SubOp>;
            fn sub(self, rhs: Rhs) -> Self::Output {
                BinExpr::new(self, rhs)
            }
        }
        impl<L, R, O> Sub<f64> for $Expr<L, R, O>
        where
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Const, SubOp>;
            fn sub(self, rhs: f64) -> Self::Output {
                BinExpr::new(self, Const(rhs))
            }
        }

        /* Mul ------------------------------------------------------- */
        impl<L, R, O, Rhs> Mul<Rhs> for $Expr<L, R, O>
        where
            Rhs: Expr + Clone,
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Rhs, MulOp>;
            fn mul(self, rhs: Rhs) -> Self::Output {
                BinExpr::new(self, rhs)
            }
        }
        impl<L, R, O> Mul<f64> for $Expr<L, R, O>
        where
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Const, MulOp>;
            fn mul(self, rhs: f64) -> Self::Output {
                BinExpr::new(self, Const(rhs))
            }
        }

        /* Div ------------------------------------------------------- */
        impl<L, R, O, Rhs> Div<Rhs> for $Expr<L, R, O>
        where
            Rhs: Expr + Clone,
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Rhs, DivOp>;
            fn div(self, rhs: Rhs) -> Self::Output {
                BinExpr::new(self, rhs)
            }
        }
        impl<L, R, O> Div<f64> for $Expr<L, R, O>
        where
            Self: Expr + Clone,
        {
            type Output = BinExpr<Self, Const, DivOp>;
            fn div(self, rhs: f64) -> Self::Output {
                BinExpr::new(self, Const(rhs))
            }
        }

        /* Neg ------------------------------------------------------- */
        impl<L, R, O> Neg for $Expr<L, R, O>
        where
            Self: Expr + Clone,
        {
            type Output = BinExpr<Const, Self, SubOp>;
            fn neg(self) -> Self::Output {
                BinExpr::new(Const(0.0), self)
            }
        }
    };
}

impl_bin_ops_expr!(BinExpr);

macro_rules! impl_assign {
    ($Trait:ident, $func:ident, $Op:ident, $sym:tt) => {
        impl<E> $Trait<E> for ADNumber
        where
            E: Expr + Clone,
        {
            fn $func(&mut self, rhs: E) {
                *self = flatten(&(self.clone() $sym rhs));
            }
        }
        impl $Trait<f64> for ADNumber {
            fn $func(&mut self, rhs: f64) {
                *self = flatten(&(self.clone() $sym Const(rhs)));
            }
        }
    };
}

impl_assign!(AddAssign, add_assign, AddOp, +);
impl_assign!(SubAssign, sub_assign, SubOp, -);
impl_assign!(MulAssign, mul_assign, MulOp, *);
impl_assign!(DivAssign, div_assign, DivOp, /);

impl<A, O, Rhs> Add<Rhs> for UnExpr<A, O>
where
    Rhs: Expr + Clone,
    Self: Expr + Clone,
{
    type Output = BinExpr<Self, Rhs, AddOp>;
    fn add(self, rhs: Rhs) -> Self::Output {
        BinExpr::new(self, rhs)
    }
}
impl<A, O> Add<f64> for UnExpr<A, O>
where
    Self: Expr + Clone,
{
    type Output = BinExpr<Self, Const, AddOp>;
    fn add(self, rhs: f64) -> Self::Output {
        BinExpr::new(self, Const(rhs))
    }
}

impl<A, O, Rhs> Sub<Rhs> for UnExpr<A, O>
where
    Rhs: Expr + Clone,
    Self: Expr + Clone,
{
    type Output = BinExpr<Self, Rhs, SubOp>;
    fn sub(self, rhs: Rhs) -> Self::Output {
        BinExpr::new(self, rhs)
    }
}
impl<A, O> Sub<f64> for UnExpr<A, O>
where
    Self: Expr + Clone,
{
    type Output = BinExpr<Self, Const, SubOp>;
    fn sub(self, rhs: f64) -> Self::Output {
        BinExpr::new(self, Const(rhs))
    }
}

impl<A, O, Rhs> Mul<Rhs> for UnExpr<A, O>
where
    Rhs: Expr + Clone,
    Self: Expr + Clone,
{
    type Output = BinExpr<Self, Rhs, MulOp>;
    fn mul(self, rhs: Rhs) -> Self::Output {
        BinExpr::new(self, rhs)
    }
}
impl<A, O> Mul<f64> for UnExpr<A, O>
where
    Self: Expr + Clone,
{
    type Output = BinExpr<Self, Const, MulOp>;
    fn mul(self, rhs: f64) -> Self::Output {
        BinExpr::new(self, Const(rhs))
    }
}

impl<A, O, Rhs> Div<Rhs> for UnExpr<A, O>
where
    Rhs: Expr + Clone,
    Self: Expr + Clone,
{
    type Output = BinExpr<Self, Rhs, DivOp>;
    fn div(self, rhs: Rhs) -> Self::Output {
        BinExpr::new(self, rhs)
    }
}
impl<A, O> Div<f64> for UnExpr<A, O>
where
    Self: Expr + Clone,
{
    type Output = BinExpr<Self, Const, DivOp>;
    fn div(self, rhs: f64) -> Self::Output {
        BinExpr::new(self, Const(rhs))
    }
}

impl<A, O> Neg for UnExpr<A, O>
where
    Self: Expr + Clone,
{
    type Output = BinExpr<Const, Self, SubOp>;
    fn neg(self) -> Self::Output {
        BinExpr::new(Const(0.0), self)
    }
}
