use rust_ad::adreal::{ADReal, IsReal};
use rust_ad::{Differentiable, differentiable};

#[differentiable(DiffSomeStruct)]
struct SomeStruct {
    a_constant: f64,
    #[diffvar]
    diff_variable: f64,
}

// impl SomeStruct {
//     #[differentiable(diff_some_inmmutable_method)]
//     fn some_inmmutable_method(&self, x: f64) -> f64 {
//         x + 1.0
//     }

//     #[differentiable(diff_some_mutable_method)]
//     fn some_mutable_method(&mut self, x: f64) -> f64 {
//         self.diff_variable += x;
//         self.diff_variable
//     }

//     #[differentiable(diff_some_mutable_method_no_return)]
//     fn some_mutable_method_no_return(&mut self, x: f64) {
//         self.diff_variable += x + 1.0;
//     }
// }

#[differentiable(diff_example)]
fn example(a: f64, #[diffvar] x: f64) -> f64 {
    x + a
}

#[differentiable(diff_literal_left)]
fn literal_left(#[diffvar] x: f64) -> f64 {
    2.0 + x
}

#[differentiable(diff_param_left)]
fn param_left(a: f64, #[diffvar] x: f64) -> f64 {
    a + x
}

#[differentiable(diff_param_more_complex)]
fn param_left_more_complex(a: f64, #[diffvar] x: f64) -> f64 {
    let tmp = (a + x) / (1.0 + x);
    tmp * x * x
}

fn assert_differentiable<T: Differentiable>() {}

#[test]
fn macro_generates_struct_companion() {
    assert_differentiable::<DiffSomeStruct>();

    let diff = DiffSomeStruct {
        a_constant: 1.0,
        diff_variable: ADReal::from(2.0),
    };

    let _: ADReal = diff.diff_variable;
}

#[test]
fn macro_generates_function_companion() {
    let result = diff_example(1.0, ADReal::from(2.0));
    let _: ADReal = result;
}

#[test]
fn macro_wraps_literals_and_params() {
    let result = diff_literal_left(ADReal::from(2.0));
    println!("{:}", result.value());

    let result = diff_param_left(1.0, ADReal::from(2.0));
    println!("{:}", result.value());
}

#[test]
fn macro_more_complex_func() {
    let result = diff_param_more_complex(1.0, ADReal::new(1.0));
    println!("{:}", result.value());
}
