use rust_ad::adreal::ADReal;
use rust_ad::{differentiable, Differentiable};

#[differentiable(DiffSomeStruct)]
struct SomeStruct {
    a_constant: f64,
    #[diffvar]
    diff_variable: f64,
}

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
    let _: ADReal = result;

    let result = diff_param_left(1.0, ADReal::from(2.0));
    let _: ADReal = result;
}
