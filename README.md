# rust_ad

`rust_ad` is a small Rust crate that implements reverse-mode automatic differentiation
using a tape-based recording model. It provides an `ADReal` scalar type that records
operations onto a thread-local tape and can propagate adjoints backward to compute
derivatives.

## Features

- Tape-based reverse-mode AD with a bump-allocated tape for efficiency.
- Expression system for building differentiable computations.
- Thread-local tape support with helpers to start/stop recording and reset adjoints.

## Quick example

```rust
use rust_ad::prelude::*;

fn main() {
    Tape::start_recording();
    let x = ADReal::new(2.0);
    let y = ADReal::new(3.0);
    let out: ADReal = (x * y + x).into();
    out.backward().unwrap();

    println!("d(out)/dx = {}", x.adjoint().unwrap()); // 4.0
    println!("d(out)/dy = {}", y.adjoint().unwrap()); // 2.0
}
```

## Development

Run the test suite with:

```sh
cargo test
```
