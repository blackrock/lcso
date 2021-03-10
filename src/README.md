# LCSO (Linearly Constrained Separable Optimization)

## Authors
The original algorithms here were developed by [Nicholas Moehle](@moehle), and this project was
authored by [Jack Gindi](@gindij).

## Introduction
This library contains a Rust implementation of the [alternating-direction
method of multipliers (ADMM) algorithm](https://stanford.edu/~boyd/admm.html).

We solve a problem of the form:
  ```
      minimize    (1/2)x'Px + q'x + ∑ g_i(x_i)
         x
      subject to  Ax = b.
  ```

In the above:
* `A` is an `m x n` sparse matrix.
* `b` is an `m`-vector.
* `P` is an `n x n` positive semidefinite sparse matrix.
* `q` is an `n`-vector.
* `x`, the decision variable, is an `n`-vector.
* `g_i` is a piecewise quadratic function for `i = 1,...,n`. For more information on these, 
  see the documentation for the `PiecewiseQuadratic` and `BoundedQuadratic`
  submodules of the `quadratics` module.

An advantage of our method is that very complicated piecewise quadratic functions (e.g., with 
thousands of pieces) can be used while maintaining fast algorithm run time.

## Project organization
The project is organized as follows:
```
src
├── opto            <-- the lcso module
│   ├── admm.rs     <-- ADMM optimization routine
│   ├── mod.rs      <-- lcso namespace module file
│   ├── prox.rs     <-- proximal operator evaluation
│   ├── structs.rs  <-- structures used to hold ADMM state
|   └── term.rs     <-- termination criteria
├── lib.rs          <-- top-level module file
└── quadratics
    ├── bq.rs       <-- bounded quadratic functions
    ├── envelope.rs <-- convex envelope implementation for piecewise
                        quadratic functions
    ├── mod.rs      <-- quadratics namespace module file
    ├── pwq.rs      <-- piecewise quadratic functions
    └── utils.rs    <-- numerical utilities
```

## Getting started
To clone the repo, run `git clone XXX(REPLACE WITH GITHUB URL)`.

## Install
First make sure you have [`Rust>=1.44.1`](https://www.rust-lang.org/tools/install) installed in your environment
The required modules are different depending on whether you are using Linux or MacOS:

### Linux
On Linux, the shared libraries to install depend on your Linux distribution:
* On Debian/Ubuntu, install `libopenblas64-dev` and `libgfortran-{9,10}-dev` (depending on your version of `gcc`).
* On Redhat/Fedora, install `blas-devel` and `libgfortran`.
Once installed, make sure the binaries are on your system path.

### MacOS
Make sure that the following binaries are installed and on your system path:
* `openblas`
* `gcc` (9 or 10)
These can both be installed using [`Homebrew`](http://brew.sh).

Next, you'll need to install [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) (5.7.2). 
You should be able to install it and and its LDL implementation with something like the following snippet:
```bash
# create a lib directory, you could also do the installation anywhere else
mkdir lib
cd lib

# clone the SuiteSparse repository
git clone "https://github.com/DrTimothyAldenDavis/SuiteSparse.git"
cd "SuiteSparse" 

# check out version 5.7.2
git checkout v5.7.2

# install SuiteSparse_config
cd "SuiteSparse_config"
make clean
make install INSTALL="${INSTALL_LOCATION}"

# install SuiteSparse LDL matrix factorization implementation
cd "../LDL"
make clean
make install INSTALL="${INSTALL_LOCATION}"
```
where `$INSTALL_LOCATION` is where you want the binary to be installed. On package-managed Linux, you can omit the 
`INSTALL=...` part of the `make install` commands.

### Compile
To compile the project, run `cargo build`. To compile in release mode, run `cargo build --release`.

To simply check that the code compiles, run `cargo check`.

### Run Tests
To run tests, in the root directory of the project, run `cargo test`.

## Example
To run the example, run `cargo run --example small_example`. To see the example source code, see `examples/small_example.rs`.

## Disclaimer
This code was used to produce results in the paper linked [here](XXX). We will
accept contributions and improvements, but updates and new versions may be
few and far between. At this time, the code is not intended to be relied upon 
in a production setting.

NOTE: Only MacOS and Linux are supported.

