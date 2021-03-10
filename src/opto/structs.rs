/*
Copyright 2021 BlackRock, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use crate::opto::term::{Objective, Residual};
use crate::quadratics::pwq::PiecewiseQuadratic;
use ndarray::Array1;
use sprs::CsMat;
use std::f64;
use std::fmt;

/// Keeps track of stats collected during ADMM for analysis.
#[derive(Clone)]
pub struct Stats {
    /// The objective values achieved over the course of a run.
    pub objective: Vec<f64>,
    /// The residual values achieved over the course of a run.
    pub residual: Vec<f64>,
    /// The number of ADMM iterations.
    pub iters: usize,
    /// The number of iterations run within the prox operator computation.
    pub prox_iters: usize,
    /// The solve time in milliseconds.
    pub solve_time_ms: usize,
}

impl Stats {
    /// Creates a new `AdmmStats` object.
    pub fn new(capacity: usize) -> Stats {
        let objective = Vec::with_capacity(capacity);
        let residual = Vec::with_capacity(capacity);
        Stats {
            objective,
            residual,
            iters: 0,
            solve_time_ms: 0,
            prox_iters: 0,
        }
    }

    /// Updates the objective, residual and iteration fields of the stat object.
    pub fn update<T: Objective + Residual>(
        &mut self,
        vars: &Variables,
        params: &ProblemData,
        obj_res_calc: &T,
        compute_stats: bool,
    ) {
        if compute_stats {
            self.objective.push(obj_res_calc.objective(vars, params));
            self.residual.push(obj_res_calc.residual(vars, params));
        }
    }
}

pub trait HasVariables {
    fn n_vars(&self) -> usize;
}

pub trait HasConstraints {
    fn n_constrs(&self) -> usize;
}

/// The ADMM algorithm hyperparameters.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Over-relaxation parameter.
    pub alpha: f64,
    /// Algorithm hyperparameter.
    pub rho: Array1<f64>,
    /// Algorithm hyperparameter.
    pub sigma: Array1<f64>,
    /// Maximum allowed iterations.
    pub max_iter: usize,
    /// Threshold used for determining convergence.
    /// The frequency with which to check whether ADMM should terminate.
    pub term_cond_freq: usize,
    /// Minimum amount by which the objective needs to improve to be considered an improvement.
    pub obj_tol: f64,
    /// Maximum allowable residual for an iterate to be considered an improvement.
    pub res_tol: f64,
    /// The maximum allowed non-improvement iterations before termination.
    pub non_improvement_iters: usize,
    /// Whether to compute objectives and residuals on every iteration.
    pub compute_stats: bool,
}

fn all_pos(v: &Array1<f64>) -> bool {
    v.fold(true, |acc, &x| acc && x > 0.)
}

impl Settings {
    /// Initializes a new bundle of ADMM settings.
    pub fn new(
        alpha: f64,
        rho: Array1<f64>,
        sigma: Array1<f64>,
        max_iter: usize,
        term_cond_freq: usize,
        compute_stats: bool,
    ) -> Settings {
        assert!(alpha > 0. && alpha < 2.);
        assert!(all_pos(&rho));
        assert!(all_pos(&sigma));
        Settings {
            alpha,
            rho,
            sigma,
            max_iter,
            term_cond_freq,
            obj_tol: 1e-5,
            res_tol: 3e-4,
            non_improvement_iters: 50,
            compute_stats,
        }
    }

    /// Gets default values for `Settings` given the size of the problem.
    pub fn defaults(n_vars: usize, n_constrs: usize) -> Self {
        Self {
            alpha: 1.5,
            rho: Array1::ones(n_constrs),
            sigma: Array1::ones(n_vars),
            max_iter: 10000,
            obj_tol: 1e-5,
            res_tol: 3e-4,
            non_improvement_iters: 50,
            compute_stats: false,
            term_cond_freq: 10,
        }
    }
}

impl HasVariables for Settings {
    /// The number of variables that this settings bundle will match.
    fn n_vars(&self) -> usize {
        self.sigma.len()
    }
}

impl HasConstraints for Settings {
    /// The number of constraints that this settings bundle will match.
    fn n_constrs(&self) -> usize {
        self.rho.len()
    }
}

/// A collection of matrices and vectors that describe the following quadratic-plus-separable (QPS) problem:
///
/// minimize (1/2) xt P x + qt x + sum(gi(xi))
/// subject to A x = b
#[derive(Debug, Clone)]
pub struct ProblemData {
    /// Quadratic matrix of non-separable part.
    pub p: CsMat<f64>,
    /// Linear coefficient of non-separable part.
    pub q: Array1<f64>,
    /// Stacked equality constraint matrix.
    pub a: CsMat<f64>,
    /// Stacked equality constraint right-hand-side.
    pub b: Array1<f64>,
    /// First column contains separable objective functions, second contains nonlinear constraint functions
    pub g: Array1<PiecewiseQuadratic>,
}

impl ProblemData {
    pub fn new(
        p: CsMat<f64>,
        q: Array1<f64>,
        a: CsMat<f64>,
        b: Array1<f64>,
        g: Array1<PiecewiseQuadratic>,
    ) -> ProblemData {
        let (m, n) = a.shape();
        assert_eq!(p.shape(), (n, n));
        assert_eq!(q.len(), n);
        assert_eq!(b.len(), m);
        assert_eq!(g.len(), n);
        ProblemData { p, q, a, b, g }
    }
}

impl HasVariables for ProblemData {
    /// The number of variables that these params will match.
    fn n_vars(&self) -> usize {
        self.a.shape().1
    }
}

impl HasConstraints for ProblemData {
    /// The number of constraints that these params will match.
    fn n_constrs(&self) -> usize {
        self.a.shape().0
    }
}

/// The variables that will need to be updated on each iteration of the ADMM algorithm.
/// Notes:
/// * `xt` and `zt` are auxiliary variables
/// * We keep the previous values of `x`, `w`, `z`, `y` in order to check for convergence.
/// * The actual problem we are solving here is
///   `minimize (1/2) xtt P xt + q xt + I_A(xt, zt) + g(x) + I_B(z)`
///   `subject to xt = x, zt = z`,
///   where `I_A(xt, zt)` is the indicator function that is zero on the affine set defined by `xt` such that
///   `Axt = zt` and inf otherwise, and `I_B(z)` is zero when `b = z`, and inf otherwise.
/// * `w` and `y` are used when minimizing the augmented Lagrangian of the original problem (and consequently
///   in ADMM's update rules)
#[derive(Debug, Clone)]
pub struct Variables {
    pub xt: Array1<f64>,
    pub zt: Array1<f64>,
    pub x: Array1<f64>,
    pub z: Array1<f64>,
    pub w: Array1<f64>,
    pub y: Array1<f64>,
}

impl Variables {
    /// Initializes a new set of ADMM variables.
    pub fn new(
        xt: Array1<f64>,
        zt: Array1<f64>,
        x: Array1<f64>,
        z: Array1<f64>,
        w: Array1<f64>,
        y: Array1<f64>,
    ) -> Variables {
        assert_eq!(x.len(), xt.len());
        assert_eq!(x.len(), w.len());
        assert_eq!(z.len(), zt.len());
        assert_eq!(z.len(), y.len());
        Variables { xt, zt, x, z, w, y }
    }

    /// Gets new variables given the desired number of variables and constraints.
    pub fn default(lin_constraints: usize, variables: usize) -> Variables {
        let xt = Array1::zeros(variables);
        let zt = Array1::zeros(lin_constraints);
        let x = Array1::zeros(variables);
        let z = Array1::zeros(lin_constraints);
        let w = Array1::zeros(variables);
        let y = Array1::zeros(lin_constraints);

        Variables::new(xt, zt, x, z, w, y)
    }

    /// Creates variables with the appropriate shapes given the shapes of params
    pub fn from_problem_data(params: &ProblemData) -> Variables {
        Variables::default(params.n_constrs(), params.n_vars())
    }
}

impl HasVariables for Variables {
    /// The number of variables that these variables will match.
    fn n_vars(&self) -> usize {
        self.x.len()
    }
}

impl HasConstraints for Variables {
    /// The number of constraints that these variables will match.
    fn n_constrs(&self) -> usize {
        self.z.len()
    }
}

impl fmt::Display for Variables {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "x: {}\nxt: {}\nz: {}\nzt: {}\nw: {}\ny: {}",
            self.x, self.xt, self.z, self.zt, self.w, self.y
        )
    }
}

pub fn validate(variables: &Variables, settings: &Settings, params: &ProblemData) {
    let n = variables.n_vars();
    let m = variables.n_constrs();
    assert_eq!(n, settings.n_vars());
    assert_eq!(n, params.n_vars());
    assert_eq!(m, settings.n_constrs());
    assert_eq!(m, params.n_constrs());
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::array;

    // SETTINGS

    #[test]
    fn test_settings_correct() {
        let alpha = 1.;
        let sigma = array![1.];
        let rho = array![1.];
        let max_iter = 100;
        let term_cond_freq = 100;
        Settings::new(alpha, rho, sigma, max_iter, term_cond_freq, false);
    }

    #[test]
    #[should_panic]
    fn test_settings_alpha_oob() {
        let alpha = 3.;
        let sigma = array![1.];
        let rho = array![1.];
        let max_iter = 100;
        let term_cond_freq = 100;
        Settings::new(alpha, rho, sigma, max_iter, term_cond_freq, false);
    }

    #[test]
    #[should_panic]
    fn test_settings_sigma_neg() {
        let alpha = 1.;
        let sigma = array![-1.];
        let rho = array![1.];
        let max_iter = 100;
        let term_cond_freq = 100;
        Settings::new(alpha, rho, sigma, max_iter, term_cond_freq, false);
    }

    #[test]
    #[should_panic]
    fn test_settings_rho_neg() {
        let alpha = 1.;
        let sigma = array![1.];
        let rho = array![-1.];
        let max_iter = 100;
        let term_cond_freq = 100;
        Settings::new(alpha, rho, sigma, max_iter, term_cond_freq, false);
    }

    // PARAMS

    #[test]
    fn test_params_correct() {
        let p = CsMat::csc_from_dense(array![[1., 0.], [0., 2.]].view(), f64::EPSILON);
        let q = array![1., 2.];
        let a = CsMat::csc_from_dense(array![[1., 2.], [0., 1.], [1., 2.]].view(), f64::EPSILON);
        let b = array![1., 2., 3.];
        let g = array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ];

        ProblemData::new(p, q, a, b, g);
    }

    #[test]
    #[should_panic]
    fn test_params_p_incorrect_dims() {
        let p = CsMat::csc_from_dense(array![[1., 0., 3.], [0., 2., 2.]].view(), f64::EPSILON); // wrong dimension
        let q = array![1., 2.];
        let a = CsMat::csc_from_dense(array![[1., 2.], [0., 1.], [1., 2.]].view(), f64::EPSILON);
        let b = array![1., 2., 3.];
        let g = array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ];

        ProblemData::new(p, q, a, b, g);
    }

    #[test]
    #[should_panic]
    fn test_params_q_incorrect_dims() {
        let p = CsMat::csc_from_dense(array![[1., 0.], [0., 2.]].view(), f64::EPSILON);
        let q = array![1., 2., 3.]; // wrong dimension
        let a = CsMat::csc_from_dense(array![[1., 2.], [0., 1.], [1., 2.]].view(), f64::EPSILON);
        let b = array![1., 2., 3.];
        let g = array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ];

        ProblemData::new(p, q, a, b, g);
    }

    #[test]
    #[should_panic]
    fn test_params_b_incorrect_dims() {
        let p = CsMat::csc_from_dense(array![[1., 0.], [0., 2.]].view(), f64::EPSILON);
        let q = array![1., 2.];
        let a = CsMat::csc_from_dense(array![[1., 2.], [0., 1.], [1., 2.]].view(), f64::EPSILON);
        let b = array![1., 2.]; // wrong dimension
        let g = array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ];

        ProblemData::new(p, q, a, b, g);
    }

    #[test]
    #[should_panic]
    fn test_params_g_incorrect_dims() {
        let p = CsMat::csc_from_dense(array![[1., 0.], [0., 2.]].view(), f64::EPSILON);
        let q = array![1., 2.];
        let a = CsMat::csc_from_dense(array![[1., 2.], [0., 1.], [1., 2.]].view(), f64::EPSILON);
        let b = array![1., 2., 3.];
        let g = array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ];

        ProblemData::new(p, q, a, b, g);
    }

    // VARIABLES

    #[test]
    fn test_variables_defaults_correct() {
        Variables::default(5, 1);
    }

    #[test]
    fn test_variables_manual_correct() {
        let x = array![1.];
        let xt = array![1.];
        let w = array![1.];
        let z = array![1., 2.];
        let zt = array![1., 2.];
        let y = array![1., 2.];
        Variables::new(xt, zt, x, z, w, y);
    }

    #[test]
    #[should_panic]
    fn test_variables_xs_dont_match() {
        let x = array![1.];
        let xt = array![1., 2.]; // must match other x dims
        let w = array![1.];
        let z = array![1., 2.];
        let zt = array![1., 2.];
        let y = array![1., 2.];
        Variables::new(xt, zt, x, z, w, y);
    }

    #[test]
    #[should_panic]
    fn test_variables_x_doesnt_match_w() {
        let x = array![1.];
        let xt = array![1.];
        let w = array![1., 2.]; // must match x dim
        let z = array![1., 2.];
        let zt = array![1., 2.];
        let y = array![1., 2.];
        Variables::new(xt, zt, x, z, w, y);
    }

    #[test]
    #[should_panic]
    fn test_variables_zs_dont_match() {
        let x = array![1.];
        let xt = array![1.];
        let w = array![1.];
        let z = array![1., 2., 3.]; // must match other z dims
        let zt = array![1., 2.];
        let y = array![1., 2.];
        Variables::new(xt, zt, x, z, w, y);
    }

    #[test]
    #[should_panic]
    fn test_variables_z_doesnt_match_y() {
        let x = array![1.];
        let xt = array![1.];
        let w = array![1.];
        let z = array![1., 2.];
        let zt = array![1., 2.];
        let y = array![1., 2., 3.]; // must match z dims
        Variables::new(xt, zt, x, z, w, y);
    }

    // BUNDLE

    #[test]
    fn test_bundle_correct() {
        let p = CsMat::csc_from_dense(array![[1., 0.], [0., 2.]].view(), f64::EPSILON);
        let q = array![1., 2.];
        let a = CsMat::csc_from_dense(array![[1., 2.], [0., 1.], [1., 2.]].view(), f64::EPSILON);
        let b = array![1., 2., 3.];
        let g = array![
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
        ];
        let params = ProblemData::new(p, q, a, b, g);

        let variables = Variables::default(3, 2);

        let alpha = 1.;
        let sigma = array![1., 2.];
        let rho = array![1., 2., 3.];
        let max_iter = 100;
        let term_cond_freq = 100;
        let settings = Settings::new(alpha, rho, sigma, max_iter, term_cond_freq, false);

        // variables: A columns, sigma length, x length = 2
        // constraints: A rows, rho length, z length = 3
        validate(&variables, &settings, &params);
    }
}
