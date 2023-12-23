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

use crate::opto::prox::{Prox, UnconstrainedProxCache};
use crate::opto::structs::{validate, HasVariables, ProblemData, Settings, Stats, Variables};
use crate::opto::term::{Objective, Residual, TermCache};
use crate::quadratics::envelope::envelope;
use crate::quadratics::pwq::PiecewiseQuadratic;
use chrono::Utc;
use ndarray::{stack, Array1, Axis};
use sprs::{bmat, CsMat};
use sprs_suitesparse_ldl::LdlNumeric;
use std::f64;

/// Factorize the coefficient matrix of that KKT system representing the optimality equations
/// that will be used (and reused) the constrained least squares problem:
///
/// `minimize_z (1/2)x^TPx + q^Tx + sum_i(g_i(x_i))`
/// `subject to Ax = b`
///
/// We can formulate this equivalently as
///
/// `minimize (1/2)xt^TPxt + q^Txt + I_A(xt, zt) + sum_i(g_i(x_i)) + I_b(z) (= f(x, xt, z, zt))`
/// `subject to xt = x, zt = z`
///
/// Where `I_A(xt, zt) = 0` if `Axt = zt` and infinity otherwise, and `I_b = 0` if `z = b` and infinity
/// otherwise. The augmented Lagrangian for this problem is
///
/// `L_{S,R}(x, xt, z, zt, w, y) = f(x, xt, z, zt) + (1/2)norm(zt - (z - R^{-1}y), R)^2 + (1/2)norm(xt - (x - S^{-1}), S)^2`,
///
/// where parameters `S` and `R` are positive definite diagonal matrices, `norm(u, M) = sqrt(u'Mu)`, and where
/// `w` and `y` are dual variables.
/// Solving for the optimality conditions, can find optimal `xt` and `zt` by solving the linear system
///
/// [ P + S   A^T    ] [ xt ]   =  [ Sx - w - q  ]
/// [  A      -R^{-1}] [ v  ]      [ z - R^{-1}y ],
///
/// where `zt` can then be recovered as `zt = z + R^{-1}(v - y)`.
/// In this function, we factor the coefficient matrix on the left hand side so that we repeatedly
/// solve the linear system efficiently.
///
/// # Panics:
/// * If the LDL factorization fails. The problem has been constructed to make sure this is not
///   likely to happen.
fn factorize_kkt(a: &CsMat<f64>, p: &CsMat<f64>, r: &Array1<f64>, s: &Array1<f64>) -> LdlNumeric {
    let s_diag = CsMat::new_csc(
        (s.len(), s.len()),
        (0..=s.len()).collect(),
        (0..s.len()).collect(),
        s.to_vec(),
    );
    let top_left = p + &s_diag;

    let neg_r_inv = -1. / r;
    let bottom_right = CsMat::new_csc(
        (r.len(), r.len()),
        (0..=r.len()).collect(),
        (0..r.len()).collect(),
        neg_r_inv.to_vec(),
    );

    // construct the block matrix K
    let k = bmat(&[
        [Some(top_left.view()), Some(a.transpose_view())],
        [Some(a.view()), Some(bottom_right.view())],
    ]);

    // attempt to come up with an LDLt factorization
    match LdlNumeric::new(k.view()) {
        Ok(v) => v,
        Err(e) => panic!("{:?}", e),
    }
}

/// Solves the KKT system set up in the description of `factorize_kkt` for a particular right hand side.
fn solve_kkt(
    variables: &Variables,
    problem_data: &ProblemData,
    rho: &Array1<f64>,
    sigma: &Array1<f64>,
    ldl: &LdlNumeric,
) -> (Array1<f64>, Array1<f64>) {
    // form the right hand side of the system we want to solve
    let top = sigma * &variables.x - &variables.w - &problem_data.q;
    let bottom = &variables.z - &(&variables.y / rho);
    let right_hand_side = stack![Axis(0), top, bottom];
    // solve the linear system using the ldl factorization
    let split_index = variables.n_vars();
    let mut solution = ldl.solve(&right_hand_side.to_vec());
    // split the soltion into xt and nu
    let nu = Array1::from(solution.split_off(split_index));
    let next_xt = Array1::from(solution);
    // compute zt using nu
    let next_zt = &variables.z + &(&(&nu - &variables.y) / rho);
    (next_xt, next_zt)
}

/// Carries out a single ADMM iteration by:
///     1. Solving the KKT system to carry out the `xt` and `zt` updates.
///     2. Using a mixture of the result of (1) and the previous `x` iterate, solves `n` univariate
///        subproblems by minimizing the Lagrangian in the description of `factorize_kkt` w.r.t.
///        `x`. (Requires evaluation of a proximal operator.)
///     3. Updating `w` and `y` using the residuals associated with the constraints `x = xt` and `z = zt`.
/// Note that for the entire run of the algorithm, the `z` update is just `z = b`, so we don't need
/// to do anything here.
fn step<U: Prox + ?Sized>(
    variables: &mut Variables,
    problem_data: &ProblemData,
    settings: &Settings,
    ldl: &LdlNumeric,
    pc: &mut U,
) {
    // update xt and zt by solving the KKT system
    let (next_xt, next_zt) = solve_kkt(
        variables,
        problem_data,
        &settings.rho,
        &settings.sigma,
        ldl,
    );
    // get the next x value by using the proximal operators of the separable function pieces
    let prox_arg = settings.alpha * &next_xt
        + (1. - settings.alpha) * &variables.x
        + &(&variables.w / &settings.sigma);
    let next_x = pc.prox(problem_data, settings, &prox_arg);
    // update the dual variables
    let next_w = &variables.w
        + &(&settings.sigma
            * &(settings.alpha * &next_xt + (1. - settings.alpha) * &variables.x - next_x));
    let next_y = &variables.y + &(settings.alpha * &settings.rho * &(&next_zt - &problem_data.b));

    variables.xt = next_xt;
    variables.zt = next_zt;
    variables.x = next_x.to_owned();
    variables.w = next_w;
    variables.y = next_y;
}

/// Carries out the ADMM algorithm by calling `admm_step` until termination conditions are met.
fn _optimize<T: Objective + Residual>(
    ldl: &LdlNumeric,
    pc: &mut UnconstrainedProxCache,
    tc: &mut TermCache,
    variables: &mut Variables,
    problem_data: &ProblemData,
    settings: &Settings,
    stats: &mut Stats,
    obj_res_calc: &T,
) -> Variables {
    let mut iters = 0;
    // Note: should_terminate modifies the variables if it returns true
    while iters < settings.max_iter
        && !tc.terminate(obj_res_calc, variables, problem_data, settings, iters)
    {
        iters += 1;
        step(variables, problem_data, settings, ldl, pc);
        stats.update(
            variables,
            problem_data,
            obj_res_calc,
            settings.compute_stats,
        );
    }
    stats.iters += iters;
    stats.prox_iters = pc.call_count();
    variables.to_owned()
}

/// Carry out the ADMM algorithm given `problem_data`, `settings`, and a structure that can compute objective
/// and residual values. If `convexify` is set to `true`, a convex relaxation is solved first, followed
/// by an attempt at the non-convex version with the result of the convex solve as a warm start.
///
/// # Panics
/// If `settings`, `problem_data`, or `variables` are not compatible with one another.
pub fn optimize_structs<T: Objective + Residual>(
    settings: &Settings,
    problem_data: &mut ProblemData,
    variables: &mut Variables,
    obj_res_calc: &T,
    convexify: bool,
) -> (Variables, Stats) {
    // verify that the dimensions of the settings, problem_data, and variables all mutually match up
    let begin_solve = Utc::now();
    validate(variables, settings, problem_data);
    variables.z = problem_data.b.to_owned();
    let mut term_cache = TermCache::new(problem_data, settings);
    let mut stats = Stats::new(settings.max_iter);
    let ldl = factorize_kkt(
        &problem_data.a,
        &problem_data.p,
        &settings.rho,
        &settings.sigma,
    );

    // solve the relaxed problem if convexify is true
    if convexify {
        let g_orig = problem_data.g.to_owned();
        // compute envelopes
        for i in 0..variables.n_vars() {
            problem_data.g[i] = envelope(&g_orig[i]);
        }
        // get the right prox cache
        let mut prox_cache = UnconstrainedProxCache::new(problem_data, settings);
        // solve the relaxed version
        let sol = _optimize(
            &ldl,
            &mut prox_cache,
            &mut term_cache,
            variables,
            problem_data,
            settings,
            &mut stats,
            obj_res_calc,
        );
        // warm start the next call with solution to the relaxed problem
        variables.x = sol.x;
        variables.z = sol.z;
        variables.w = sol.w;
        variables.y = sol.y;
        variables.xt = sol.xt;
        variables.zt = sol.zt;

        // reset the original g
        problem_data.g = g_orig;
    }

    // clear termination cache, create new prox cache, and solve again
    term_cache.clear();
    let mut prox_cache = UnconstrainedProxCache::new(problem_data, settings);
    let sol = _optimize(
        &ldl,
        &mut prox_cache,
        &mut term_cache,
        variables,
        problem_data,
        settings,
        &mut stats,
        obj_res_calc,
    );
    stats.solve_time_ms = (Utc::now() - begin_solve).num_milliseconds() as usize;
    stats
        .objective
        .push(obj_res_calc.objective(&sol, problem_data));
    stats
        .residual
        .push(obj_res_calc.residual(&sol, problem_data));
    (sol, stats)
}

/// Solves the optimization problem shown in the readme given problem data and something that can
/// compute objective and residuals.
#[allow(non_snake_case)]
pub fn optimize<T: Objective + Residual>(
    A: CsMat<f64>,
    b: Array1<f64>,
    P: CsMat<f64>,
    q: Array1<f64>,
    g: Array1<PiecewiseQuadratic>,
    o: &T,
    s: &Settings,
    convexify: bool,
) -> (Variables, Stats) {
    let mut problem_data = ProblemData::new(P, q, A, b, g);
    let mut variables = Variables::from_problem_data(&problem_data);
    optimize_structs(s, &mut problem_data, &mut variables, o, convexify)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::opto::structs::{ProblemData, Settings};
    use crate::quadratics::bq::BoundedQuadratic;
    use crate::quadratics::pwq::PiecewiseQuadratic;
    use approx::AbsDiffEq;
    use ndarray::linalg::Dot;
    use ndarray::{array, Array2};

    #[derive(Clone)]
    struct SimpleObjRes {}

    impl Objective for SimpleObjRes {
        fn objective(&self, vars: &Variables, problem_data: &ProblemData) -> f64 {
            let f_obj =
                vars.x.t().dot(&problem_data.p.dot(&vars.x)) + problem_data.q.t().dot(&vars.x);
            let g_obj: f64 = (0..problem_data.n_vars())
                .map(|i| problem_data.g[i].eval(vars.x[i]))
                .sum();
            f_obj + g_obj
        }
    }

    impl Residual for SimpleObjRes {
        fn residual(&self, vars: &Variables, problem_data: &ProblemData) -> f64 {
            (&problem_data.a.dot(&vars.x) - &problem_data.b).sum()
        }
    }

    #[test]
    fn test_unconstrained_qp() {
        let settings = Settings::new(1., array![1.], array![1., 1., 1., 1.], 50, 2, false);

        let problem_data = ProblemData::new(
            CsMat::csc_from_dense(Array2::eye(4).view(), f64::EPSILON),
            array![1., 1., 1., 1.],
            CsMat::csc_from_dense(Array2::zeros((1, 4)).view(), f64::EPSILON),
            array![1.],
            array![
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            ],
        );
        let variables = Variables::from_problem_data(&problem_data);
        let res = optimize_structs(
            &settings.to_owned(),
            &mut problem_data.to_owned(),
            &mut variables.to_owned(),
            &SimpleObjRes {},
            false,
        );
        let res_cvx = optimize_structs(
            &settings.to_owned(),
            &mut problem_data.to_owned(),
            &mut variables.to_owned(),
            &SimpleObjRes {},
            true,
        );
        assert!(res.0.x.abs_diff_eq(&array![-1., -1., -1., -1.], 1e-5));
        assert!(res_cvx.0.x.abs_diff_eq(&array![-1., -1., -1., -1.], 1e-5));
        assert!(res.1.iters < res_cvx.1.iters);
    }

    #[test]
    fn test_identity_constrained_qp() {
        let settings = Settings::new(1., array![1.], array![1., 1., 1., 1.], 50, 2, false);

        let mut problem_data = ProblemData::new(
            CsMat::csc_from_dense(Array2::eye(4).view(), f64::EPSILON),
            array![1., 1., 1., 1.],
            CsMat::csc_from_dense(Array2::zeros((1, 4)).view(), f64::EPSILON),
            array![1.],
            array![
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            ],
        );

        let mut variables = Variables::from_problem_data(&problem_data);
        let res = optimize_structs(
            &settings,
            &mut problem_data,
            &mut variables,
            &SimpleObjRes {},
            false,
        )
        .0
        .x;
        assert!(res.abs_diff_eq(&array![-1., -1., -1., -1.], 1e-5));
    }

    #[test]
    fn test_diag_constrained_qp() {
        let settings = Settings::new(1., array![1.], array![1., 1., 1., 1.], 50, 2, false);

        let mut problem_data = ProblemData::new(
            CsMat::csc_from_dense(
                Array2::from_diag(&array![1., 2., 3., 4.]).view(),
                f64::EPSILON,
            ),
            array![1., 2., 3., 4.],
            CsMat::csc_from_dense(Array2::ones((1, 4)).view(), f64::EPSILON),
            array![1.],
            array![
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            ],
        );

        let mut variables = Variables::from_problem_data(&problem_data);
        let res = optimize_structs(
            &settings,
            &mut problem_data,
            &mut variables,
            &SimpleObjRes {},
            false,
        )
        .0
        .x;
        assert!(res.abs_diff_eq(&array![1.4, 0.2, -0.2, -0.4], 1e-5));
    }

    #[test]
    fn test_with_rho_and_sigma() {
        let settings = Settings::new(1., array![2., 2.], array![2., 2.], 50, 2, false);

        let mut problem_data = ProblemData::new(
            CsMat::csc_from_dense(array![[65., 76.], [76., 89.]].view(), f64::EPSILON),
            array![7., 3.],
            CsMat::csc_from_dense(array![[7., 8.], [5., 4.]].view(), f64::EPSILON),
            array![8., 8.],
            array![
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            ],
        );

        let mut variables = Variables::from_problem_data(&problem_data);
        let res = optimize_structs(
            &settings,
            &mut problem_data,
            &mut variables,
            &SimpleObjRes {},
            false,
        )
        .0
        .x;
        assert!(res.abs_diff_eq(&array![2.66666651, -1.33333319], 1e-5))
    }

    #[test]
    fn test_with_gi_intervals_containing_feasible_point() {
        let settings = Settings::new(1., array![2.], array![2., 2.], 50, 2, false);

        let mut problem_data = ProblemData::new(
            CsMat::csc_from_dense(array![[5., 3.], [3., 2.]].view(), f64::EPSILON),
            array![1., 2.],
            CsMat::csc_from_dense(array![[1., 1.]].view(), f64::EPSILON),
            array![-3.],
            array![
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            ],
        );

        let mut variables = Variables::from_problem_data(&problem_data);

        let res = optimize_structs(
            &settings,
            &mut problem_data,
            &mut variables,
            &SimpleObjRes {},
            false,
        );

        assert!(res.0.x.abs_diff_eq(&array![4., -7.], 1e-5));
    }

    #[test]
    fn test_with_gi_quadratic() {
        let settings = Settings::new(1., array![5., 5.], array![5., 5.], 50, 2, false);

        let bq1 = BoundedQuadratic::new_extended(1., 0., 0.);
        let g1 = PiecewiseQuadratic::new(vec![bq1]);

        let mut problem_data = ProblemData::new(
            CsMat::csc_from_dense(array![[5., 3.], [3., 2.]].view(), f64::EPSILON),
            array![1., 2.],
            CsMat::csc_from_dense(array![[1., 1.], [1., 2.]].view(), f64::EPSILON),
            array![1., 1.],
            array![
                g1,
                PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY),
            ],
        );

        let mut variables = Variables::from_problem_data(&problem_data);
        let res = optimize_structs(
            &settings,
            &mut problem_data,
            &mut variables,
            &SimpleObjRes {},
            false,
        );
        assert!(res.0.x.abs_diff_eq(&array![1., 0.], 1e-5))
    }
}
