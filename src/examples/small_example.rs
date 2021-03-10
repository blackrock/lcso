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

use lcso::opto::admm;
use lcso::opto::structs::{ProblemData, Variables};
use lcso::opto::term::{Objective, Residual};
use lcso::quadratics::pwq::PiecewiseQuadratic;
use ndarray::linalg::Dot;
use ndarray::{array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::SeedableRng;
use sprs::CsMat;
use lcso::quadratics::bq::BoundedQuadratic;

struct ObjRes {}

impl Objective for ObjRes {
    fn objective(&self, vars: &Variables, problem_data: &ProblemData) -> f64 {
        let quad_plus_lin =
            vars.x.t().dot(&problem_data.p.dot(&vars.x)) + problem_data.q.t().dot(&vars.x);
        let mut sep = 0.;
        for i in 0..vars.x.len() {
            sep += problem_data.g[i].eval(vars.x[i]);
        }
        quad_plus_lin + sep
    }
}

impl Residual for ObjRes {
    fn residual(&self, vars: &Variables, problem_data: &ProblemData) -> f64 {
        (problem_data.a.dot(&vars.x) - &problem_data.b).sum().abs()
    }
}

#[allow(non_snake_case)]
pub fn main() {
    let mut rng: StdRng = SeedableRng::seed_from_u64(1);
    let u01 = Uniform::new(0., 1.);
    let (m, n) = (2, 4);

    // Construct problem data (ensuring that the problem is feasible)
    let x0 = Array1::random_using(n, u01, &mut rng);
    let A = CsMat::csc_from_dense(
        Array2::random_using((m, n), u01, &mut rng).view(),
        f64::EPSILON,
    );
    let b = A.dot(&x0);
    let X = Array2::random_using((n, n), u01, &mut rng);
    // Make sure P is positive definite
    let P = CsMat::csc_from_dense(X.t().dot(&X).view(), f64::EPSILON);
    let q = Array1::random_using(n, u01, &mut rng);

    // x1 has to be in union([-1, 2], [2.5, 3.5]) and has a quadratic penalty if it lies in [-1, 2]
    // and a linear penalty if it lies in [2.5, 3.5]
    let g11 = BoundedQuadratic::new(-1., 2., 1., 0., 0.);
    let g12 = BoundedQuadratic::new(2.5, 3.5, 0., 1., 0.);
    let g1 = PiecewiseQuadratic::new(vec![g11, g12]);
    // x2 has to be between -20 and 10
    let g2 = PiecewiseQuadratic::indicator(-20., 10.);
    // x3 has to be between -5 and 10
    let g3 = PiecewiseQuadratic::indicator(-5., 10.);
    // x4 has to be exactly 1.2318
    let g4 = PiecewiseQuadratic::indicator(1.2318, 1.2318);

    let g = array![g1, g2, g3, g4];

    let o = ObjRes {};

    println!("A: {:.4}", A.to_dense());
    println!("b: {:.4}", b);

    let (vars, stats) = admm::optimize(A, b, P, q, g, &o, false);
    println!("optimal x: {:.4}", vars.x);
    println!("optimal value: {:.4}", stats.objective[0]);
    println!("optimal residual: {:.10}", stats.residual[0]);
    println!("runtime: {} ms", stats.solve_time_ms);
}
