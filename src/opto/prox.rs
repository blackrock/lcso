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

use crate::opto::structs::{HasVariables, ProblemData, Settings};
use crate::quadratics::envelope::envelope;
use crate::quadratics::pwq::PiecewiseQuadratic;
use ndarray::Array1;
use std::f64;
use std::vec::Vec;

/// Requires implementation of a function that computes the `prox_f(u)`, where `f` is implied
/// by the `problem_data`.
pub trait Prox {
    fn prox(&mut self, data: &ProblemData, settings: &Settings, u: &Array1<f64>) -> &Array1<f64>;
    fn call_count(&self) -> usize;
}

/// An object that stores proximal operators corresponding to each element of a vector of `PiecewiseQuadratic`s.
/// The proximal operator of `f`, `rho`, denoted `prox_{f, rho}(u) = argmin_{x in dom(f)} f(x) + (rho/2)||x - u||_2^2`.
#[derive(Debug)]
pub struct UnconstrainedProxCache {
    /// The functions that can evaluate unconstrained proximal operators.
    proxes: Vec<PiecewiseQuadratic>,
    /// The total number of proximal operators evaluated during an optimization.
    call_count: usize,
    /// Memory to be reused holding the proximal operator result.
    x_best: Array1<f64>,
}

impl UnconstrainedProxCache {
    /// Constructs a new `SeparableProxCache` by computing `prox_{g_i, sigma_i}(u)` for each `g_i`
    /// in `gs` and `sigma_i` in `sigma`.
    pub(crate) fn new(problem_data: &ProblemData, settings: &Settings) -> Self {
        assert_eq!(settings.sigma.len(), problem_data.g.shape()[0]);
        let n = problem_data.g.len();
        let mut proxes = Vec::with_capacity(n);
        for i in 0..n {
            let prox = Self::prox_operator(&problem_data.g[i], settings.sigma[i]);
            proxes.push(prox);
        }
        UnconstrainedProxCache {
            proxes,
            call_count: 0,
            x_best: Array1::from_elem(problem_data.n_vars(), f64::NAN),
        }
    }

    /// In what follows, rho is a scalar. The proximal operator of `f` evaluated at `u` is denoted
    ///
    /// `prox_f(u) = argmin_{x in dom(f)} f(x) + (rho / 2) * (x - u)^2`.
    ///
    /// By some algebraic manipulation, we can deduce that
    ///
    /// `prox_f(u) = -argmax_x (rho * u) * x - (f(x) + rho * x^2 / 2)`.
    ///
    /// If we let `g(x) = f(x) + (rho / 2) * x^2`, then `prox_f(u) = -argmax_x (rho * u) * x - g(x)`.
    /// If we replaced the argmax with a max, this would be `-g^*(rho * u)` where `g^*` is the
    /// conjugate of `g`.
    ///
    /// Thus, `prox_f(u) = x` that minimizes `-g^*(rho * u)`.
    ///
    /// # Panics:
    /// * If the output (a `PiecewiseQuadratic`) does not extend to infinity in both directions
    fn prox_operator(f: &PiecewiseQuadratic, rho: f64) -> PiecewiseQuadratic {
        let mut f = f.to_owned();
        // Add quadratic to each piece of f
        for f in &mut f.functions {
            f.a += rho / 2.;
        }
        // Take the envelope of the sum so that we are taking the conjugate maximizer of a convex function.
        let env = envelope(&f);
        // Scale by rho so that we don't have to deal with rho * u in the argument.
        let mut cm = env.conjugate_maximizer();
        cm.scale_arg_in_place(rho);
        // The conj maximizer should extend to inf in both directions (because we added a quadratic
        assert!(cm.extends_left(), "{} does not extend to -inf", cm);
        assert!(cm.extends_right(), "{} does not extend to inf", cm);
        cm
    }
}

impl Prox for UnconstrainedProxCache {
    /// Evaluates each element in `u` using the cached `proxes`, quantizing if necessary.
    fn prox(
        &mut self,
        _problem_data: &ProblemData,
        _settings: &Settings,
        u: &Array1<f64>,
    ) -> &Array1<f64> {
        assert_eq!(self.proxes.len(), u.len());
        for i in 0..self.proxes.len() {
            let prox_gi = &self.proxes[i];
            self.x_best[i] = prox_gi.eval(u[i]);
        }
        self.call_count += 1;
        &self.x_best
    }

    fn call_count(&self) -> usize {
        self.call_count
    }
}

#[cfg(test)]
mod tests {

    use crate::opto::prox::UnconstrainedProxCache;
    use crate::quadratics::bq::BoundedQuadratic;
    use crate::quadratics::pwq::PiecewiseQuadratic;
    use len_trait::len::Len;
    use std::f64;

    #[test]
    fn test_prox_of_indicator() {
        let indicator01 = PiecewiseQuadratic::indicator(0., 1.);
        let expected_first = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., 0., 0.);
        let expected_second = BoundedQuadratic::new(0., 1., 0., 1., 0.);
        let expected_third = BoundedQuadratic::new(1., f64::INFINITY, 0., 0., 1.);
        let prox = UnconstrainedProxCache::prox_operator(&indicator01, 1.);
        assert!(prox[0].approx(&expected_first));
        assert!(prox[1].approx(&expected_second));
        assert!(prox[2].approx(&expected_third));
    }

    #[test]
    fn test_prox_of_abs_val() {
        let abs = PiecewiseQuadratic::new(vec![
            BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.),
            BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.),
        ]);
        let expected_first = BoundedQuadratic::new(f64::NEG_INFINITY, -1., 0., 1., 1.);
        let expected_second = BoundedQuadratic::new(-1., 1., 0., 0., 0.);
        let expected_third = BoundedQuadratic::new(1., f64::INFINITY, 0., 1., -1.);
        let prox = UnconstrainedProxCache::prox_operator(&abs, 1.);
        assert_eq!(prox.len(), 3, "{}", prox);
        assert!(prox[0].approx(&expected_first));
        assert!(prox[1].approx(&expected_second));
        assert!(prox[2].approx(&expected_third));
    }

    #[test]
    fn test_prox_of_quadratic() {
        let quad = PiecewiseQuadratic::new(vec![BoundedQuadratic::new_extended(1., 1., 1.)]);
        let expected = BoundedQuadratic::new_extended(0., 1. / 3., -1. / 3.);
        let prox = UnconstrainedProxCache::prox_operator(&quad, 1.);
        assert_eq!(prox.len(), 1);
        assert!(prox[0].approx(&expected));
    }

    #[test]
    fn test_prox_of_f_with_nontrivial_envelope() {
        let linear_piece = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., 0.5, 0.);
        let quadratic_piece = BoundedQuadratic::new(0., f64::INFINITY, 1., 0., 0.);
        let f = PiecewiseQuadratic::new(vec![linear_piece, quadratic_piece]);
        let prox = UnconstrainedProxCache::prox_operator(&f, 1.);
        let expected_first =
            BoundedQuadratic::new(f64::NEG_INFINITY, 0.3169872981077807, 0., 1., -0.5);
        let expected_second =
            BoundedQuadratic::new(0.3169872981077807, f64::INFINITY, 0., 1. / 3., 0.);
        assert_eq!(prox.len(), 2);
        assert!(prox[0].approx(&expected_first));
        assert!(prox[1].approx(&expected_second));
    }

    #[test]
    fn test_rho_non_one() {
        let abs = PiecewiseQuadratic::new(vec![
            BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.),
            BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.),
        ]);
        let expected_first = BoundedQuadratic::new(f64::NEG_INFINITY, -1. / 3., 0., 1., 1. / 3.);
        let expected_second = BoundedQuadratic::new(-1. / 3., 1. / 3., 0., 0., 0.);
        let expected_third = BoundedQuadratic::new(1. / 3., f64::INFINITY, 0., 1., -1. / 3.);
        let prox = UnconstrainedProxCache::prox_operator(&abs, 3.);
        assert_eq!(prox.len(), 3);
        assert!(prox[0].approx(&expected_first));
        assert!(prox[1].approx(&expected_second));
        assert!(prox[2].approx(&expected_third));
    }
}
