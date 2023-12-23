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

use crate::opto::structs::{HasConstraints, HasVariables, ProblemData, Settings, Variables};
use ndarray::Array1;

pub trait Objective {
    fn objective(&self, vars: &Variables, problem_data: &ProblemData) -> f64;
}

pub trait Residual {
    fn residual(&self, vars: &Variables, problem_data: &ProblemData) -> f64;
}

/// A cache of state that helps determine whether or not to terminate.
pub(crate) struct TermCache {
    /// The best encountered objective value.
    pub obj_best: f64,
    /// The best encountered residual value.
    pub res_best: f64,
    /// The best encountered x iterate.
    pub x_best: Array1<f64>,
    /// The best encountered z iterate.
    pub z_best: Array1<f64>,
    /// The best encountered w (dual) iterate.
    pub w_best: Array1<f64>,
    /// The best encountered y (dual) iterate.
    pub y_best: Array1<f64>,
    /// The number of iterations without improvement seeing improvement.
    pub not_improved_count: usize,
    /// The max allowed non-improved iterations.
    pub not_improved_count_max: usize,
}

impl TermCache {
    /// Instantiates a new `TermCache`.
    pub fn new(params: &ProblemData, settings: &Settings) -> TermCache {
        TermCache {
            obj_best: f64::INFINITY,
            res_best: f64::INFINITY,
            x_best: Array1::zeros(params.n_vars()),
            z_best: Array1::zeros(params.n_constrs()),
            w_best: Array1::zeros(params.n_vars()),
            y_best: Array1::zeros(params.n_constrs()),
            not_improved_count: 0,
            not_improved_count_max: (settings.non_improvement_iters as f64
                / settings.term_cond_freq as f64)
                .ceil() as usize,
        }
    }

    /// Clears a term cache for repeated use.
    pub fn clear(&mut self) {
        let n = self.x_best.len();
        let m = self.z_best.len();
        for i in 0..n {
            self.x_best[i] = 0.;
            self.w_best[i] = 0.;
        }
        for i in 0..m {
            self.z_best[i] = 0.;
            self.y_best[i] = 0.;
        }
        self.obj_best = f64::INFINITY;
        self.res_best = f64::INFINITY;
        self.not_improved_count = 0;
    }

    /// Returns `true` if iteration should terminate, and `false` otherwise.
    ///
    /// At a high level, optimization should terminate if a certain number of iterations have
    /// elapsed without objective or residual improvement.
    pub fn terminate<T: Objective + Residual>(
        &mut self,
        obj_res_calc: &T,
        vars: &mut Variables,
        params: &ProblemData,
        settings: &Settings,
        iter: usize,
    ) -> bool {
        // We only want to check for convergence every so often
        if iter % settings.term_cond_freq == 0 {
            let objective = obj_res_calc.objective(vars, params);
            let res = obj_res_calc.residual(vars, params);
            // in order to have "improved", the objective value must decrease, the linear and non
            // linear residuals must be sufficiently small
            let has_improved =
                objective < self.obj_best - settings.obj_tol && res < settings.res_tol;
            if has_improved {
                self.not_improved_count = 0;
            } else {
                self.not_improved_count += 1;
            }
            // if found a new best objective/residual pair, so save the iterate
            if objective < self.obj_best && res < settings.res_tol {
                self.x_best = vars.x.to_owned();
                self.z_best = vars.z.to_owned();
                self.w_best = vars.w.to_owned();
                self.y_best = vars.y.to_owned();
                self.obj_best = objective;
                self.res_best = res;
            }
            // if there hasn't been sufficient improvement over the best for a certain number of
            // iterations, set the vars to contain the best iterate
            let converged =
                self.obj_best.is_finite() && self.not_improved_count > self.not_improved_count_max;
            if converged {
                vars.x = self.x_best.to_owned();
                vars.z = self.z_best.to_owned();
                vars.w = self.w_best.to_owned();
                vars.y = self.y_best.to_owned();
            }
            return converged;
        }
        false
    }
}
