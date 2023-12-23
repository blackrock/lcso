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

use crate::quadratics::bq::BoundedQuadratic;
use crate::quadratics::utils;
use len_trait::{Empty, Len};
use num::traits::Zero;
use std::f64;
use std::fmt;
use std::ops::{Add, Index};
use std::vec::Vec;

/// A structure useful for carrying out synchronization of `PiecewiseQuadratic`s.
pub struct SyncWorkspace {
    /// The cross section of functions that will be synchronized next.
    pub active_functions: Vec<BoundedQuadratic>,
    /// The index of each active `BoundedQuadratic` within its parent `PiecewiseQuadratic`.
    pub active_piece_idx: Vec<usize>,
    /// A boolean corresponding to each `PiecewiseQuadratic` indicating whether or not it's active.
    pub pwq_is_active: Vec<bool>,
    /// Upper bounds of the active `PiecewiseQuadratics`.
    pub uppers: Vec<f64>,
    /// A boolean corresponding to each `PiecewiseQuadratic` indicating whether or not to increment
    /// its active member.
    pub incr_pwq: Vec<bool>,
    /// The number of functions that will be synchronized with this workspace.
    pub capacity: usize,
}

impl SyncWorkspace {
    pub fn new(capacity: usize) -> SyncWorkspace {
        let active_functions = Vec::<BoundedQuadratic>::with_capacity(capacity);
        //unsafe { active_functions.set_len(capacity) }

        let active_piece_idx = Vec::<usize>::with_capacity(capacity);
        //unsafe { active_piece_idx.set_len(capacity) }

        let pwq_is_active = Vec::<bool>::with_capacity(capacity);
        //unsafe { pwq_is_active.set_len(capacity) }

        let uppers = Vec::<f64>::with_capacity(capacity);
        //unsafe { uppers.set_len(capacity) }

        let incr_pwq = Vec::<bool>::with_capacity(capacity);
        //unsafe { incr_pwq.set_len(capacity) }

        SyncWorkspace {
            active_functions,
            active_piece_idx,
            pwq_is_active,
            uppers,
            incr_pwq,
            capacity,
        }
    }
}

/// A piecewise quadratic function. Each piece is a [`BoundedQuadratic`](../bq/struct.BoundedQuadratic.html).
#[derive(Debug, Clone)]
pub struct PiecewiseQuadratic {
    /// The `BoundedQuadratic` functions that make up the `PiecewiseQuadratic`.
    pub functions: Vec<BoundedQuadratic>,
}

impl PiecewiseQuadratic {
    /// Initializes a new `PiecewiseQuadratic`.
    ///
    /// # Panics
    /// * If the intervals are not in order.
    ///
    /// # Example
    /// ```
    /// // constructing f(x) = |x| as a piecewise quadratic
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// use lcso::quadratics::pwq::PiecewiseQuadratic;
    /// let left = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1.,0.);
    /// let right = BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.);
    /// let abs = PiecewiseQuadratic::new(vec![left, right]);
    /// // note that if the argument were vec![right, left], the above would panic
    /// ```
    pub fn new(functions: Vec<BoundedQuadratic>) -> PiecewiseQuadratic {
        if !functions.is_empty() {
            for i in 1..functions.len() - 1 {
                assert!(
                    utils::approx_le(functions[i - 1].upper, functions[i].lower),
                    "Domains of pieces must be in order"
                )
            }
        }
        PiecewiseQuadratic { functions }
    }

    /// Initializes an empty `PiecewiseQuadratic` with a known capacity (can be used to prevent reallocating
    /// as pieces are added).
    pub fn new_empty_with_capacity(capacity: usize) -> PiecewiseQuadratic {
        PiecewiseQuadratic {
            functions: Vec::with_capacity(capacity),
        }
    }

    /// Add a piece at the right of the rightmost piece in `self`.
    ///
    /// # Panics:
    /// * If the added piece is not to the right of the existing rightmost piece (domain-wise)
    ///
    /// # Example:
    /// ```
    /// // constructing f(x) = |x| as a piecewise quadratic
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// use lcso::quadratics::pwq::PiecewiseQuadratic;
    /// let first = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1.,0.);
    /// let second = BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.);
    /// let mut new = PiecewiseQuadratic::new_empty_with_capacity(2);
    /// new.add_piece_at_right(first);
    /// new.add_piece_at_right(second);
    /// // note that if we reversed the order, it would panic
    /// ```
    pub fn add_piece_at_right(&mut self, bq: BoundedQuadratic) {
        if self.is_empty() {
            self.functions.push(bq);
            return;
        }
        let rightmost = self.functions.last().unwrap();
        assert!(
            utils::approx_le(rightmost.upper, bq.lower),
            "Cannot add a piece whose to domain is not to the right of the rest"
        );
        self.functions.push(bq);
    }

    /// Constructs a function that is 0 on the interval demarcated by `lower` and `upper` and `inf`
    /// everywhere else.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::pwq::PiecewiseQuadratic;
    /// // The indicator function on the interval [0, 1]
    /// let indicator01 = PiecewiseQuadratic::indicator(0., 1.);
    /// ```
    pub fn indicator(lower: f64, upper: f64) -> PiecewiseQuadratic {
        let zero = BoundedQuadratic::new(lower, upper, 0., 0., 0.);
        PiecewiseQuadratic::new(vec![zero])
    }

    /// Determines whether a `PiecewiseQuadratic` is convex. A `PiecewiseQuadratic` is convex if for all `i`:
    /// * `f_i` is convex
    /// * `f_i.upper == f_{i+1}.lower`
    /// * `derivative(f_i)(f_i.upper) <= derivative(f_{i+1})(f_{i+1}.lower)`
    pub fn is_convex(&self) -> bool {
        if self.len() <= 1 {
            return self[0].is_convex();
        }
        for i in 0..self.len() - 1 {
            let f_left = self[i];
            let f_right = self[i + 1];
            let left_derivative = f_left.eval_derivative(f_left.upper);
            let right_derivative = f_right.eval_derivative(f_right.lower);
            // not convex if not continuous or f_curr's derivative is greater than f_next's
            if !f_left.overlaps_continuously_with(&f_right)
                || left_derivative - right_derivative > f64::EPSILON
                || !f_left.is_convex()
            {
                return false;
            }
        }
        true
    }

    /// Returns true if `self` extends to negative infinity and false otherwise.
    pub fn extends_left(&self) -> bool {
        self[0].lower == f64::NEG_INFINITY
    }

    /// Returns true if `self` extends to infinity and false otherwise.
    pub fn extends_right(&self) -> bool {
        self[self.len() - 1].upper == f64::INFINITY
    }

    /// Minimizes `self` by minimizing each piece and taking the minimum over all of those.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// use lcso::quadratics::pwq::PiecewiseQuadratic;
    /// let left = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.);
    /// let right = BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.);
    /// let abs = PiecewiseQuadratic::new(vec![left, right]);
    /// assert_eq!(abs.minimize(), (0., 0., 0));
    /// ```
    pub fn minimize(&self) -> (f64, f64, usize) {
        let (mut min_x, mut min_val, mut min_idx) = (f64::NAN, f64::INFINITY, 0);
        for (i, fi) in self.functions.iter().enumerate() {
            let (cand_x, cand_val) = fi.minimize();
            if utils::lt_ep(cand_val, min_val) {
                min_val = cand_val;
                min_x = cand_x;
                min_idx = i;
            }
        }
        (min_x, min_val, min_idx)
    }

    /// Pre-scales each function by `alpha`. For more information, see `BoundedQuadratic::scale`
    /// in [`BoundedQuadratic`](../bq/struct.BoundedQuadratic.html).
    pub fn scale(&self, scale_factor: f64) -> PiecewiseQuadratic {
        let functions: Vec<_> = self
            .functions
            .iter()
            .map(|&f| f.scale(scale_factor))
            .collect();
        PiecewiseQuadratic::new(functions)
    }

    pub fn scale_in_place(&mut self, scale_factor: f64) {
        for fi in &mut self.functions {
            fi.scale_in_place(scale_factor)
        }
    }

    /// Scales the argument of each function by `alpha`. For more information, see
    /// `BoundedQuadratic::scale_arg` in [`BoundedQuadratic`](../bq/struct.BoundedQuadratic.html).
    pub fn scale_arg(&self, scale_factor: f64) -> PiecewiseQuadratic {
        assert!(!relative_eq!(scale_factor, 0., epsilon = f64::EPSILON));
        let mut functions: Vec<_> = self
            .functions
            .iter()
            .map(|&f| f.scale_arg(scale_factor))
            .collect();
        if scale_factor < 0. {
            functions.reverse();
        }
        PiecewiseQuadratic::new(functions)
    }

    pub fn scale_arg_in_place(&mut self, scale_factor: f64) {
        assert!(relative_ne!(scale_factor, 0., epsilon = f64::EPSILON));
        for fi in &mut self.functions {
            fi.scale_arg_in_place(scale_factor)
        }
        if scale_factor < 0. {
            self.functions.reverse();
        }
    }

    /// Returns perspective of each function with argument `alpha`. For more information, see
    /// `BoundedQuadratic::perspective` in [`BoundedQuadratic`](../bq/struct.BoundedQuadratic.html).
    pub fn perspective(&self, scale_factor: f64) -> PiecewiseQuadratic {
        assert!(!relative_eq!(scale_factor, 0., epsilon = f64::EPSILON));
        let mut functions: Vec<_> = self
            .functions
            .iter()
            .map(|&f| f.perspective(scale_factor))
            .collect();
        if scale_factor < 0. {
            functions.reverse();
        }
        PiecewiseQuadratic::new(functions)
    }

    pub fn perspective_in_place(&mut self, scale_factor: f64) {
        assert!(!relative_eq!(scale_factor, 0., epsilon = f64::EPSILON));
        for fi in &mut self.functions {
            fi.perspective_in_place(scale_factor);
        }
        if scale_factor < 0. {
            self.functions.reverse();
        }
    }

    /// Shifts each function by `alpha`. For more information, see `BoundedQuadratic::shift` in [`BoundedQuadratic`](../bq/struct.BoundedQuadratic.html).
    pub fn shift(&self, offset: f64) -> PiecewiseQuadratic {
        PiecewiseQuadratic::new(self.functions.iter().map(|&f| f.shift(offset)).collect())
    }

    pub fn shift_in_place(&mut self, offset: f64) {
        for fi in &mut self.functions {
            fi.shift_in_place(offset);
        }
    }

    /// Reflects each function over the y-axis. For more information, see `BoundedQuadratic::shift`
    /// in [`BoundedQuadratic`](../bq/struct.BoundedQuadratic.html).
    pub fn reflect_over_y(&self) -> PiecewiseQuadratic {
        let mut reflected_functions: Vec<_> =
            self.functions.iter().map(|&f| f.reflect_over_y()).collect();
        reflected_functions.reverse();
        PiecewiseQuadratic::new(reflected_functions)
    }

    pub fn reflect_over_y_in_place(&mut self) {
        for fi in &mut self.functions {
            fi.reflect_over_y_in_place();
        }
        self.functions.reverse();
    }

    /// Computes the x that maximizes the conjugate of `self`. The output is always piecewise affine.
    /// The maximizing x for a given slope y turns out to be the x at which fi'(x) = y (where fi is a single
    /// piece of `self`). For each piece fi(x) = ax^2 + bx + c with x in (li, ri), we can write y in terms of x as:
    /// y = fi'(x) = 2ax + b for x in (li, ri). Rearranging for x in terms of y, we get
    /// x(y) = y/2a - b/2a for y in (2a*li + b, 2a*ri + b).
    ///
    /// In the case that there is a gap between the right y bound computed with the _previous_ fi and the
    /// left y bound computed with the _current_ fi, we set x(y) = li between them (i.e., we assign all
    /// intervening slopes to the x value at the kink between fi-1 and fi).
    ///
    /// (This function is implemented for use in opto.)
    ///
    /// # Panics
    /// * The argument is not convex (see `is_convex`).
    ///
    /// # Examples
    /// For examples, see the test cases below.
    pub fn conjugate_maximizer(&self) -> PiecewiseQuadratic {
        assert!(
            self.is_convex(),
            "Can only find the conjugate maximizer of convex functions. {} is not convex.",
            self
        );
        let mut xstar = Vec::<BoundedQuadratic>::with_capacity(self.len());
        let mut i = 0;
        let mut prev_ub = f64::NEG_INFINITY;
        if self[0].is_affine() && self.extends_left() {
            i = 1;
            prev_ub = self[0].b;
        };
        while i < self.len() {
            let fi = self[i];
            let change_vars = |x: f64| 2. * fi.a * x + fi.b;
            // note lower_slope <= upper_slope because fi is convex
            let lb = change_vars(fi.lower);
            let ub = change_vars(fi.upper);
            // fill gap in ys if necessary. the x we assign to the intervening slopes is the lower
            // bound of f_i (or the upper bound of f_{i-1}), which is a kink beteween f_i and f_{i-1}
            if lb > prev_ub {
                let constant_lb = BoundedQuadratic::new(prev_ub, lb, 0., 0., fi.lower);
                xstar.push(constant_lb);
            }
            if fi.a > 0. {
                // solve so that we have have a function of x in terms of slopes y
                // ==> x(y) = y/(2a) - b/(2a)
                let slope_to_x_map =
                    BoundedQuadratic::new(lb, ub, 0., 1. / (2. * fi.a), -fi.b / (2. * fi.a));
                xstar.push(slope_to_x_map)
            }
            // if fi.a == 0, gets handled in the next iteration of the loop
            prev_ub = ub;
            i += 1;
        }

        // if the last function has a finite upper bound, add a piece that extends to infinity
        let last = self.functions.last().unwrap();
        if last.upper.is_finite() {
            let to_inf = BoundedQuadratic::new(prev_ub, f64::INFINITY, 0., 0., last.upper);
            xstar.push(to_inf);
        }
        PiecewiseQuadratic::new(xstar)
    }

    /// Evaluates a `PiecewiseQuadratic` at a point. In the case that the function is multiply defined
    /// at `x`, we adopt the convention that the function takes the minimum of all obtained values.
    pub fn eval(&self, x: f64) -> f64 {
        // binary search for domain that contains point
        let mut right = false;
        let mut left = false;
        let mut idx = -1;
        let mut cand = f64::INFINITY;
        let mut lo = 0_i64;
        let mut hi = (self.len() - 1) as i64;
        while lo <= hi {
            let mid = (hi + lo) / 2;
            let f_mid = self[mid as usize];
            if f_mid.interior_contains_element(x) {
                // if it's in the interior of a piece, no other pieces could supersede
                return f_mid.eval(x);
            } else if f_mid.domain_contains_element(x) {
                // if value is at one of the endpoints of the domain, must check pieces to left or right
                idx = mid;
                cand = f_mid.eval(x);
                // record whether we need to look to the left or the right of this piece after breaking
                if relative_eq!(x, f_mid.lower, epsilon = f64::EPSILON) {
                    left = true;
                }
                if relative_eq!(x, f_mid.upper, epsilon = f64::EPSILON) {
                    right = true;
                }
                break;
            }
            if utils::approx_ge(x, f_mid.upper) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        if idx == -1 {
            return f64::INFINITY;
        }

        // find minimum value among pieces whose domains x is in
        if left {
            let mut next_idx = idx - 1;
            // search until the domain is too far to the left
            while next_idx >= 0 && self[next_idx as usize].domain_contains_element(x) {
                let next_cand = self[next_idx as usize].eval(x);
                if utils::approx_le(next_cand, cand) {
                    cand = next_cand;
                }
                next_idx -= 1;
            }
        }
        if right {
            let mut next_idx = idx + 1;
            // search until the domain is too far to the left
            while (next_idx as usize) < self.len()
                && self[next_idx as usize].domain_contains_element(x)
            {
                let next_cand = self[next_idx as usize].eval(x);
                if utils::approx_le(next_cand, cand) {
                    cand = next_cand;
                }
                next_idx += 1;
            }
        }
        cand
    }

    /// Simplifies a `PiecewiseQuadratic`. fi and and the current rightmost piece should be combined
    /// (to become the new rightmost piece) if
    /// * Either of fi or the rightmost piece is a point and they overlap at their lower and upper
    /// endpoints respectively
    /// * fi and the rightmost piece have the same coefficients and they correspond at their lower and
    /// upper endpoints respectively (they're just currently separate parts of the same function).
    ///
    /// We also eliminate functions that are points at inf or -inf if they come up.
    ///
    /// # Examples
    /// ```
    /// use len_trait::len::Len;
    /// use lcso::quadratics::pwq::PiecewiseQuadratic;
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// let f = BoundedQuadratic::new(-1., 0., 0., -1., 0.);
    /// let pt = BoundedQuadratic::new(0., 0., 0., 0., 0.);
    /// let g = BoundedQuadratic::new(0., 1., 0., 1., 0.);
    /// let h = BoundedQuadratic::new(1., 2., 0., 1., 0.);
    /// let c = BoundedQuadratic::new(3., 5., 1., 1., 1.);
    /// let p = PiecewiseQuadratic::new(vec![f, pt, g, h, c]);
    /// let s = p.simplify();
    /// assert_eq!(s.len(), 3);
    /// assert!(s[0].approx(&f));
    /// assert!(s[1].approx(&BoundedQuadratic::new(0., 2., 0., 1., 0.)));
    /// assert!(s[2].approx(&c));
    /// ```
    pub fn simplify(&self) -> PiecewiseQuadratic {
        if self.len() <= 1 {
            return self.clone();
        }
        let (mut start, end) = (0, self.len());
        if self[start].is_empty() {
            start += 1;
        }
        let mut simplified = PiecewiseQuadratic::new_empty_with_capacity(end - start);
        simplified.add_piece_at_right(self[start]);
        for &curr in &self.functions[start + 1..end] {
            if curr.is_empty() {
                continue;
            }
            let last = simplified.functions.last().unwrap();
            // if we have redundant points, eliminate all but the one that takes the minimum value
            if last.is_point()
                && curr.is_point()
                && relative_eq!(curr.lower, last.lower, epsilon = f64::EPSILON)
            {
                let curr_val = curr.eval(curr.lower);
                let last_val = last.eval(last.upper);
                // if the value of the "right" point is at least the value of the left point,
                // do not consider it.
                if utils::approx_ge(curr_val, last_val) {
                    continue;
                }
            }
            if last.overlaps_continuously_with(&curr)
                && ((last.is_point() || curr.is_point()) || last.same_coefficients(&curr))
            {
                // take the coefficients of whichever function is not a point. note that if
                // we are combining and neither function is a point, both coefficient sets must be
                // the same, so it doesn't matter which we take.
                let (a, b, c) = if last.is_point() {
                    (curr.a, curr.b, curr.c)
                } else {
                    (last.a, last.b, last.c)
                };
                let combined = BoundedQuadratic::new(last.lower, curr.upper, a, b, c);
                let last_ind = simplified.len() - 1;
                simplified.functions[last_ind] = combined;
            } else {
                // if we don't combine, we just add the current piece to the end of the result
                simplified.add_piece_at_right(curr);
            }
        }
        simplified
    }

    /// Takes the sum of the piecewise quadratic functions in `fns` in a memory efficient way. At each step,
    /// there is an active set of `BoundedQuadratic`s whose sum makes up the next piece of the `PiecewiseQuadratic`
    /// result. After each piece is added, the active set is updated and the above is repeated until the active set
    /// is empty.
    ///
    ///
    /// # Panics
    /// * The capacity of the workspace doesn't match the length of the list of functions to add.
    /// * Any of the functions has length 0.
    /// * The workspace has 0 capacity.
    pub fn sum_pwq(
        work: &mut SyncWorkspace,
        fns: &[&PiecewiseQuadratic],
        syncd: bool,
    ) -> PiecewiseQuadratic {
        if !syncd {
            let synchronized = PiecewiseQuadratic::synchronize(work, fns);
            let synchronized_refs = synchronized.iter().collect::<Vec<_>>();
            PiecewiseQuadratic::synchronized_sum_pwq(&synchronized_refs)
        } else {
            PiecewiseQuadratic::synchronized_sum_pwq(fns)
        }
    }

    fn synchronized_sum_pwq(fns: &[&PiecewiseQuadratic]) -> PiecewiseQuadratic {
        // if not already synchronized, synchronize before adding
        let n_intervals = fns[0].len();
        let mut result = PiecewiseQuadratic::new_empty_with_capacity(n_intervals);
        for i in 0..n_intervals {
            let to_sum = fns.iter().map(|&f| f[i]).collect::<Vec<_>>();
            // note that because of the call to synchronize, sum_many will always return something
            if let Some(sum) = BoundedQuadratic::sum_bq(&to_sum) {
                result.add_piece_at_right(sum);
            }
        }
        result
    }

    fn identify_functions_to_update(
        bqs: &[BoundedQuadratic],
        active: &[bool],
        uppers: &mut [f64],
        update: &mut [bool],
    ) {
        // Find the minimum upper bound (updating the upper bounds and their closedness along the way)
        let mut min_ub = f64::INFINITY;
        for i in 0..bqs.len() {
            if active[i] {
                uppers[i] = bqs[i].upper;
                if utils::lt_ep(uppers[i], min_ub) {
                    min_ub = uppers[i];
                }
            }
        }

        // should increment the active function for a pwq if its upper bound is equal the min ub of
        // current active functions.
        for i in 0..bqs.len() {
            update[i] = active[i] && relative_eq!(uppers[i], min_ub, epsilon = f64::EPSILON)
        }
    }

    /// Synchronizes the domains of a collection of `PiecewiseQuadraticFunctions`.
    ///
    /// # Panics
    /// * If any of the functions are empty.
    /// * The capacity of the workspace is 0.
    pub fn synchronize(
        work: &mut SyncWorkspace,
        fns: &[&PiecewiseQuadratic],
    ) -> Vec<PiecewiseQuadratic> {
        assert!(
            fns.iter().all(|f| f.len() >= 1),
            "all functions must be nonempty"
        );
        assert!(work.capacity > 0, "workspace must have nonzero capacity");

        let max_intervals = fns.iter().map(|&f| f.len()).max().unwrap();

        // all first pieces should be marked as active
        for (i, &pwq) in fns.iter().enumerate() {
            work.active_functions[i] = pwq[0];
            work.active_piece_idx[i] = 0;
            work.pwq_is_active[i] = pwq.len() > 1; // TODO: why?
        }

        // set up output
        let mut out = Vec::with_capacity(fns.len());
        for _ in 0..fns.len() {
            out.push(PiecewiseQuadratic::new_empty_with_capacity(max_intervals))
        }

        if let Some(active) = BoundedQuadratic::restrict_to_common_domain(&work.active_functions) {
            if !active[0].is_empty() {
                for i in 0..active.len() {
                    out[i].add_piece_at_right(active[i]);
                }
            }
        }

        while work.pwq_is_active.iter().any(|&x| x) {
            // identify which of the pwq pointers need to be incremented
            PiecewiseQuadratic::identify_functions_to_update(
                &work.active_functions,
                &work.pwq_is_active,
                &mut work.uppers,
                &mut work.incr_pwq,
            );

            // increment active function and update active indicators for each pwq that needs to be updated
            for (i, &func) in fns.iter().enumerate() {
                if work.incr_pwq[i] {
                    work.active_piece_idx[i] += 1;
                    work.active_functions[i] = func[work.active_piece_idx[i]];
                    work.pwq_is_active[i] = work.active_piece_idx[i] < func.len() - 1;
                }
            }

            if let Some(active) =
                BoundedQuadratic::restrict_to_common_domain(&work.active_functions)
            {
                if !active[0].is_empty() {
                    for i in 0..active.len() {
                        out[i].add_piece_at_right(active[i]);
                    }
                }
            }
        }
        out
    }
}

// Can print out bounded quadratics
impl fmt::Display for PiecewiseQuadratic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut result = String::new();
        result.push_str("PiecewiseQuadratic:\n");
        for bq in &self.functions {
            result.push('\t');
            result.push_str(&bq.to_string());
            result.push('\n');
        }
        write!(f, "{}", result)
    }
}

impl Add for PiecewiseQuadratic {
    type Output = Self;
    fn add(self, _other: Self) -> Self {
        panic!(
            "The add trait is implemented so that we can also implement the Zero trait. \
                Use sum_pwq to sum PiecewiseQuadratics."
        )
    }
}

impl Zero for PiecewiseQuadratic {
    fn zero() -> Self {
        Self::indicator(f64::NEG_INFINITY, f64::INFINITY)
    }

    fn is_zero(&self) -> bool {
        self.len() == 1
            && self[0].lower == f64::NEG_INFINITY
            && self[0].upper == f64::INFINITY
            && relative_eq!(self[0].a, 0.)
            && relative_eq!(self[0].b, 0.)
            && relative_eq!(self[0].c, 0.)
    }
}

impl Empty for PiecewiseQuadratic {
    fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }
}

impl Len for PiecewiseQuadratic {
    fn len(&self) -> usize {
        self.functions.len()
    }
}

impl Index<usize> for PiecewiseQuadratic {
    type Output = BoundedQuadratic;
    fn index(&self, index: usize) -> &Self::Output {
        &self.functions[index]
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // SYNCHRONIZE

    #[test]
    fn test_synchronize() {
        let f1 = BoundedQuadratic::new(-1., 0., 1., 0., 0.);
        let g1 = BoundedQuadratic::new(1., 2., 0., 1., 0.);
        let f2 = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        let g2 = BoundedQuadratic::new(1., 3., 0., 1., 0.);
        let h = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., 0., 1.);

        let p1 = PiecewiseQuadratic::new(vec![f1, g1]);
        let p2 = PiecewiseQuadratic::new(vec![f2, g2]);
        let p3 = PiecewiseQuadratic::new(vec![h]);

        let mut workspace = SyncWorkspace::new(3);
        let res = PiecewiseQuadratic::synchronize(&mut workspace, &[&p1, &p2, &p3]);

        assert!(res.iter().all(|f| f.len() == 3));
        assert!(res.iter().all(|f| (f[0].lower, f[0].upper) == (-1., 0.)));
        assert!(res.iter().all(|f| (f[1].lower, f[1].upper) == (1., 1.)));
        assert!(res.iter().all(|f| (f[2].lower, f[2].upper) == (1., 2.)));
    }

    // EXTENDS LEFT AND EXTENDS RIGHT

    #[test]
    fn test_extends_left_and_right() {
        let f1 = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 1., 0., 0.);
        let g1 = BoundedQuadratic::new(0., 1., 1., 0., 1.);
        let p1 = PiecewiseQuadratic::new(vec![f1, g1]);

        assert!(p1.extends_left() && !p1.extends_right());

        let f2 = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        let g2 = BoundedQuadratic::new(2., f64::INFINITY, 0., 0., 0.);
        let p2 = PiecewiseQuadratic::new(vec![f2, g2]);

        assert!(p2.extends_right() && !p2.extends_left());
    }

    // EVAL

    #[test]
    fn test_eval() {
        let f = BoundedQuadratic::new(-1., 0., 1., 0., 0.);
        let g = BoundedQuadratic::new(1., 2., 0., 1., 0.);
        let p = PiecewiseQuadratic::new(vec![f, g]);

        // x is in the domain of one of the pieces
        assert_eq!(p.eval(-0.5), 0.25);

        // x is ood
        assert_eq!(p.eval(0.5), f64::INFINITY);
    }

    #[test]
    fn test_eval_multiple() {
        // between
        let f = BoundedQuadratic::new(-1., 0., 1., 0., 1.);
        let g = BoundedQuadratic::new_point(0., 0.);
        let h = BoundedQuadratic::new(0., 2., 0., 1., -1.);
        let p = PiecewiseQuadratic::new(vec![f, g, h]);
        assert_eq!(-1., p.eval(0.));

        let f = BoundedQuadratic::new(-1., 0., 1., 0., -1.);
        let g = BoundedQuadratic::new_point(0., 0.);
        let h = BoundedQuadratic::new(1., 2., 0., 1., 10.);
        let p = PiecewiseQuadratic::new(vec![f, g, h]);
        assert_eq!(-1., p.eval(0.));

        let f = BoundedQuadratic::new(-1., -0.5, 1., 0., 10.);
        let g = BoundedQuadratic::new_point(0., 0.);
        let h = BoundedQuadratic::new(0., 2., 0., 1., -1.);
        let p = PiecewiseQuadratic::new(vec![f, g, h]);
        assert_eq!(-1., p.eval(0.))
    }

    #[test]
    fn test_eval_big() {
        let mut fs_even = vec![];
        for i in 0..100 {
            let bq = BoundedQuadratic::new(i as f64, (i + 1) as f64, 0., 0., i as f64);
            fs_even.push(bq);
        }
        let pwq = PiecewiseQuadratic::new(fs_even);
        assert_eq!(pwq.eval(0.5), 0.);
        assert_eq!(pwq.eval(50.5), 50.);
        assert_eq!(pwq.eval(99.5), 99.);

        let mut fs_odd = vec![];
        for i in 0..101 {
            let bq = BoundedQuadratic::new(i as f64, (i + 1) as f64, 0., 0., i as f64);
            fs_odd.push(bq);
        }
        let pwq = PiecewiseQuadratic::new(fs_odd);
        assert_eq!(pwq.eval(0.5), 0.);
        assert_eq!(pwq.eval(50.5), 50.);
        assert_eq!(pwq.eval(100.5), 100.);
    }

    // ADD

    #[test]
    fn test_add_empty_intersection() {
        let f = BoundedQuadratic::new(-1., 0., 1., 0., 0.);
        let p1 = PiecewiseQuadratic::new(vec![f]);
        let g = BoundedQuadratic::new(1., 2., 0., 1., 0.);
        let p2 = PiecewiseQuadratic::new(vec![g]);
        let mut work = SyncWorkspace::new(2);
        let sum = PiecewiseQuadratic::sum_pwq(&mut work, &[&p1, &p2], false);
        assert_eq!(sum.len(), 0);
    }

    #[test]
    fn test_add() {
        let f1 = BoundedQuadratic::new(-1., 0., 1., 0., 0.);
        let g1 = BoundedQuadratic::new(1., 2., 0., 1., 0.);
        let f2 = BoundedQuadratic::new(-0.5, 1.5, 1., 0., 0.);
        let g2 = BoundedQuadratic::new(1.5, 2.5, 0., 1., 0.);
        let p1 = PiecewiseQuadratic::new(vec![f1, g1]);
        let p2 = PiecewiseQuadratic::new(vec![f2, g2]);
        let mut work = SyncWorkspace::new(2);
        let sum = PiecewiseQuadratic::sum_pwq(&mut work, &[&p1, &p2], false);
        assert_eq!(sum.len(), 3);
        let expected_first = BoundedQuadratic::new(-0.5, 0., 2., 0., 0.);
        let expected_second = BoundedQuadratic::new(1., 1.5, 1., 1., 0.);
        let expected_third = BoundedQuadratic::new(1.5, 2., 0., 2., 0.);
        assert!(sum[0].approx(&expected_first));
        assert!(sum[1].approx(&expected_second));
        assert!(sum[2].approx(&expected_third));
    }

    #[test]
    fn test_sum_many() {
        let f1 = BoundedQuadratic::new(-1., 0., 1., 0., 0.);
        let g1 = BoundedQuadratic::new(1., 2., 0., 1., 0.);
        let f2 = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        let g2 = BoundedQuadratic::new(1., 3., 0., 1., 0.);
        let h = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., 0., 1.);
        let p1 = PiecewiseQuadratic::new(vec![f1, g1]);
        let p2 = PiecewiseQuadratic::new(vec![f2, g2]);
        let p3 = PiecewiseQuadratic::new(vec![h]);
        let mut workspace = SyncWorkspace::new(3);
        let res = PiecewiseQuadratic::sum_pwq(&mut workspace, &[&p1, &p2, &p3], false).simplify();
        assert_eq!(2, res.len(), "{}", res);
        assert!(res[0].approx(&BoundedQuadratic::new(-1., 0., 2., 0., 1.)));
        assert!(res[1].approx(&BoundedQuadratic::new(1., 2., 0., 2., 1.)));
    }

    // SCALE

    #[test]
    fn test_scale_negative() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let p = PiecewiseQuadratic::new(vec![f, g]);
        let scaled = p.scale_arg(-2.);
        assert_eq!(scaled.len(), 2);
        assert!(scaled[0].approx(&g.scale_arg(-2.)));
        assert!(scaled[1].approx(&f.scale_arg(-2.)));
    }

    #[test]
    #[should_panic]
    fn test_scale_zero() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let p = PiecewiseQuadratic::new(vec![f, g]);
        p.scale_arg(0.);
    }

    // PERSPECTIVE

    #[test]
    fn test_perspective_negative() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let p = PiecewiseQuadratic::new(vec![f, g]);
        let perspective = p.perspective(-2.);
        assert_eq!(perspective.len(), 2);
        assert!(perspective[0].approx(&g.perspective(-2.)));
        assert!(perspective[1].approx(&f.perspective(-2.)));
    }

    #[test]
    #[should_panic]
    fn test_perspective_zero() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 2., 0.);
        let p = PiecewiseQuadratic::new(vec![f, g]);
        p.perspective(0.);
    }

    // SIMPLIFY

    #[test]
    fn test_simplify_empty_first() {
        let f = BoundedQuadratic::new(0., 0., 0., 0., 0.);
        let g = BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.);
        let p = PiecewiseQuadratic::new(vec![f, g]);
        let s = p.simplify();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0], g);
    }

    #[test]
    fn test_simplify_points() {
        let point = BoundedQuadratic::new(0., 0., 0., 0., 0.);
        let p = PiecewiseQuadratic::new(vec![point, point, point]);
        let s = p.simplify();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0], point);
    }

    #[test]
    fn test_simplify_diff_points() {
        let p1 = BoundedQuadratic::new(0., 0., 0., 0., 0.);
        let p2 = BoundedQuadratic::new(1., 1., 0., 0., 0.);
        let p = PiecewiseQuadratic::new(vec![p1, p2]);
        let s = p.simplify();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], p1);
        assert_eq!(s[1], p2);
    }

    #[test]
    fn test_simplify_point_curve() {
        let point = BoundedQuadratic::new(0., 0., 0., 0., 0.);
        let curve = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        let p = PiecewiseQuadratic::new(vec![point, curve]);
        let s = p.simplify();
        assert_eq!(s.len(), 1);
        assert!(s[0].approx(&curve));
    }

    #[test]
    fn test_simplify_curve_point() {
        let point = BoundedQuadratic::new(0., 0., 0., 0., 0.);
        let curve = BoundedQuadratic::new(-1., 0., 1., 0., 0.);
        let p = PiecewiseQuadratic::new(vec![curve, point]);
        let s = p.simplify();
        assert_eq!(s.len(), 1);
        assert!(s[0].approx(&curve));
    }

    #[test]
    fn test_no_simplifications() {
        let f = BoundedQuadratic::new(-1., 0., 0., -1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 1., 0.);
        let p = PiecewiseQuadratic::new(vec![f, g]);
        let s = p.simplify();
        assert_eq!(s.len(), 2);
        assert!(s[0].approx(&f));
        assert!(s[1].approx(&g));
    }

    #[test]
    fn test_simplify_combine() {
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 1., 0.);
        let h = BoundedQuadratic::new(1., 2., 0., 1., 0.);

        let p = PiecewiseQuadratic::new(vec![f, g]);
        let s = p.simplify();
        assert_eq!(s.len(), 1);
        let expected = BoundedQuadratic::new(-1., 1., 0., 1., 0.);
        assert!(s[0].approx(&expected));

        // recursive combine
        let p = PiecewiseQuadratic::new(vec![f, g, h]);
        let s = p.simplify();
        assert_eq!(s.len(), 1);
        let expected = BoundedQuadratic::new(-1., 2., 0., 1., 0.);
        assert!(s[0].approx(&expected));
    }

    #[test]
    fn test_simplify_complex() {
        let f = BoundedQuadratic::new(-1., 0., 0., -1., 0.);
        let pt = BoundedQuadratic::new(0., 0., 0., 0., 0.);
        let g = BoundedQuadratic::new(0., 1., 0., 1., 0.);
        let h = BoundedQuadratic::new(1., 2., 0., 1., 0.);
        let c = BoundedQuadratic::new(3., 5., 1., 1., 1.);
        let p = PiecewiseQuadratic::new(vec![f, pt, g, h, c]);

        let s = p.simplify();
        assert_eq!(s.len(), 3);
        assert!(s[0].approx(&f));
        assert!(s[1].approx(&BoundedQuadratic::new(0., 2., 0., 1., 0.)));
        assert!(s[2].approx(&c));
    }

    #[test]
    fn test_simplify_mutliple_points_same_x() {
        let g = BoundedQuadratic::new_point(0., 0.);
        let h = BoundedQuadratic::new_point(0., 1.);
        let f = BoundedQuadratic::new_point(0., 2.);
        let p = PiecewiseQuadratic::new(vec![g, h, f]);
        let s = p.simplify();
        // g and h should not be combined because neither includes 1.
        assert_eq!(1, s.len());
        assert!(s[0].is_point() && s[0].c == 0.);
    }

    // CONJUGATE MINIMIZER

    #[test]
    fn test_conjugate_maximizer() {
        let f = BoundedQuadratic::new(-2., -1., 0., -1., 0.);
        let g = BoundedQuadratic::new(-1., 0., 0., -0.5, 0.5);
        let h = BoundedQuadratic::new(0., 1.5, 0., 1., 0.5);
        let p = PiecewiseQuadratic::new(vec![f, g, h]);
        let conj_maximizer = p.conjugate_maximizer();
        assert_eq!(conj_maximizer.len(), 4);
        let expected_first = BoundedQuadratic::new(f64::NEG_INFINITY, -1., 0., 0., -2.);
        let expected_second = BoundedQuadratic::new(-1., -0.5, 0., 0., -1.);
        let expected_third = BoundedQuadratic::new(-0.5, 1., 0., 0., 0.);
        let expected_fourth = BoundedQuadratic::new(1., f64::INFINITY, 0., 0., 1.5);
        assert!(conj_maximizer[0].approx(&expected_first));
        assert!(conj_maximizer[1].approx(&expected_second));
        assert!(conj_maximizer[2].approx(&expected_third));
        assert!(conj_maximizer[3].approx(&expected_fourth));
    }

    #[test]
    #[should_panic(
        expected = "Can only find the conjugate maximizer of convex functions. PiecewiseQuadratic:\
                    \n\tBoundedQuadratic: f(x) = -x² + x , ∀x ∈ (-inf, 0]\n is not convex."
    )]
    fn test_conjugate_maximizer_non_convex_piece() {
        let f = BoundedQuadratic::new(f64::NEG_INFINITY, 0., -1., 1., 0.);
        let p = PiecewiseQuadratic::new(vec![f]);
        p.conjugate_maximizer();
    }

    // IS CONVEX
    #[test]
    fn test_is_convex_non_convex() {
        // test single piece that is non convex (never enters loop)
        let f = BoundedQuadratic::new(0., 1., -1., 0., 1.);
        let p = PiecewiseQuadratic::new(vec![f]);
        assert!(!p.is_convex());

        // test non continuous
        let f = BoundedQuadratic::new(0., 1., 1., 0., 1.);
        let g = BoundedQuadratic::new(5., 6., 2., 0., 1.);
        let p = PiecewiseQuadratic::new(vec![f, g]);
        assert!(!p.is_convex());

        // test derivatives out of order
        let f = BoundedQuadratic::new(-1., 0., 0., 1., 1.);
        let g = BoundedQuadratic::new(0., 1., 0., -1., 1.);
        let p = PiecewiseQuadratic::new(vec![f, g]);
        assert!(!p.is_convex());

        // test middle piece not convex
        let f = BoundedQuadratic::new(-1., 0., 0., -1., 0.);
        let g = BoundedQuadratic::new(0., 1., -1., 0., 0.);
        let h = BoundedQuadratic::new(1., 2., 0., 1., -2.);
        let p = PiecewiseQuadratic::new(vec![f, g, h]);
        assert!(!p.is_convex());
    }

    #[test]
    fn test_is_convex_convex() {
        // test indicator is convex
        let f = PiecewiseQuadratic::indicator(f64::NEG_INFINITY, f64::INFINITY);
        assert!(f.is_convex());

        // test single convex piece
        let f = BoundedQuadratic::new(0., 1., 1., 0., 1.);
        let p = PiecewiseQuadratic::new(vec![f]);
        assert!(p.is_convex());

        // test multi-piece convex
        let f1 = BoundedQuadratic::new(-1., 0., 0., -1., 0.);
        let f2 = BoundedQuadratic::new(0., 1., 0., 0., 0.);
        let f3 = BoundedQuadratic::new(1., 2., 1., -2., 1.);
        let f4 = BoundedQuadratic::new(2., 3., 0., 3., -5.);
        let p = PiecewiseQuadratic::new(vec![f1, f2, f3, f4]);
        assert!(p.is_convex());
    }
}
