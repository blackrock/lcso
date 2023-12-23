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

use crate::quadratics::utils;
use std::f64;
use std::fmt;

/// This structure represents a bounded quadratic function f(x) = `a`x^2 + `b`x + `c` for x in `domain`.
///
/// Note: Do not construct these directly. Use `BoundedQuadratic::new` so that you don't try
/// to construct invalid functions.
#[derive(Debug, Copy, Clone)]
pub struct BoundedQuadratic {
    pub lower: f64,
    pub upper: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl BoundedQuadratic {
    /// Constructs a new `BoundedQuadratic` from an interval and coefficients.
    ///
    /// # Panics
    /// * If `a`, `b`, `c` are not all finite
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below corresponds to f(x) = x^2 + x + 1 for x in [-1, 1]
    /// let quad = BoundedQuadratic::new(-1., 1., 1., 1., 1.);
    /// ```
    pub fn new(lower: f64, upper: f64, a: f64, b: f64, c: f64) -> BoundedQuadratic {
        assert!(
            a.is_finite() && b.is_finite() && c.is_finite(),
            "Quadratic, linear, and constant coefficients must be finite."
        );
        BoundedQuadratic {
            lower,
            upper,
            a,
            b,
            c,
        }
    }

    /// Constructs a new `BoundedQuadratic` whose domain is (-inf, inf).
    pub fn new_extended(a: f64, b: f64, c: f64) -> BoundedQuadratic {
        BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, a, b, c)
    }

    /// Constructs a new line from an interval and coefficients.
    ///
    /// # Panics
    /// * If `slope` and `intercept` are not both finite.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below corresponds to the line given by f(x) = x + 1 for x in [-1, 1]
    /// let line = BoundedQuadratic::new_line(-1., 1., 1., 1.);
    /// ```
    pub fn new_line(lower: f64, upper: f64, slope: f64, intercept: f64) -> BoundedQuadratic {
        assert!(slope.is_finite() && intercept.is_finite());
        BoundedQuadratic::new(lower, upper, 0., slope, intercept)
    }

    /// Constructs a new line from two points.
    ///
    /// # Panics
    /// * If the two points have approximately identical x-values.
    ///
    /// # Example
    /// ```
    /// // The function below corresponds to the line given by f(x) = x for x in (-inf, inf)
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// let line = BoundedQuadratic::new_line_from_points((0., 0.), (1., 1.));
    /// ```
    pub fn new_line_from_points(p1: (f64, f64), p2: (f64, f64)) -> BoundedQuadratic {
        // calculates an unbounded line through two points assuming that the line is not (approximately)
        // vertical.
        assert!(
            relative_ne!(p1.0, p2.0, epsilon = f64::EPSILON),
            "Can't create line with infinite slope."
        );

        let (x1, y1) = p1;
        let (x2, y2) = p2;
        let slope = (y2 - y1) / (x2 - x1);
        let intercept = y1 - slope * x1;
        BoundedQuadratic::new_line(f64::NEG_INFINITY, f64::INFINITY, slope, intercept)
    }

    /// Constructs a `BoundedQuadratic` that is just a point.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below corresponds to the point (0., 1.)
    /// let line = BoundedQuadratic::new_point(0., 1.);
    /// ```
    pub fn new_point(x: f64, y: f64) -> BoundedQuadratic {
        BoundedQuadratic::new(x, x, 0., 0., y)
    }

    /// Returns whether `self` is convex.
    pub fn is_convex(&self) -> bool {
        self.a >= 0.
    }

    /// Determines whether a `BoundedQuadratic` is affine.
    pub fn is_affine(&self) -> bool {
        relative_eq!(self.a, 0., epsilon = f64::EPSILON)
    }

    /// Determines whether the domain of a function is the empty interval.
    pub fn is_empty(&self) -> bool {
        utils::gt_ep(self.lower, self.upper)
    }

    /// Determines whether two quadratic functions are approximately the same. They are approximately
    /// the same if their coefficients and domains are approximately equal.
    pub fn approx(&self, other: &BoundedQuadratic) -> bool {
        relative_eq!(self.a, other.a, epsilon = f64::EPSILON)
            && relative_eq!(self.b, other.b, epsilon = f64::EPSILON)
            && relative_eq!(self.c, other.c, epsilon = f64::EPSILON)
            && relative_eq!(self.upper, other.upper, epsilon = f64::EPSILON)
            && relative_eq!(self.lower, other.lower, epsilon = f64::EPSILON)
    }

    /// Returns a new `BoundedQuadratic` with a domain restricted to the domain with lower bound
    /// equal to `max(self.lower, new_lower)` and upper bound equal to `min(self.upper, new_upper)`. Whether
    /// the new domain's bounds are closed or open depends on the openness or closedness of the
    /// original domain.
    ///
    /// # Panics
    /// * If the new lower bound is greater than the new upper bound.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below corresponds to the line given by f(x) = x + 1 for x in [-1, 1]
    /// let line = BoundedQuadratic::new_line(-1., 1., 1., 1.);
    /// // The restricted function is given by f(x) = x + 1 for x in [0, 0.5]
    /// let restricted_line = line.restrict_domain(0., 0.5);
    /// ```
    pub fn restrict_domain(&self, new_lower: f64, new_upper: f64) -> BoundedQuadratic {
        let l = f64::max(new_lower, self.lower);
        let u = f64::min(new_upper, self.upper);
        assert!(
            utils::approx_le(l, u),
            "New lower bound must be less than or equal to the upper bound"
        );
        BoundedQuadratic::new(l, u, self.a, self.b, self.c)
    }

    pub fn restrict_domain_in_place(&mut self, new_lower: f64, new_upper: f64) {
        let l = f64::max(new_lower, self.lower);
        let u = f64::min(new_upper, self.upper);
        assert!(
            utils::approx_le(l, u),
            "New lower bound must be less than or equal to the upper bound"
        );
        self.lower = l;
        self.upper = u;
    }

    /// Returns a new unbounded `BoundedQuadratic` representing the (unbounded) tangent line to `self`
    /// at the point `x`.
    ///
    /// # Panics
    /// * If the inferred intercept is not finite.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below is given by f(x) = x^2 + x + 1 for x in [-1, 1]
    /// let quad = BoundedQuadratic::new(-1., 1., 1., 1., 1.);
    /// // The tangent line is given by f(x) = 2x + 1 for x in (-inf, inf)
    /// let tangent = quad.get_tangent_line_at(0.5);
    /// ```
    pub fn get_tangent_line_at(&self, x: f64) -> BoundedQuadratic {
        let slope = 2. * self.a * x + self.b;
        let intercept = self.eval(x) - slope * x;
        assert!(intercept.is_finite(), "Intercept must be finite");
        BoundedQuadratic::new_line(f64::NEG_INFINITY, f64::INFINITY, slope, intercept)
    }

    /// Returns a function with the same coefficients as `self` with the domain extended to the whole line.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below is given by f(x) = x^2 + x + 1 for x in [-1, 1]
    /// let quad = BoundedQuadratic::new(-1., 1., 1., 1., 1.);
    /// // The extended function is given by f(x) = x^2 + x + 1 for x in (-inf, inf)
    /// let extended = quad.extend_domain();
    /// ```
    pub fn extend_domain(&self) -> BoundedQuadratic {
        BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, self.a, self.b, self.c)
    }

    pub fn extend_domain_in_place(&mut self) {
        self.lower = f64::NEG_INFINITY;
        self.upper = f64::INFINITY;
    }

    /// Returns a new `BoundedQuadratic` that has been _pre-scaled_ by `alpha`. That is, given f(x),
    /// a returns `BoundedQuadratic` representing g(x) = `alpha` * f(x).
    ///
    /// # Example
    /// ```
    /// // The function below is given by f(x) = x^2 + x + 1 for x in [-1, 1]
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// let quad = BoundedQuadratic::new(-1., 1., 1., 1., 1.);
    /// // The pre-scaled function is given by alpha * f(x) = alpha * (x^2 + x + 1) for x in [-1, 1]
    /// let scaled = quad.scale(2.);
    /// ```
    pub fn scale(&self, scale_factor: f64) -> BoundedQuadratic {
        BoundedQuadratic::new(
            self.lower,
            self.upper,
            self.a * scale_factor,
            self.b * scale_factor,
            self.c * scale_factor,
        )
    }

    pub fn scale_in_place(&mut self, scale_factor: f64) {
        self.a *= scale_factor;
        self.b *= scale_factor;
        self.c *= scale_factor;
    }

    fn new_scaled_interval(lower: f64, upper: f64, scale_factor: f64) -> (f64, f64) {
        let new_upper = if scale_factor < 0. {
            lower / scale_factor
        } else {
            upper / scale_factor
        };
        let new_lower = if scale_factor < 0. {
            upper / scale_factor
        } else {
            lower / scale_factor
        };
        (new_lower, new_upper)
    }

    /// Returns a new `BoundedQuadratic` whose _argument_ has been scaled by `alpha`. That is,
    /// given f(x) and `alpha`, it returns f(`alpha` * x). Note that this operation requires scaling
    /// of the domain.
    ///
    /// # Panics
    /// * If `alpha` equals 0.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below is given by f(x) = x^2 + x + 1 for x in [-1, 1]
    /// let quad = BoundedQuadratic::new(-1., 1., 1., 1., 1.);
    /// // The scaled function is given by f(2 * x) = (2 * x)^2 + 2 * x + 1 for x in
    /// // [-1 / 2, 1 / 2]
    /// let scaled = quad.scale_arg(2.);
    /// ```
    pub fn scale_arg(&self, scale_factor: f64) -> BoundedQuadratic {
        assert!(!relative_eq!(scale_factor, 0., epsilon = f64::EPSILON));
        let (lower, upper) =
            BoundedQuadratic::new_scaled_interval(self.lower, self.upper, scale_factor);
        BoundedQuadratic::new(
            lower,
            upper,
            scale_factor * scale_factor * self.a,
            scale_factor * self.b,
            self.c,
        )
    }

    pub fn scale_arg_in_place(&mut self, scale_factor: f64) {
        assert!(!relative_eq!(scale_factor, 0., epsilon = f64::EPSILON));
        let (lower, upper) =
            BoundedQuadratic::new_scaled_interval(self.lower, self.upper, scale_factor);
        self.lower = lower;
        self.upper = upper;
        self.a *= scale_factor * scale_factor;
        self.b *= scale_factor;
    }

    /// Returns a new `BoundedQuadratic` representing the perspective function of `self`. That is,
    /// given f(x) and `alpha`, return `alpha` * f(x / `alpha`). Note that this operation
    /// requires scaling of the domain.
    ///
    /// # Panics
    /// * If `alpha` equals 0.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below is given by f(x) = x^2 + x + 1 for x in [-1, 1]
    /// let quad = BoundedQuadratic::new(-1., 1., 1., 1., 1.);
    /// // The perspective function is given by 2 * f(x / 2) = 2 * (x / 2)^2 + 2 * x / 2 + 2 for x in [-2, 2]
    /// let persp = quad.perspective(2.);
    /// ```
    pub fn perspective(&self, scale_factor: f64) -> BoundedQuadratic {
        assert!(!relative_eq!(scale_factor, 0., epsilon = f64::EPSILON));
        let (lower, upper) =
            BoundedQuadratic::new_scaled_interval(self.lower, self.upper, 1. / scale_factor);
        BoundedQuadratic::new(
            lower,
            upper,
            self.a / scale_factor,
            self.b,
            self.c * scale_factor,
        )
    }

    pub fn perspective_in_place(&mut self, scale_factor: f64) {
        assert!(!relative_eq!(scale_factor, 0., epsilon = f64::EPSILON));
        let (lower, upper) =
            BoundedQuadratic::new_scaled_interval(self.lower, self.upper, 1. / scale_factor);
        self.lower = lower;
        self.upper = upper;
        self.a /= scale_factor;
        self.c *= scale_factor;
    }

    /// Returns a new `BoundedQuadratic` representing `self` shifted _left_ by `alpha`. Note that
    /// for `alpha` > 0, this is a right shift. For `alpha` < 0, this is a left shift.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function is given by f(x) = x^2 + x + 1 for x in [-1, 1]
    /// let quad = BoundedQuadratic::new(-1., 1., 1., 1., 1.);
    /// // The shifted function is given by f(x - 2) = (x - 2)^2 + (x - 2) + 1 for x in [1, 3]
    /// let shifted = quad.shift(2.);
    /// ```
    pub fn shift(&self, offset: f64) -> BoundedQuadratic {
        let lower = self.lower + offset;
        let upper = self.upper + offset;
        BoundedQuadratic::new(
            lower,
            upper,
            self.a,
            self.b - 2. * offset * self.a,
            self.a * offset * offset - self.b * offset + self.c,
        )
    }

    pub fn shift_in_place(&mut self, offset: f64) {
        self.upper += offset;
        self.lower += offset;
        let (a, b, c) = (self.a, self.b, self.c);
        self.a = a;
        self.b = b - 2. * a * offset;
        self.c = a * offset * offset - b * offset + c;
    }

    /// Returns a new `BoundedQuadratic` that is reflected over the y-axis. That is, given f(x), return
    /// f(-x).
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below is given by f(x) = x^2 + x + 1 for x in [-1, 1]
    /// let quad = BoundedQuadratic::new(-1., -1., 1., 1., 1.);
    /// // The opened function is given by f(x) = x^2 - x + 1 for x in [-1, 1]
    /// let reflected = quad.reflect_over_y();
    /// ```
    pub fn reflect_over_y(&self) -> BoundedQuadratic {
        // Given a function f(x), returns f(-x)
        BoundedQuadratic::new(-self.upper, -self.lower, self.a, -self.b, self.c)
    }

    pub fn reflect_over_y_in_place(&mut self) {
        let tmp = self.upper;
        self.upper = -self.lower;
        self.lower = -tmp;
        self.b = -self.b;
    }

    /// Returns true if an element `x` is in the domain of `self`, and false otherwise.
    pub fn domain_contains_element(&self, x: f64) -> bool {
        utils::approx_ge(x, self.lower) && utils::approx_le(x, self.upper)
    }

    /// Returns true if an element `x` is in the interior of the domain of `self`, and false otherwise.
    pub fn interior_contains_element(&self, x: f64) -> bool {
        utils::gt_ep(x, self.lower) && utils::lt_ep(x, self.upper)
    }

    /// Returns the ordered real roots (if there are any) of the function represented by `self`. If
    /// the roots are not in the domain of `self`, return NaNs. If there are no real roots, return
    /// NaNs.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below is given by f(x) = x^2 - x + 6 for x in [-5, 5]
    /// let quad = BoundedQuadratic::new(-5., 5., 1., -1., -6.);
    /// // This can be factored as f(x) = (x - 3)(x + 2), so the roots should be
    /// // -2 and 3 (in that order).
    /// let (x1, x2) = quad.find_roots();
    /// assert_eq!(x1, -2.);
    /// assert_eq!(x2, 3.);
    /// ```
    pub fn find_roots(&self) -> (f64, f64) {
        Self::_find_roots(self.lower, self.upper, self.a, self.b, self.c)
    }

    /// See `find_roots`.
    pub fn _find_roots(lower: f64, upper: f64, a: f64, b: f64, c: f64) -> (f64, f64) {
        // linear or constant
        let (mut x1, mut x2) = (f64::INFINITY, f64::INFINITY);
        if relative_eq!(a, 0., epsilon = f64::EPSILON) {
            // affine case
            if b != 0. {
                x1 = -c / b;
                // if not in domain, infs
            }
        // constant returns infs
        } else {
            let discriminant = b * b - 4. * a * c;
            if utils::approx_ge(discriminant, 0.) {
                // numerically stable quadratic formula
                let sqrt_discriminant = discriminant.sqrt();
                let x1_numerator = if utils::gt_ep(b, 0.) {
                    -b - sqrt_discriminant
                } else {
                    -b + sqrt_discriminant
                };
                x1 = x1_numerator / (2. * a);
                x2 = c / (a * x1);
            }
            // if discrim is neg, will return infs
        }
        // either both x1 and x2 will be finite or neither will be
        if utils::lt_ep(x1, lower) || utils::gt_ep(x1, upper) {
            x1 = f64::INFINITY;
        }
        if utils::lt_ep(x2, lower) || utils::gt_ep(x2, upper) {
            x2 = f64::INFINITY;
        }
        if utils::lt_ep(x1, x2) {
            (x1, x2)
        } else {
            (x2, x1)
        }
    }

    /// Determines whether or not the domain of `self` is a point.
    pub fn is_point(&self) -> bool {
        relative_eq!(self.lower, self.upper, epsilon = f64::EPSILON)
    }

    /// Determines whether `self`'s right endpoint (x, self(x)) corresponds with other's left endpoint (y, other(y)).
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// let f = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.);
    /// let g = BoundedQuadratic::new(0., f64::INFINITY, 1., 0., 0.);
    /// let h = BoundedQuadratic::new(1., 100., 1., 0., 0.);
    /// assert!(f.overlaps_continuously_with(&g));
    /// assert!(!f.overlaps_continuously_with(&h));
    /// ```
    pub fn overlaps_continuously_with(&self, other: &BoundedQuadratic) -> bool {
        relative_eq!(self.upper, other.lower, epsilon = f64::EPSILON)
            && relative_eq!(
                self.eval(self.upper),
                other.eval(other.lower),
                epsilon = f64::EPSILON
            )
    }

    /// Evaluates `self` at a point `x`. If `x` is not in the domain of `self`, return inf.
    ///
    /// # Panics
    /// * If `x` is NaN.
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below is given by f(x) = x^2 + x + 1 for x in [-1, 1]
    /// let quad = BoundedQuadratic::new(-1., 1., 1., 1., 1.);
    /// assert_eq!(quad.eval(1.), 3.);
    /// ```
    pub fn eval(&self, x: f64) -> f64 {
        assert!(!x.is_nan());
        if self.domain_contains_element(x) {
            self.a * x * x + self.b * x + self.c
        } else {
            f64::INFINITY
        }
    }

    /// Given f(x), returns f'(x).
    pub fn derivative(&self) -> BoundedQuadratic {
        BoundedQuadratic::new(self.lower, self.upper, 0., 2. * self.a, self.b)
    }

    /// Evaluates the derivative of `self` at a point `x`. OOD rule is same as `BoundedQuadratic::eval`.
    /// # Panics
    /// * If `x` is NaN.
    pub fn eval_derivative(&self, x: f64) -> f64 {
        assert!(!x.is_nan());
        if self.domain_contains_element(x) {
            2. * self.a * x + self.b
        } else {
            f64::INFINITY
        }
    }

    /// Finds the (x, f(x)) pair that minimizes `self`. If there is no minimum, returns (NaN, -inf).
    ///
    /// # Example
    /// ```
    /// use lcso::quadratics::bq::BoundedQuadratic;
    /// // The function below is given by f(x) = x^2 for x in [-1, 1]
    /// let quad = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
    /// assert_eq!(quad.minimize(), (0., 0.));
    /// ```
    pub fn minimize(&self) -> (f64, f64) {
        Self::_minimize(self.lower, self.upper, self.a, self.b, self.c)
    }

    /// See `minimize`.
    pub fn _minimize(lower: f64, upper: f64, a: f64, b: f64, c: f64) -> (f64, f64) {
        if relative_eq!(lower, upper, epsilon = f64::EPSILON) {
            return (lower, a * lower * lower + b * lower + c);
        }
        let x_min = if utils::gt_ep(a, 0.) {
            // convex quadratic: return -b/2a if it is between lower and upper
            utils::clamp(-b / (a * 2.), lower, upper)
        } else if utils::gt_ep(b, 0.) {
            // upward-sloping linear: the domain's lower bound if it is finite, otherwise nan
            if lower.is_finite() {
                lower
            } else {
                f64::NAN
            }
        } else if utils::lt_ep(b, 0.) {
            // downward sloping linear: see above logic for upward-sloping linear
            if upper.is_finite() {
                upper
            } else {
                f64::NAN
            }
        } else {
            // constant
            // in this case, we decide to return the lower of the two domain
            // endpoints if either is finite, and 0 if they're both infinite
            if lower.is_finite() {
                lower
            } else if upper.is_finite() {
                upper
            } else {
                0.
            }
        };
        // return the tuple containing the x value at which the min value occurs and the corresponding
        // y-value
        if x_min.is_finite() {
            (x_min, a * x_min * x_min + b * x_min + c)
        } else {
            (x_min, f64::NEG_INFINITY)
        }
    }

    /// Returns `true` if two `BoundedQuadratic`s have the same coefficients.
    pub fn same_coefficients(&self, g: &BoundedQuadratic) -> bool {
        relative_eq!(g.a, self.a, epsilon = f64::EPSILON)
            && relative_eq!(g.b, self.b, epsilon = f64::EPSILON)
            && relative_eq!(g.c, self.c, epsilon = f64::EPSILON)
    }

    fn intersect_domains(fs: &[BoundedQuadratic]) -> Option<(f64, f64)> {
        assert!(!fs.is_empty());
        let mut max_lower = f64::NEG_INFINITY;
        let mut min_upper = f64::INFINITY;
        for fi in fs.iter() {
            if utils::gt_ep(fi.lower, max_lower) {
                max_lower = fi.lower;
            }
            if utils::lt_ep(fi.upper, min_upper) {
                min_upper = fi.upper;
            }
            // if, at any point, the max lower bound is greater than the smallest upper bound, stop
            if utils::gt_ep(max_lower, min_upper) {
                return None;
            }
        }
        Some((max_lower, min_upper))
    }

    /// Restrict a group of bounded quadratic functions to their common domain. If there is no common
    /// domain, returns none.
    pub fn restrict_to_common_domain(fs: &[BoundedQuadratic]) -> Option<Vec<BoundedQuadratic>> {
        let (lower, upper) = Self::intersect_domains(fs)?;
        let mut restricted = Vec::with_capacity(fs.len());
        for f in fs {
            restricted.push(BoundedQuadratic::new(lower, upper, f.a, f.b, f.c));
        }
        Some(restricted)
    }

    /// Sums a list of a `BoundedQuadratic`s in a memory efficient way.
    pub fn sum_bq(summands: &[BoundedQuadratic]) -> Option<BoundedQuadratic> {
        let (lower, upper) = Self::intersect_domains(summands)?;
        let mut sa = 0.;
        let mut sb = 0.;
        let mut sc = 0.;
        for bq in summands {
            sa += bq.a;
            sb += bq.b;
            sc += bq.c;
        }
        Some(BoundedQuadratic::new(lower, upper, sa, sb, sc))
    }
}

// Can print out bounded quadratics
impl fmt::Display for BoundedQuadratic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let quad_term = if relative_eq!(self.a, 0., epsilon = f64::EPSILON) {
            "".to_string()
        } else {
            let sign = if utils::lt_ep(self.a, 0.) { "-" } else { "" };
            if relative_eq!(self.a.abs(), 1., epsilon = f64::EPSILON) {
                format!("{}x\u{b2} ", sign)
            } else {
                format!("{}{}x\u{b2} ", sign, self.a.abs())
            }
        };
        let lin_term = if relative_eq!(self.b, 0., epsilon = f64::EPSILON) {
            "".to_string()
        } else {
            let sign = if utils::lt_ep(self.b, 0.) { "-" } else { "" };
            let sign = if sign.is_empty() && !quad_term.is_empty() {
                "+"
            } else {
                sign
            };
            let sep = if quad_term.is_empty() { "" } else { " " };
            if relative_eq!(self.b.abs(), 1., epsilon = f64::EPSILON) {
                format!("{}{}x ", sign, sep)
            } else {
                format!("{}{}{}x ", sign, sep, self.b.abs())
            }
        };
        let const_term = if self.c == 0. {
            "".to_string()
        } else {
            let sign = if utils::lt_ep(self.c, 0.) { "-" } else { "" };
            let sign = if sign.is_empty() && (!quad_term.is_empty() || !lin_term.is_empty()) {
                "+"
            } else {
                sign
            };
            let sep = if quad_term.is_empty() && lin_term.is_empty() {
                ""
            } else {
                " "
            };
            format!("{}{}{}", sign, sep, self.c.abs())
        };
        let func_expr = quad_term + &lin_term + &const_term;
        let func_expr = if func_expr.is_empty() {
            "0".to_string()
        } else {
            func_expr
        };
        let left_brace = if self.lower.is_infinite() { "(" } else { "[" };
        let right_brace = if self.upper.is_infinite() { ")" } else { "]" };
        write!(
            f,
            "BoundedQuadratic: f(x) = {}, \u{2200}x \u{2208} {}{}, {}{}",
            func_expr, left_brace, self.lower, self.upper, right_brace
        )
    }
}

impl PartialEq for BoundedQuadratic {
    fn eq(&self, other: &Self) -> bool {
        if self.is_point()
            && other.is_point()
            && relative_eq!(self.lower, other.lower, epsilon = f64::EPSILON)
        {
            relative_eq!(
                self.eval(self.lower),
                other.eval(other.lower),
                epsilon = f64::EPSILON
            )
        } else {
            self.approx(other)
        }
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    // RESTRICT DOMAIN
    #[test]
    fn test_restrict_to_common_domain() {
        let f = BoundedQuadratic::new(1., 3., 1., 1., 1.);
        let g = BoundedQuadratic::new(-1., 2., 1., 1., 1.);
        let h = BoundedQuadratic::new(-3., 3., 1., 1., 1.);
        let restricted = BoundedQuadratic::restrict_to_common_domain(&vec![f, g, h]).unwrap();
        assert_eq!(3, restricted.len());
        let (lower, upper) = (restricted[0].lower, restricted[0].upper);
        assert!(restricted
            .iter()
            .all(|f| f.lower == lower && f.upper == upper));
        assert_eq!((1., 2.), (lower, upper), "{:?}", (lower, upper));
    }

    // IS CONVEX

    #[test]
    fn test_is_convex() {
        let convex = BoundedQuadratic::new_extended(1., 0., 0.);
        let non_convex = BoundedQuadratic::new_extended(-1., 0., 0.);
        let affine = BoundedQuadratic::new_extended(0., 1., 0.);
        assert!(convex.is_convex());
        assert!(affine.is_convex());
        assert!(!non_convex.is_convex());
    }

    // RESTRICT DOMAIN

    #[test]
    fn test_restrict_domain() {
        let original = BoundedQuadratic::new(0., 2., 1., 0., 0.);
        let expected_restricted = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        assert_eq!(original.restrict_domain(-1., 3.), original); // restricting to a (same-width or) wider domain does nothing
        assert_eq!(original.restrict_domain(0., 1.), expected_restricted); // restricting works as expected
    }

    #[test]
    #[should_panic(expected = "New lower bound must be less than or equal to the upper bound")]
    fn test_restrict_domain_bad_inputs() {
        let original = BoundedQuadratic::new(0., 2., 1., 0., 0.);
        original.restrict_domain(100., -100.);
    }

    // TANGENT LINE AT A POIN
    #[test]
    fn test_tangent_line_at_point() {
        // f(x) = x^2
        let quad = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        // Slope at x should be f'(x) = 2x. At x = 0.5, should be 2 * 0.5.
        // Intercept should be f(x) - slope * x = f(0.5) - 1 * 0.5 = 0.25 - 1 * 0.5
        let expected_tangent_line = BoundedQuadratic::new_extended(0., 2. * 0.5, 0.25 - 1. * 0.5);
        assert_eq!(quad.get_tangent_line_at(0.5), expected_tangent_line);
    }

    #[test]
    fn test_tangent_line_at_point_horizontal() {
        // f(x) = x^2
        let quad = BoundedQuadratic::new(0., 1., 1., 0., 0.);
        // Tangent line at x = 0 is horizontal (y = 0)
        let expected_tangent_line = BoundedQuadratic::new_extended(0., 0., 0.);
        assert_eq!(quad.get_tangent_line_at(0.), expected_tangent_line);
    }

    #[test]
    fn test_tangent_line_to_line() {
        let line = BoundedQuadratic::new(0., 1., 0., 1., 1.);
        let expected_tangent_line = line.extend_domain();
        assert_eq!(line.get_tangent_line_at(0.5), expected_tangent_line);
        assert_eq!(line.get_tangent_line_at(0.3), expected_tangent_line);
    }

    // REFLECT OVER Y-AXIS

    #[test]
    fn test_reflect_over_y_axis() {
        let quad = BoundedQuadratic::new(1., 2., 1., 1., 1.);
        let expected_reflection = BoundedQuadratic::new(-2., -1., 1., -1., 1.);
        assert_eq!(quad.reflect_over_y(), expected_reflection);
        assert_eq!(quad.reflect_over_y().reflect_over_y(), quad);
    }

    // DOMAIN IS POINT

    #[test]
    fn test_is_point() {
        assert!(BoundedQuadratic::new(1., 1., 1., 1., 1.).is_point());
        assert!(!BoundedQuadratic::new(1., 2., 1., 1., 1.).is_point());
    }

    // FIND ROOTS

    #[test]
    fn test_find_roots_linear() {
        let line = BoundedQuadratic::new(0., 2., 0., 1., -1.);
        let (x1, x2) = line.find_roots();
        assert!(!x2.is_finite());
        assert_eq!(x1, 1.);
    }

    #[test]
    fn test_find_roots_linear_ood() {
        let line = BoundedQuadratic::new(2., 3., 0., 1., -1.);
        let (x1, x2) = line.find_roots();
        assert!(!x1.is_finite() && !x2.is_finite());
    }

    #[test]
    fn test_find_roots_constant() {
        let constant = BoundedQuadratic::new(0., 1., 0., 0., 1.);
        let (x1, x2) = constant.find_roots();
        assert!(!x1.is_finite() && !x2.is_finite());
    }

    #[test]
    fn test_find_roots_quad_no_real_solutions() {
        let no_real_solutions = BoundedQuadratic::new(-1., 1., 1., 0., 1.);
        let (x1, x2) = no_real_solutions.find_roots();
        assert!(!x1.is_finite() && !x2.is_finite());
    }

    #[test]
    fn test_find_roots_quad() {
        let quad = BoundedQuadratic::new(-1., 5., 1., -5., 6.);
        let (x1, x2) = quad.find_roots();
        assert_eq!(x1, 2.);
        assert_eq!(x2, 3.);
    }

    #[test]
    fn test_find_roots_multiple_same() {
        let quad = BoundedQuadratic::new(-1., 3., 1., -4., 4.);
        let (x1, x2) = quad.find_roots();
        assert_eq!(x1, 2.);
        assert_eq!(x2, 2.);
    }

    #[test]
    fn test_find_roots_one_in_one_out() {
        let quad = BoundedQuadratic::new(2.5, 5., 1., -5., 6.);
        let extended = quad.extend_domain();
        let (quad_x1, quad_x2) = quad.find_roots();
        let (ext_x1, ext_x2) = extended.find_roots();
        assert_eq!(quad_x1, 3.);
        assert!(!quad_x2.is_finite());
        assert_eq!(ext_x1, 2.);
        assert_eq!(ext_x2, 3.);
    }

    // SCALE, PERSPECTIVE

    #[test]
    fn test_scale_positive() {
        let quad = BoundedQuadratic::new(2.5, 5., 1., -5., 6.);
        let scaled = quad.scale_arg(2.);
        assert_eq!(scaled.lower, 1.25);
        assert_eq!(scaled.upper, 2.5);
        assert_eq!(scaled.a, 4.);
        assert_eq!(scaled.b, -10.);
        assert_eq!(scaled.c, 6.);
    }

    #[test]
    fn test_scale_negative() {
        let quad = BoundedQuadratic::new(2.5, 5., 1., -5., 6.);
        let scaled = quad.scale_arg(-2.);
        assert_eq!(scaled.upper, -1.25);
        assert_eq!(scaled.lower, -2.5);
        assert_eq!(scaled.a, 4.);
        assert_eq!(scaled.b, 10.);
        assert_eq!(scaled.c, 6.);
    }

    #[test]
    #[should_panic]
    fn test_scale_zero() {
        let quad = BoundedQuadratic::new(2.5, 5., 1., -5., 6.);
        quad.scale_arg(0.);
    }

    #[test]
    fn test_perspective_positive() {
        let quad = BoundedQuadratic::new(2.5, 5., 1., -5., 6.);
        let perspective = quad.perspective(2.);
        assert_eq!(perspective.lower, 5.);
        assert_eq!(perspective.upper, 10.);
        assert_eq!(perspective.a, 0.5);
        assert_eq!(perspective.b, -5.);
        assert_eq!(perspective.c, 12.);
    }

    #[test]
    fn test_perspective_negative() {
        let quad = BoundedQuadratic::new(2.5, 5., 1., -5., 6.);
        let perspective = quad.perspective(-2.);
        assert_eq!(perspective.lower, -10.);
        assert_eq!(perspective.upper, -5.);
        assert_eq!(perspective.a, -0.5);
        assert_eq!(perspective.b, -5.);
        assert_eq!(perspective.c, -12.);
    }

    #[test]
    #[should_panic]
    fn test_perspective_zero() {
        let quad = BoundedQuadratic::new(2.5, 5., 1., -5., 6.);
        quad.perspective(0.);
    }

    // SHIFT

    #[test]
    fn test_shift() {
        let quad = BoundedQuadratic::new(2.5, 5., 1., -5., 6.);
        let shifted = quad.shift(2.);
        assert_eq!(shifted.lower, 4.5);
        assert_eq!(shifted.upper, 7.);
        assert_eq!(shifted.a, quad.a);
        assert_eq!(shifted.b, quad.b - 2. * 2. * 1.);
        assert_eq!(shifted.c, quad.a * 2. * 2. - 2. * quad.b + quad.c);
    }

    // EVAL AND EVAL DERIVATIVE

    #[test]
    fn test_eval_in_bounds() {
        let quad = BoundedQuadratic::new(0., 3., 1., 1., 1.);
        assert_eq!(quad.eval(1.), 3.);
    }

    #[test]
    fn test_eval_ood() {
        let quad = BoundedQuadratic::new(0., 3., 1., 1., 1.);
        assert!(!quad.eval(-1.).is_finite());
    }

    #[test]
    fn test_eval_derivative_in_bounds() {
        let quad = BoundedQuadratic::new(0., 3., 1., 1., 1.);
        assert_eq!(quad.eval_derivative(2.), 5.);
    }

    #[test]
    fn test_eval_derivative_ood() {
        let quad = BoundedQuadratic::new(0., 3., 1., 1., 1.);
        assert!(!quad.eval_derivative(-1.).is_finite());
    }

    // MINIMIZE

    #[test]
    fn test_minimize_in_bounds() {
        let quad = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        assert_eq!(quad.minimize(), (0., 0.));
    }

    #[test]
    fn test_minimize_end_point() {
        let quad = BoundedQuadratic::new(-1., 0.5, 1., -2., 1.);
        assert_eq!(quad.minimize(), (0.5, 0.25));
    }

    #[test]
    fn test_minimize_linear_up() {
        let quad = BoundedQuadratic::new(-1., 1., 0., 1., 0.);
        assert_eq!(quad.minimize(), (-1., -1.));
    }

    #[test]
    fn test_minimize_linear_down() {
        let quad = BoundedQuadratic::new(-1., 1., 0., -1., 0.);
        assert_eq!(quad.minimize(), (1., -1.));
    }

    #[test]
    fn test_minimize_constant_finite_bound() {
        let quad = BoundedQuadratic::new(-1., 0.5, 0., 0., 1.);
        assert_eq!(quad.minimize(), (-1., 1.));
    }

    #[test]
    fn test_minimize_constant_infinite_bound() {
        let quad = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., 0., 1.);
        assert_eq!(quad.minimize(), (0., 1.));
    }

    #[test]
    fn test_minimize_linear_infinite_lower() {
        let quad = BoundedQuadratic::new(f64::NEG_INFINITY, f64::INFINITY, 0., 1., 0.);
        let (x1, x2) = quad.minimize();
        assert!(x1.is_nan());
        assert_eq!(x2, f64::NEG_INFINITY);
    }
}
