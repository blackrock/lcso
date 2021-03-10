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

use crate::quadratics;
use crate::quadratics::bq::BoundedQuadratic;
use crate::quadratics::pwq::PiecewiseQuadratic;
use crate::quadratics::utils;
use len_trait::Len;
use std::f64;

/// Computes `(xf, xg, slope, intercept)` between an interior point of `f`'s domain
/// and an interior point of `g`'s domain, storing the result in `env`. We assume
/// `f` and `g` are strongly convex.
///
/// To find `(xf, xg, slope, intercept)` if they exist, we solve the system of four
/// equations in four variables:
/// `f(xf) = h(xf)`, (h is tangent at x)
/// `f'(xf) = h'(xf)`, (h has the same slope as f at x)
/// `g(xg) = h(xg)`, (h is tangent at z)
/// `g'(xg) = h'(xg)`, (h has the same slope as g at z)
/// These can be turned into a single quadratic equation in a single variable `xf` (or `xg`).
/// Solving the equation will yield two candidate values for `xf`, which we then use to calculate the
/// other three values in turn.
fn envelope_mm(
    f: BoundedQuadratic,
    g: BoundedQuadratic,
    env: &mut Vec<BoundedQuadratic>,
) -> (bool, bool) {
    let (a1, b1, c1) = (f.a, f.b, f.c);
    let (a2, b2, c2) = (g.a, g.b, g.c);
    if a1 > 0. && a2 > 0. {
        let a = (a1 * a1) / a2 - a1;
        let b = a1 * (b1 - b2) / a2;
        let c = c1 - c2 + (b1 - b2) * (b1 - b2) / (4. * a2);
        let (x1, x2) = BoundedQuadratic::_find_roots(f64::NEG_INFINITY, f64::INFINITY, a, b, c);
        for &candidate_xf in &[x1, x2] {
            if candidate_xf.is_finite() {
                let cand_xg = (2. * a1 * candidate_xf + b1 - b2) / (2. * a2);
                // check that xf is in the domain of f and xg is in the domain of g
                if f.domain_contains_element(candidate_xf) && g.domain_contains_element(cand_xg) {
                    let slope = 2. * a1 * candidate_xf + b1;
                    let intercept = -a1 * candidate_xf * candidate_xf + c1;
                    let left = f.restrict_domain(f.lower, candidate_xf);
                    let mid = BoundedQuadratic::new_extended(0., slope, intercept);
                    let right = g.restrict_domain(cand_xg, g.upper);
                    if !left.is_point() {
                        env.push(left);
                    }
                    env.push(mid.restrict_domain(candidate_xf, cand_xg));
                    if !right.is_point() {
                        env.push(right);
                    }
                    return (false, false);
                }
            }
        }
    }
    (false, false)
}

/// See comment in envelope_mm, but this time solve the system
/// without the equation `g'(xg) = h'(xg)`. Also note that in this case
/// because `xg` is an endpoint, `xg` and `g(xg)` are both known. Instead of
/// 4 equations in 4 vars, we solve an equation with three equations in three variables.
fn envelope_point_quad(point: (f64, f64), f: BoundedQuadratic) -> f64 {
    let (x, y) = point;

    let a = f.a;
    let b = -2. * f.a * x;
    let c = -f.c + y - f.b * x;
    let (x1, _) = BoundedQuadratic::_find_roots(f64::NEG_INFINITY, f64::INFINITY, a, b, c);
    x1
}

/// Computes `(xf, xg, slope, intercept)` between an interior point of `f`'s domain
/// and an endpoint of `g`'s domain, storing the result in `env`. See the docstrings for
/// `envelope_point_quad` and `envelope_mm`.
fn envelope_me(
    f: BoundedQuadratic,
    g: BoundedQuadratic,
    env: &mut Vec<BoundedQuadratic>,
) -> (bool, bool) {
    // mid to finite lower
    let xg = g.lower;
    let xf = envelope_point_quad((xg, g.eval(xg)), f);
    if xf.is_finite() && f.domain_contains_element(xf) {
        let h = f.get_tangent_line_at(xf);
        let left = f.restrict_domain(f64::NEG_INFINITY, xf);
        let mid = h.restrict_domain(xf, xg);
        let right = g.restrict_domain(xg, f64::INFINITY);
        // check that derivative of the middle function is less than that of rightmost piece
        if utils::approx_le(mid.b, right.eval_derivative(xg)) {
            if !left.is_point() {
                env.push(left);
            }
            env.push(mid);
            if !right.is_point() {
                env.push(right);
            }
            return (false, false);
        }
    }

    let xg = g.upper;
    if xg.is_finite() {
        // mid to finite upper
        let xf = envelope_point_quad((xg, g.eval(xg)), f);
        if xf.is_finite() && f.domain_contains_element(xf) {
            let h = f.get_tangent_line_at(xf);
            // we know that h intersects g at (xg, g(xg)). therefore, we need only check
            // that the derivative of g at its upper bound is smaller than that of h (this
            // implies that h lower bounds g)
            if utils::approx_le(g.eval_derivative(g.upper), h.b) {
                env.push(f.restrict_domain(f64::NEG_INFINITY, xf));
                env.push(h.restrict_domain(xf, xg));
                return (false, true);
            }
        }
    } else if g.is_affine() {
        // mid to infinite upper
        let xf = (g.b - f.b) / (2. * f.a);
        if f.domain_contains_element(xf) {
            let h = BoundedQuadratic::new_extended(0., g.b, f.extend_domain().eval(xf) - g.b * xf);
            // by construction, h has the same slope as g. therefore, it is sufficient to check
            // that h lower bounds g at a single point in the domain of g
            if utils::approx_le(h.eval(g.lower), g.eval(g.lower)) {
                env.push(f.restrict_domain(f64::NEG_INFINITY, xf));
                env.push(h.restrict_domain(xf, f64::INFINITY));
                return (false, false);
            }
        }
    }

    (false, false)
}

/// Computes `(xf, xg, slope, intercept)` between an endpoint point of `f`'s domain
/// and an endpoint of `g`'s domain, storing the result in `env`. At a high level, this is done
/// by trying out all lines that connect one endpoint of `f`'s domain to an endpoint of `g`'s domain.
/// (There is some subtlety about how infinite bounds are handled.)
///
/// The `do_symm` parameter is used to avoid checking cases that have already been checked. For more
/// information on how symmetry plays a role, see the docstring for `envelope_bq_bq`.
fn envelope_ee(
    f: BoundedQuadratic,
    g: BoundedQuadratic,
    env: &mut Vec<BoundedQuadratic>,
    do_symm: bool,
) -> (bool, bool) {
    if do_symm {
        // upper to lower, no gap
        if relative_eq!(f.upper, g.lower, epsilon = f64::EPSILON) {
            if f.is_point() && g.is_point() {
                // if both functions are points, take the lower
                if utils::approx_ge(f.c, g.c) {
                    env.push(g);
                } else {
                    env.push(f);
                }
                return (true, true);
            } else if f.is_point() && utils::approx_ge(f.eval(f.upper), g.eval(g.lower)) {
                // if f is a point and it overlaps with g, take g
                env.push(g);
                return (true, g.is_point());
            } else if g.is_point() && utils::approx_ge(g.eval(g.lower), f.eval(f.upper)) {
                // if g is a point and it overlaps with g, take f
                env.push(f);
                return (f.is_point(), true);
            } else if relative_eq!(f.eval(f.upper), g.eval(g.lower), epsilon = f64::EPSILON)
                && utils::approx_le(f.eval_derivative(f.upper), g.eval_derivative(g.lower))
            {
                // if f and g meet at their upper and lower endpoints, respectively, and
                // f'(f.upper) <= g'(g.lower), we already have the envelope, so take both f and g
                env.push(f);
                env.push(g);
                return (f.is_point(), g.is_point());
            }
        } else {
            // upper to lower with gap
            let h = BoundedQuadratic::new_line_from_points(
                (f.upper, f.eval(f.upper)),
                (g.lower, g.eval(g.lower)),
            );
            // check that derivative are in increasing order (f <= h, h <= g)
            if utils::approx_le(f.eval_derivative(f.upper), h.b)
                && utils::approx_le(h.b, g.eval_derivative(g.lower))
            {
                if !f.is_point() {
                    env.push(f);
                }
                env.push(h.restrict_domain(f.upper, g.lower));
                if !g.is_point() {
                    env.push(g);
                }
                return (f.is_point(), g.is_point());
            }
        }

        // finite lower to finite upper
        if f.lower.is_finite() && g.upper.is_finite() {
            let h = BoundedQuadratic::new_line_from_points(
                (f.lower, f.eval(f.lower)),
                (g.upper, g.eval(g.upper)),
            );
            // in this case, we need to verify that h is a lower bound for both f and g. to verify
            // this, we make sure that h has a smaller slope than f at its left end and a larger slope
            // than g at its right end
            if utils::approx_le(h.b, f.eval_derivative(f.lower))
                && utils::approx_le(g.eval_derivative(g.upper), h.b)
            {
                env.push(h.restrict_domain(f.lower, g.upper));
                return (true, true);
            }
        }
    }

    // finite lower to infinite upper
    if f.lower.is_finite() && g.upper.is_infinite() && g.is_affine() {
        let h = BoundedQuadratic::new_extended(0., g.b, f.eval(f.lower) - g.b * f.lower);
        // because h connects to the lower of endpoint of f, we need its derivative to be smaller
        // than that of f
        if utils::approx_le(h.b, f.eval_derivative(f.lower)) {
            // TODO: correct?
            env.push(h.restrict_domain(f.lower, g.upper));
            return (true, false);
        }
    }

    // upper to finite upper
    if g.upper.is_finite() && !relative_eq!(f.upper, g.upper, epsilon = f64::EPSILON) {
        let h = BoundedQuadratic::new_line_from_points(
            (f.upper, f.eval(f.upper)),
            (g.upper, g.eval(g.upper)),
        );
        // because h connects two upper endpoints, the slope at each of the endpoints it connects
        // must be larger than f and g at those endpoints
        if utils::approx_le(f.eval_derivative(f.upper), h.b)
            && utils::approx_le(g.eval_derivative(g.upper), h.b)
        {
            env.push(f);
            env.push(h.restrict_domain(f.upper, g.upper));
            return (f.is_point(), true);
        }
    }

    // upper to infinite upper
    if g.upper.is_infinite() && g.is_affine() {
        let h = BoundedQuadratic::new_extended(0., g.b, f.eval(f.upper) - g.b * f.upper);
        // by construction, h has the same slope as g. therefore, it is sufficient to check
        // that h lower bounds g at a single point in the domain of g
        if utils::approx_le(h.eval(g.lower), g.eval(g.lower)) {
            env.push(f);
            env.push(h.restrict_domain(f.upper, f64::INFINITY));
            return (f.is_point(), false);
        }
    }
    (false, false)
}

/// Uses guess-and-check algorithm to find the envelope of a two-piece piecewise quadratic function.
/// This function makes use of symmetry to reduce the number of explicit cases that need to be checked.
///
/// For example, to check the endpoint-midpoint case for functions `f` and `g`, it suffices to check
/// the midpoint-endpoint case for `g.reflect_over_y()` and `f.reflect_over_y()`. If a result comes back,
/// its reverse is the endpoint-midpoint result.
fn envelope_bq_bq(
    f: BoundedQuadratic,
    g: BoundedQuadratic,
    env: &mut Vec<BoundedQuadratic>,
) -> bool {
    // this function computes the convex envelope of two bounded quadratic functions. it returns
    // a tuple containing (left endpoint, right endpoint, slope, intercept). the envelope consists
    // of some piece of f, a line h, and some piece of g. note that the pieces of f and g might be
    // points. the line h may connect two points on the interiors of f's and g's domains (envelope_mm).
    // it can also connect a midpoint of f (or g) to an endoint of g (or f) (envelope_me). finally, h might
    // connect one of the interior points of f's (or g's) domain to one of the endpoints of g's (or
    // f's) domain (envelope_ee).

    assert!(utils::approx_le(f.upper, g.lower));

    let (of, og) = (f.to_owned(), g.to_owned());
    let mut f = f.to_owned();
    let mut g = g.to_owned();

    let (intersection_at_left, _) = envelope_ee(f, g, env, true);
    if !env.is_empty() {
        return intersection_at_left;
    }

    envelope_mm(f, g, env);
    if !env.is_empty() {
        return false;
    }

    envelope_me(f, g, env);
    if !env.is_empty() {
        return false;
    }

    // flip
    f.reflect_over_y_in_place();
    g.reflect_over_y_in_place();

    let (_, intersection_at_right) = envelope_me(g, f, env);
    if !env.is_empty() {
        env.reverse();
        env.iter_mut()
            .for_each(quadratics::bq::BoundedQuadratic::reflect_over_y_in_place);
        return intersection_at_right;
    }

    // println!("end end rev...");
    let (_, intersection_at_right) = envelope_ee(g, f, env, false);
    if !env.is_empty() {
        env.reverse();
        env.iter_mut()
            .for_each(quadratics::bq::BoundedQuadratic::reflect_over_y_in_place);
        return intersection_at_right;
    }

    panic!(format!(
        "One of the above should have returned! Could not construct envelope under:\n{}\n{}",
        of, og
    ));
}

/// Constructs the envelope of a convex `PiecewiseQuadratic` `f_pwq` and a `BoundedQuadratic`
/// `f_bq`. This function is used to incrementally update a partially constructed envelope (see use
/// in `envelope`).
fn add_piece(
    f_pwq: &mut Vec<BoundedQuadratic>,
    f_bq: BoundedQuadratic,
    env: &mut Vec<BoundedQuadratic>,
) {
    // computes the envelope under a piecewise quadratic and a bounded quadratic (to the right of all the pieces).
    // if the pwq is empty, return a piecewise quadratic with the one piece
    if f_pwq.is_empty() {
        f_pwq.push(f_bq);
        return;
    }

    // the function to add must be to the left of the pieces we want to add it to
    assert!(
        f_bq.lower >= f_pwq.last().unwrap().upper,
        "f_bq must be to the right (interval-wise) of f_pwq"
    );

    let mut intersection_at_left = true;

    // on each iteration, compare the left endpoint of the rightmost piece to the left part
    // of the envelope of that piece and the piece we want to add (f_bq). if the endpoint
    // of the envelope is (approximately) equal to the left endpoint of the rightmost piece's
    // domain, we check the next piece to the right. otherwise, quit
    while !f_pwq.is_empty() && intersection_at_left {
        env.clear();
        let rightmost_piece = f_pwq.pop().unwrap();
        intersection_at_left = envelope_bq_bq(rightmost_piece, f_bq, env);
    }
    f_pwq.append(env);
}

/// Computes the greatest convex lower bound of a `PiecewiseQuadratic` `f`.
///
/// # Example
/// ```
/// use std::f64;
/// use len_trait::len::Len;
/// use lcso::quadratics::bq::BoundedQuadratic;
/// use lcso::quadratics::pwq::PiecewiseQuadratic;
/// use lcso::quadratics::envelope::envelope;
/// let line = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., 1., 0.);
/// let constant = BoundedQuadratic::new(0., 3., 0., 0., 0.);
/// let env = envelope(&PiecewiseQuadratic::new(vec![line, constant]));
/// assert_eq!(env.len(), 1);
/// assert_eq!(env[0].b, line.b);
/// assert_ne!(env[0].c, line.c);
/// ```
pub fn envelope(f: &PiecewiseQuadratic) -> PiecewiseQuadratic {
    if f.is_convex() {
        return f.to_owned();
    }
    // computes the envelope of a piecewise quadratic function by adding the bounded quadratic pieces
    // one-by-one
    let mut work = Vec::<BoundedQuadratic>::with_capacity(2 * f.len());
    let mut env = Vec::<BoundedQuadratic>::with_capacity(3);
    for &fi in &f.simplify().functions {
        add_piece(&mut work, fi, &mut env);
    }
    PiecewiseQuadratic::new(work).simplify()
}

#[cfg(test)]
mod tests {

    use super::*;
    use len_trait::Empty;

    // ENVELOPE EE (ENDPOINT -> ENDPOINT)
    #[test]
    fn test_ee_upper_to_inf_upper() {
        let quad1 = BoundedQuadratic::new(1., 3., 0., 1., 1.);
        let quad2 = BoundedQuadratic::new(4., f64::INFINITY, 0., 2., 10.);
        let mut env = Vec::new();
        envelope_ee(quad1, quad2, &mut env, true);
        assert_eq!(2, env.len());
        let (xf, xg, slope, intercept) = (env[1].lower, env[1].upper, env[1].b, env[1].c);
        assert_eq!(xf, 3.);
        assert_eq!(xg, f64::INFINITY);
        assert_eq!(slope, 2.);
        assert_eq!(intercept, -2.);
    }

    #[test]
    fn test_ee_lower_to_inf_upper() {
        let quad1 = BoundedQuadratic::new(-1., 0., 0., 1., 1.);
        let quad2 = BoundedQuadratic::new(1., f64::INFINITY, 0., -1., 1.);
        let mut env = Vec::new();
        envelope_ee(quad1, quad2, &mut env, true);
        assert_eq!(1, env.len());
        let (xf, xg, slope, intercept) = (env[0].lower, env[0].upper, env[0].b, env[0].c);
        assert_eq!(xf, -1.);
        assert_eq!(xg, f64::INFINITY);
        assert_eq!(slope, -1.);
        assert_eq!(intercept, -1.);
    }

    #[test]
    fn test_ee_reverse_inf_lower_to_upper() {
        let quad1 = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., 1., 1.);
        let quad2 = BoundedQuadratic::new(1., 2., 0., -1., 2.);
        let mut env = Vec::new();
        // shows that infinite lower to finite upper is captured by reversing
        envelope_ee(
            quad2.reflect_over_y(),
            quad1.reflect_over_y(),
            &mut env,
            true,
        );
        assert_eq!(1, env.len());
        let (xf, xg, slope, intercept) = (env[0].lower, env[0].upper, env[0].b, env[0].c);
        let (rev_xf, rev_xg, rev_slope) = (-xg, -xf, -slope);
        assert_eq!(rev_xf, f64::NEG_INFINITY);
        assert_eq!(rev_xg, 2.);
        assert_eq!(rev_slope, 1.);
        assert_eq!(intercept, -2.);
    }

    #[test]
    fn test_ee_reverse_inf_lower_to_lower() {
        let quad1 = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., 1., 1.);
        let quad2 = BoundedQuadratic::new(2., 4., 0., 2., -4.);
        let mut env = Vec::new();
        // shows that infinite lower to finite upper is captured by reversing
        envelope_ee(
            quad2.reflect_over_y(),
            quad1.reflect_over_y(),
            &mut env,
            true,
        );
        assert_eq!(2, env.len(), "{:?}", env);
        let (xf, xg, slope, intercept) = (env[1].lower, env[1].upper, env[1].b, env[1].c);
        let (rev_xf, rev_xg, rev_slope) = (-xg, -xf, -slope);
        assert_eq!(rev_xf, f64::NEG_INFINITY);
        assert_eq!(rev_xg, 2.);
        assert_eq!(rev_slope, 1.);
        assert_eq!(intercept, -2.);
    }

    #[test]
    fn test_ee_upper_to_upper() {
        let quad1 = BoundedQuadratic::new(1., 3., 0., 1., 1.);
        let quad2 = BoundedQuadratic::new(4., 6., 0., 2., 10.);
        let mut env = Vec::new();
        envelope_ee(quad1, quad2, &mut env, true);
        assert_eq!(2, env.len());
        let (xf, xg, slope, intercept) = (env[1].lower, env[1].upper, env[1].b, env[1].c);
        let expected_line =
            BoundedQuadratic::new_line_from_points((3., quad1.eval(3.)), (6., quad2.eval(6.)));
        assert_eq!(xf, 3.);
        assert_eq!(xg, 6.);
        assert_eq!(slope, expected_line.b);
        assert_eq!(intercept, expected_line.c);
    }

    #[test]
    fn test_ee_reverse_lower_to_lower() {
        let quad2 = BoundedQuadratic::new(1., 3., 0., 1., -0.5);
        let quad1 = BoundedQuadratic::new(-1., 0., 0., 1., 1.);
        let mut env = Vec::new();
        envelope_ee(
            quad2.reflect_over_y(),
            quad1.reflect_over_y(),
            &mut env,
            true,
        );
        assert_eq!(2, env.len());
        let (xf, xg, slope, intercept) = (env[1].lower, env[1].upper, env[1].b, env[1].c);
        let expected_line =
            BoundedQuadratic::new_line_from_points((-1., quad1.eval(-1.)), (1., quad2.eval(1.)));
        assert_eq!(-xg, -1.);
        assert_eq!(-xf, 1.);
        assert_eq!(-slope, expected_line.b);
        assert_eq!(intercept, expected_line.c);
    }

    #[test]
    fn test_ee_lower_to_upper() {
        let quad1 = BoundedQuadratic::new(-1., 0., 0., 1., 1.);
        let quad2 = BoundedQuadratic::new(1., 3., 0., -1., 1.);
        let mut env = Vec::new();
        envelope_ee(quad1, quad2, &mut env, true);
        assert_eq!(1, env.len());
        let expected_line =
            BoundedQuadratic::new_line_from_points((-1., quad1.eval(-1.)), (3., quad2.eval(3.)));
        assert_eq!(env[0].lower, -1.);
        assert_eq!(env[0].upper, 3.);
        assert_eq!(env[0].b, expected_line.b);
        assert_eq!(env[0].c, expected_line.c);
    }

    #[test]
    fn test_ee_upper_to_lower_no_gap() {
        let quad1 = BoundedQuadratic::new(-1., 0., 0., -1., 0.);
        let quad2 = BoundedQuadratic::new(0., 1., 0., 1., 0.);
        let mut env = Vec::new();
        envelope_ee(quad1, quad2, &mut env, true);
        assert_eq!(2, env.len());
        assert!(env[0].approx(&quad1));
        assert!(env[1].approx(&quad2));
    }

    #[test]
    fn test_ee_upper_to_lower_with_gap() {
        let quad1 = BoundedQuadratic::new(-1., 0., 0., -1., 1.);
        let quad2 = BoundedQuadratic::new(1., 3., 0., 1., 0.5);
        let mut env = Vec::new();
        envelope_ee(
            quad2.reflect_over_y(),
            quad1.reflect_over_y(),
            &mut env,
            true,
        );
        assert_eq!(3, env.len());
        let (xf, xg, slope, intercept) = (-env[1].upper, -env[1].lower, -env[1].b, env[1].c);
        let expected_line =
            BoundedQuadratic::new_line_from_points((0., quad1.eval(0.)), (1., quad2.eval(1.)));
        assert_eq!(xf, 0.);
        assert_eq!(xg, 1.);
        assert_eq!(slope, expected_line.b);
        assert_eq!(intercept, expected_line.c);
    }

    #[test]
    fn test_ee_no_env_between_endpts() {
        let quad1 = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        let quad2 = BoundedQuadratic::new(2., 4., 1., -6., 9.);
        let mut env = Vec::new();
        envelope_ee(quad1, quad2, &mut env, true);
        assert_eq!(0, env.len());
        envelope_ee(
            quad2.reflect_over_y(),
            quad1.reflect_over_y(),
            &mut env,
            true,
        );
        assert_eq!(0, env.len());
    }

    // ENVELOPE ME (MIDPOINT -> ENDPOINT)

    #[test]
    fn test_me_mid_to_upper() {
        let quad1 = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        let quad2 = BoundedQuadratic::new(2., 4., 0., 0.5, 0.);
        let mut env = Vec::new();
        envelope_me(quad1, quad2, &mut env);
        assert_eq!(2, env.len());
        let (xf, xg, slope, intercept) = (env[1].lower, env[1].upper, env[1].b, env[1].c);
        assert!(relative_eq!(xf, 0.2583426132260586));
        assert!(relative_eq!(xg, 4.));
        assert!(relative_eq!(slope, 0.5166852264521172));
        assert!(relative_eq!(intercept, -0.06674090580846892));
    }

    #[test]
    fn test_me_mid_to_lower() {
        let quad1 = BoundedQuadratic::new(-2., 1., 0.5, 0., 0.);
        let quad2 = BoundedQuadratic::new(1.5, 4., 0., 1., -3.5);
        let mut env = Vec::new();
        envelope_me(quad1, quad2, &mut env);
        assert_eq!(3, env.len());
        let (xf, xg, slope, intercept) = (env[1].lower, env[1].upper, env[1].b, env[1].c);
        assert!(relative_eq!(xf, -1.));
        assert!(relative_eq!(xg, 1.5));
        assert!(relative_eq!(slope, -1.));
        assert!(relative_eq!(intercept, -0.5));
    }

    #[test]
    fn test_me_lower_to_mid() {
        let quad1 = BoundedQuadratic::new(-4., -1.5, 0., 2., 0.);
        let quad2 = BoundedQuadratic::new(-1., 2., 0.5, 0., 0.);
        let mut env = Vec::new();
        envelope_me(quad2.reflect_over_y(), quad1.reflect_over_y(), &mut env);
        assert_eq!(2, env.len());
        let (xf, xg, slope, intercept) = (env[1].lower, env[1].upper, env[1].b, env[1].c);
        assert!(relative_eq!(-xg, -4.));
        assert!(relative_eq!(-xf, 1.6568542494923804));
        assert!(relative_eq!(-slope, 1.6568542494923804));
        assert!(relative_eq!(intercept, -1.3725830020304794));
    }

    #[test]
    fn test_me_upper_to_mid() {
        let quad1 = BoundedQuadratic::new(-4., -1.5, 0., 0.5, -3.);
        let quad2 = BoundedQuadratic::new(-1., 2., 0.5, 0., 0.);
        let mut env = Vec::new();
        envelope_me(quad2.reflect_over_y(), quad1.reflect_over_y(), &mut env);
        assert_eq!(3, env.len());
        let (xf, xg, slope, intercept) = (env[1].lower, env[1].upper, env[1].b, env[1].c);
        assert!(relative_eq!(-xg, -1.5));
        assert!(relative_eq!(-xf, 1.6224989991991992));
        assert!(relative_eq!(-slope, 1.6224989991991992));
        assert!(relative_eq!(intercept, -1.3162515012012015));
    }

    #[test]
    fn test_me_mid_to_inf_upper() {
        let quad1 = BoundedQuadratic::new(-1., 2., 0.5, 0., 0.);
        let quad2 = BoundedQuadratic::new(3., f64::INFINITY, 0., 1., 1.);
        let mut env = Vec::new();
        envelope_me(quad1, quad2, &mut env);
        assert_eq!(2, env.len());
        let (xf, xg, slope, intercept) = (env[1].lower, env[1].upper, env[1].b, env[1].c);
        assert!(relative_eq!(xf, 1.));
        assert_eq!(xg, f64::INFINITY);
        assert!(relative_eq!(slope, 1.));
        assert!(relative_eq!(intercept, -0.5));
    }

    #[test]
    fn test_me_inf_lower_to_mid() {
        let quad1 = BoundedQuadratic::new(f64::NEG_INFINITY, -2., 0., 1., 1.);
        let quad2 = BoundedQuadratic::new(-1., 2., 0.5, 0., 0.);
        let p = PiecewiseQuadratic::new(vec![quad1, quad2]);
        let env = envelope(&p);
        assert_eq!(2, env.len());
        let (xf, xg, slope, intercept) = (env[0].lower, env[0].upper, env[0].b, env[0].c);
        assert_eq!(xg, 1.);
        assert!(relative_eq!(xf, f64::NEG_INFINITY));
        assert!(relative_eq!(slope, 1.));
        assert!(relative_eq!(intercept, -0.5));
    }

    #[test]
    fn test_me_no_env() {
        let quad1 = BoundedQuadratic::new(-1., 1., 1., 0., 0.);
        let quad2 = BoundedQuadratic::new(2., 4., 1., -6., 9.);
        let mut env = Vec::new();
        envelope_me(quad1, quad2, &mut env);
        assert_eq!(0, env.len(), "{:?}", env);
        envelope_me(quad2.reflect_over_y(), quad1.reflect_over_y(), &mut env);
        assert_eq!(0, env.len(), "{:?}", env);
    }

    // ENVELOPE MM (MIDPOINT -> MIDPOINT)
    #[test]
    fn test_mm_non_convex() {
        let quad1 = BoundedQuadratic::new(-1., 1., -1., 0., 0.);
        let quad2 = BoundedQuadratic::new(2., 4., 1., -6., 9.);
        let mut env = Vec::new();
        envelope_mm(quad1, quad2, &mut env);
        assert_eq!(0, env.len());
    }

    #[test]
    fn test_mm() {
        let f = BoundedQuadratic::new(-1., 2., 1., 0., 0.);
        let g = BoundedQuadratic::new(3., 7., 1., -8., 17.);
        let mut env = Vec::new();
        envelope_mm(f, g, &mut env);
        assert_eq!(3, env.len());
        let (xf, xg, slope, intercept) = (env[1].lower, env[1].upper, env[1].b, env[1].c);
        assert_eq!(xf, 0.125);
        assert_eq!(xg, 4.125);
        assert_eq!(slope, 0.25);
        assert_eq!(intercept, -0.015625);
    }

    #[test]
    fn test_env_pwq_bq_empty() {
        let f = BoundedQuadratic::new(0., 2., 0., 1., 0.);
        let mut p = vec![];
        add_piece(&mut p, f, &mut Vec::new());
        assert_eq!(p.len(), 1);
        assert!(p[0].approx(&f));
    }

    #[test]
    fn test_env_pwq_bq_connects_rightmost() {
        let f = BoundedQuadratic::new(-2., -1., 0., -1., 0.);
        let g = BoundedQuadratic::new(-1., 1., 0., -0.5, 0.5);
        let h = BoundedQuadratic::new(1., 3., 1., -2., 1.);
        let to_add = BoundedQuadratic::new(4., 6., 0., 2., -4.);
        let mut p = vec![f, g, h];
        add_piece(&mut p, to_add, &mut Vec::new());
        let env = PiecewiseQuadratic::new(p).simplify();
        assert_eq!(env.len(), 5);
        assert!(env[0].approx(&f));
        assert!(env[1].approx(&g));
        let expected_third = BoundedQuadratic::new(1., 1.7639320225002102, 1., -2., 1.);
        assert!(env[2].approx(&expected_third));
        let expected_fourth = BoundedQuadratic::new(
            1.7639320225002102,
            4.,
            0.,
            1.5278640450004204,
            -2.111456180001682,
        );
        assert!(env[3].approx(&expected_fourth));
        assert!(env[4].approx(&to_add));
    }

    #[test]
    fn test_env_pwq_bq_connects_middle() {
        let f = BoundedQuadratic::new(-2., -1., 0., -1., 0.);
        let g = BoundedQuadratic::new(-1., 1., 0., -0.5, 0.5);
        let h = BoundedQuadratic::new(1., 3., 1., -2., 1.);
        let to_add = BoundedQuadratic::new(4., 6., 0., 2., -10.);
        let mut p = vec![f, g, h];
        add_piece(&mut p, to_add, &mut Vec::new());
        let env = PiecewiseQuadratic::new(p).simplify();
        assert_eq!(env.len(), 3);
        assert!(env[0].approx(&f));
        println!("env {}", env);
        let expected_second = BoundedQuadratic::new(-1., 4., 0., -0.6, 0.4);
        assert!(env[1].approx(&expected_second));
        assert!(env[2].approx(&to_add));
    }

    #[test]
    fn test_env_pwq_bq_connects_leftmost() {
        let f = BoundedQuadratic::new(-2., -1., 0., -1., 0.);
        let g = BoundedQuadratic::new(-1., 1., 0., -0.5, 0.5);
        let h = BoundedQuadratic::new(1., 3., 1., -2., 1.);
        let to_add = BoundedQuadratic::new(4., 6., 0., 2., -12.);
        let mut p = vec![f, g, h];
        add_piece(&mut p, to_add, &mut Vec::new());
        let env = PiecewiseQuadratic::new(p).simplify();
        assert_eq!(env.len(), 2);
        let expected_first = BoundedQuadratic::new(-2., 4., 0., -1., 0.);
        assert!(env[0].approx(&expected_first));
        assert!(env[1].approx(&to_add));
    }

    #[test]
    #[should_panic(expected = "f_bq must be to the right (interval-wise) of f_pwq")]
    fn test_env_pwq_bq_invalid_input() {
        let f = BoundedQuadratic::new(-2., -1., 0., -1., 0.);
        add_piece(&mut vec![f], f, &mut vec![]);
    }

    #[test]
    fn test_envelope_of_convex_does_not_change() {
        let left = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., -1., 0.);
        let right = BoundedQuadratic::new(0., f64::INFINITY, 0., 1., 0.);
        let abs = PiecewiseQuadratic::new(vec![left, right]);
        let env = envelope(&abs);
        assert_eq!(env.len(), 2);
        assert!(abs[0].approx(&env[0]));
        assert!(abs[1].approx(&env[1]));
    }

    #[test]
    fn test_envelope_of_line_and_constant() {
        let line = BoundedQuadratic::new(f64::NEG_INFINITY, 0., 0., 1., 0.);
        let constant = BoundedQuadratic::new(0., 3., 0., 0., 0.);
        let env = envelope(&PiecewiseQuadratic::new(vec![line, constant]));
        assert_eq!(env.len(), 1);
        assert_eq!(env[0].b, line.b);
        assert_ne!(env[0].c, line.c);
    }

    #[test]
    fn test_env_between_point_and_quad() {
        let point = BoundedQuadratic::new_point(1., 0.);
        let quad = BoundedQuadratic::new(-6., 0., 0.5, 0., 2.);
        let p = PiecewiseQuadratic::new(vec![quad, point]);
        let env = envelope(&p);
        assert_eq!(2, env.len());
        assert!(env[0].a > 0.);
        assert!(env[1].is_affine());
    }

    #[test]
    fn tests_debugging() {
        // the below were taken from real examples encountered while debugging
        let f1 = BoundedQuadratic::new(
            0.,
            0.0013798462966265416,
            10.351752842693296,
            0.016231136312524996,
            0.0022444606012840007,
        );
        let f2 = BoundedQuadratic::new(
            0.0013798462966265416,
            0.0038939730235308346,
            10.351752842693296,
            -0.0266067015253128,
            0.0023035702331800287,
        );
        let f3 = BoundedQuadratic::new(
            0.0038939730235308346,
            0.007191571800214606,
            10.351752842693296,
            -0.015345440264051541,
            0.0022597191856177446,
        );
        let f4 = BoundedQuadratic::new(
            0.007191571800214606,
            0.01161175739449285,
            10.351752842693296,
            -0.1976877826063939,
            0.0035710472327920117,
        );
        let f5 = BoundedQuadratic::new(
            0.01161175739449285,
            0.014897662558493346,
            10.351752842693296,
            -0.17219228711089843,
            0.003274999724445933,
        );
        let f6 = BoundedQuadratic::new(
            0.014897662558493346,
            0.015903313249255063,
            10.351752842693296,
            -0.09381390873252002,
            0.0021073450914829406,
        );
        let f7 = BoundedQuadratic::new(
            0.015903313249255063,
            0.018206954947860392,
            10.351752842693296,
            -0.10183192675053805,
            0.0022348581436616526,
        );
        let f8 = BoundedQuadratic::new(
            0.018206954947860392,
            0.021960604619191918,
            10.351752842693296,
            -0.11003012494873629,
            0.0023841223689098777,
        );
        let f9 = BoundedQuadratic::new(
            0.021960604619191918,
            1.0000000000000002,
            10.351752842693296,
            0.000456361537750218,
            -0.000042227676583542915,
        );
        let p = PiecewiseQuadratic::new(vec![f1, f2, f3, f4, f5, f6, f7, f8, f9]);
        let env = envelope(&p);
        assert!(!env.is_empty());

        let f1 = BoundedQuadratic::new(
            0.,
            0.0015401857426080355,
            0.18651752738618688,
            -0.2983320669907763,
            0.010157944949219818,
        );
        let f2 = BoundedQuadratic::new(
            0.0015401857426080355,
            0.003696445782259288,
            0.18651752738618688,
            -0.25182043908379953,
            0.010086308403052002,
        );
        let mut env = Vec::new();
        assert!(PiecewiseQuadratic::new(vec![f1, f2]).is_convex());
        envelope_bq_bq(f1, f2, &mut env);
        assert!(!env.is_empty());

        let f1 = BoundedQuadratic::new(
            0.,
            0.00027844376687653647,
            10.602516867396563,
            -0.1410274857850643,
            0.00020274421100087257,
        );
        let f2 = BoundedQuadratic::new(
            0.00027844376687653647,
            0.000556887533753073,
            10.602516867396563,
            -0.14223845369441468,
            0.00020308139746711866,
        );
        let f3 = BoundedQuadratic::new(
            0.000556887533753073,
            0.0008353313006296095,
            10.602516867396563,
            -0.1541405382891729,
            0.00020970952000361404,
        );
        let f4 = BoundedQuadratic::new(
            0.0008353313006296095,
            0.001113775067506146,
            10.602516867396563,
            -0.15410593920604862,
            0.0002096806183065072,
        );
        let f5 = BoundedQuadratic::new(
            0.001113775067506146,
            0.0013922188343826827,
            10.602516867396563,
            -0.13967812154321668,
            0.0001936112747151202,
        );
        let f6 = BoundedQuadratic::new(
            0.0013922188343826827,
            1.,
            10.602516867396563,
            0.0011958512965030575,
            -0.000002516123546652083,
        );
        let p = PiecewiseQuadratic::new(vec![f1, f2, f3, f4, f5, f6]);
        let env = envelope(&p);
        assert!(!env.is_empty());

        let f = BoundedQuadratic::new(
            0.019474629474713647,
            0.019474629474713647,
            0.15297931295392428,
            -0.0004090169426131135,
            0.000003273394578839177,
        );
        let g = BoundedQuadratic::new(
            0.019474629474713647,
            f64::INFINITY,
            0.15297931295392428,
            0.0015909830573868865,
            -0.000032675864370588114,
        );
        let p = PiecewiseQuadratic::new(vec![f, g]);
        let env = envelope(&p);
        assert_eq!(2, env.len());

        let f = BoundedQuadratic::new(
            0.07224140601304432,
            0.15671202612669424,
            0.,
            -2.0921004489764066,
            0.33141385344925617,
        );
        let g = BoundedQuadratic::new(
            0.15671202612669424,
            f64::INFINITY,
            34.41902720475843,
            -7.063663429176061,
            0.510821995618197,
        );
        let p = PiecewiseQuadratic::new(vec![f, g]);
        let env = envelope(&p);
        assert_eq!(3, env.len());
    }
}
