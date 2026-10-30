//! The scipy optimizers the models call, ported step for step so a fit takes Python's
//! iterates (only floating-point summation order differs: LAPACK/BLAS vs plain loops):
//! - `trust_exact`: `scipy.optimize.minimize(method="trust-exact")` (`_minimize_trust_region`
//!   with `IterativeSubproblem`, Moré–Sorensen on Cholesky factors; radius 1, max 1000,
//!   η 0.15, subproblem at most 25 iterations);
//! - `lmder`: MINPACK's Levenberg–Marquardt as `scipy.optimize.least_squares(method="lm")`
//!   calls it (`diag` from the Jacobian's column norms, factor 100), with its `qrfac`
//!   (column pivoting), `lmpar`, `qrsolv` and `enorm`.

use anyhow::{Result, bail};

/// A dense row-major matrix.
#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.cols + j] = value;
    }

    #[inline]
    pub fn add(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.cols + j] += value;
    }

    /// `self @ v`.
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        (0..self.rows).map(|i| dot(&self.data[i * self.cols..(i + 1) * self.cols], v)).collect()
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The 2-norm (`scipy.linalg.norm` of a vector).
pub fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn norm1(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

/// Upper Cholesky factor U (UᵀU = H, lower triangle zero) or the 1-based order of the first
/// leading minor that is not positive definite (LAPACK `potrf`'s `info`), with the partial
/// factor.
fn cholesky(h: &Mat) -> (Mat, usize) {
    let n = h.rows;
    let mut u = Mat::zeros(n, n);
    for j in 0..n {
        let mut d = h.at(j, j);
        for k in 0..j {
            d -= u.at(k, j) * u.at(k, j);
        }
        if d <= 0.0 || d.is_nan() {
            return (u, j + 1);
        }
        let d = d.sqrt();
        u.set(j, j, d);
        for i in j + 1..n {
            let mut s = h.at(j, i);
            for k in 0..j {
                s -= u.at(k, j) * u.at(k, i);
            }
            u.set(j, i, s / d);
        }
    }
    (u, 0)
}

/// Solve U x = b (U upper triangular, its leading `n` × `n` block).
fn solve_upper(u: &Mat, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = b.to_vec();
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in i + 1..n {
            s -= u.at(i, k) * x[k];
        }
        x[i] = s / u.at(i, i);
    }
    x
}

/// Solve Uᵀ x = b.
fn solve_upper_trans(u: &Mat, b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = b.to_vec();
    for i in 0..n {
        let mut s = x[i];
        for k in 0..i {
            s -= u.at(k, i) * x[k];
        }
        x[i] = s / u.at(i, i);
    }
    x
}

// --- trust-exact --------------------------------------------------------------------------

pub struct TrustResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub success: bool,
    pub message: &'static str,
    pub nit: usize,
}

/// `IterativeSubproblem` at one iterate (its λ state persists while a step is rejected).
struct Subproblem {
    x: Vec<f64>,
    fun: f64,
    jac: Vec<f64>,
    hess: Mat,
    jac_mag: f64,
    gershgorin_lb: f64,
    gershgorin_ub: f64,
    hess_inf: f64,
    hess_fro: f64,
    close_to_zero: f64,
    previous_tr_radius: f64,
    lambda_lb: f64,
}

const K_EASY: f64 = 0.1;
const K_HARD: f64 = 0.2;
const UPDATE_COEFF: f64 = 0.01;
const SUBPROBLEM_MAXITER: usize = 25;

impl Subproblem {
    fn new(x: Vec<f64>, fun: f64, jac: Vec<f64>, hess: Mat) -> Subproblem {
        let n = hess.rows;
        let (mut lb, mut ub, mut inf, mut fro) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64, 0.0);
        for i in 0..n {
            let row: f64 = (0..n).map(|j| hess.at(i, j).abs()).sum();
            let d = hess.at(i, i);
            lb = lb.min(d + d.abs() - row);
            ub = ub.max(d - d.abs() + row);
            inf = inf.max(row);
            fro += (0..n).map(|j| hess.at(i, j) * hess.at(i, j)).sum::<f64>();
        }
        let jac_mag = norm(&jac);
        Subproblem {
            x,
            fun,
            jac,
            jac_mag,
            gershgorin_lb: lb,
            gershgorin_ub: ub,
            hess_inf: inf,
            hess_fro: fro.sqrt(),
            close_to_zero: n as f64 * f64::EPSILON * inf,
            hess,
            previous_tr_radius: -1.0,
            lambda_lb: f64::NAN,
        }
    }

    /// The model's value at step p: f + g·p + ½ pᵀHp.
    fn model(&self, p: &[f64]) -> f64 {
        self.fun + dot(&self.jac, p) + 0.5 * dot(p, &self.hess.mul_vec(p))
    }

    fn shifted(&self, lambda: f64) -> Mat {
        let mut h = self.hess.clone();
        for i in 0..h.rows {
            h.add(i, i, lambda);
        }
        h
    }

    fn initial_values(&self, tr_radius: f64) -> (f64, f64, f64) {
        let lambda_ub = (self.jac_mag / tr_radius
            + (-self.gershgorin_lb).min(self.hess_fro).min(self.hess_inf))
        .max(0.0);
        let min_diag =
            (0..self.hess.rows).map(|i| self.hess.at(i, i)).fold(f64::INFINITY, f64::min);
        let mut lambda_lb = 0.0f64.max(-min_diag).max(
            self.jac_mag / tr_radius - self.gershgorin_ub.min(self.hess_fro).min(self.hess_inf),
        );
        if tr_radius < self.previous_tr_radius {
            lambda_lb = self.lambda_lb.max(lambda_lb);
        }
        let initial = if lambda_lb == 0.0 {
            0.0
        } else {
            (lambda_lb * lambda_ub).sqrt().max(lambda_lb + UPDATE_COEFF * (lambda_ub - lambda_lb))
        };
        (initial, lambda_lb, lambda_ub)
    }

    fn solve(&mut self, tr_radius: f64) -> Result<(Vec<f64>, bool)> {
        let n = self.hess.rows;
        let (mut lambda_current, mut lambda_lb, mut lambda_ub) = self.initial_values(tr_radius);
        let mut hits_boundary = true;
        let mut already_factorized = false;
        let mut h = Mat::zeros(n, n);
        let mut u = Mat::zeros(n, n);
        let mut info = 0;
        let mut p: Option<Vec<f64>> = None;
        let mut niter = 0;
        while niter < SUBPROBLEM_MAXITER {
            if already_factorized {
                already_factorized = false;
            } else {
                h = self.shifted(lambda_current);
                (u, info) = cholesky(&h);
            }
            niter += 1;
            if info == 0 && self.jac_mag > self.close_to_zero {
                let minus_g: Vec<f64> = self.jac.iter().map(|g| -g).collect();
                let mut step = solve_upper(&u, &solve_upper_trans(&u, &minus_g));
                let p_norm = norm(&step);
                if p_norm <= tr_radius && lambda_current == 0.0 {
                    hits_boundary = false;
                    p = Some(step);
                    break;
                }
                let w = solve_upper_trans(&u, &step);
                let w_norm = norm(&w);
                let delta_lambda = (p_norm / w_norm).powi(2) * (p_norm - tr_radius) / tr_radius;
                let lambda_new = lambda_current + delta_lambda;
                if p_norm < tr_radius {
                    let (s_min, z_min) = smallest_singular_value(&u);
                    let (ta, tb) = boundaries_intersections(&step, &z_min, tr_radius);
                    let step_len = if ta.abs() <= tb.abs() { ta } else { tb };
                    let quadratic_term = dot(&step, &h.mul_vec(&step));
                    let relative_error = (step_len.powi(2) * s_min.powi(2))
                        / (quadratic_term + lambda_current * tr_radius.powi(2));
                    if relative_error <= K_HARD {
                        for (s, z) in step.iter_mut().zip(&z_min) {
                            *s += step_len * z;
                        }
                        p = Some(step);
                        break;
                    }
                    lambda_ub = lambda_current;
                    lambda_lb = lambda_lb.max(lambda_current - s_min.powi(2));
                    let shifted = self.shifted(lambda_new);
                    let (c, c_info) = cholesky(&shifted);
                    if c_info == 0 {
                        lambda_current = lambda_new;
                        already_factorized = true;
                        h = shifted;
                        u = c;
                        info = 0;
                    } else {
                        lambda_lb = lambda_lb.max(lambda_new);
                        lambda_current = (lambda_lb * lambda_ub)
                            .abs()
                            .sqrt()
                            .max(lambda_lb + UPDATE_COEFF * (lambda_ub - lambda_lb));
                    }
                } else {
                    let relative_error = (p_norm - tr_radius).abs() / tr_radius;
                    if relative_error <= K_EASY {
                        p = Some(step);
                        break;
                    }
                    lambda_lb = lambda_current;
                    lambda_current = lambda_new;
                }
                p = Some(step);
            } else if info == 0 && self.jac_mag <= self.close_to_zero {
                if lambda_current == 0.0 {
                    p = Some(vec![0.0; n]);
                    hits_boundary = false;
                    break;
                }
                let (s_min, z_min) = smallest_singular_value(&u);
                let step_len = tr_radius;
                p = Some(z_min.iter().map(|z| step_len * z).collect());
                if step_len.powi(2) * s_min.powi(2) <= K_HARD * lambda_current * tr_radius.powi(2) {
                    break;
                }
                lambda_ub = lambda_current;
                lambda_lb = lambda_lb.max(lambda_current - s_min.powi(2));
                lambda_current = (lambda_lb * lambda_ub)
                    .sqrt()
                    .max(lambda_lb + UPDATE_COEFF * (lambda_ub - lambda_lb));
            } else {
                let (delta, v) = singular_leading_submatrix(&h, &u, info);
                let v_norm = norm(&v);
                lambda_lb = lambda_lb.max(lambda_current + delta / v_norm.powi(2));
                lambda_current = (lambda_lb * lambda_ub)
                    .abs()
                    .sqrt()
                    .max(lambda_lb + UPDATE_COEFF * (lambda_ub - lambda_lb));
            }
        }
        self.lambda_lb = lambda_lb;
        self.previous_tr_radius = tr_radius;
        match p {
            Some(p) => Ok((p, hits_boundary)),
            None => bail!("trust-exact: the subproblem found no step"),
        }
    }
}

/// `estimate_smallest_singular_value(U)`.
fn smallest_singular_value(u: &Mat) -> (f64, Vec<f64>) {
    let n = u.rows;
    let mut p = vec![0.0; n];
    let mut w = vec![0.0; n];
    for k in 0..n {
        let wp = (1.0 - p[k]) / u.at(k, k);
        let wm = (-1.0 - p[k]) / u.at(k, k);
        let pp: Vec<f64> = (k + 1..n).map(|i| p[i] + u.at(k, i) * wp).collect();
        let pm: Vec<f64> = (k + 1..n).map(|i| p[i] + u.at(k, i) * wm).collect();
        if wp.abs() + norm1(&pp) >= wm.abs() + norm1(&pm) {
            w[k] = wp;
            p[k + 1..].copy_from_slice(&pp);
        } else {
            w[k] = wm;
            p[k + 1..].copy_from_slice(&pm);
        }
    }
    let v = solve_upper(u, &w);
    let (v_norm, w_norm) = (norm(&v), norm(&w));
    (w_norm / v_norm, v.iter().map(|x| x / v_norm).collect())
}

/// `get_boundaries_intersections`: t with ‖z + t d‖ = radius, sorted.
fn boundaries_intersections(z: &[f64], d: &[f64], radius: f64) -> (f64, f64) {
    let a = dot(d, d);
    let b = 2.0 * dot(z, d);
    let c = dot(z, z) - radius.powi(2);
    let sqrt_discriminant = (b * b - 4.0 * a * c).sqrt();
    let aux = b + sqrt_discriminant.copysign(b);
    let (ta, tb) = (-aux / (2.0 * a), -2.0 * c / aux);
    if ta <= tb { (ta, tb) } else { (tb, ta) }
}

/// `singular_leading_submatrix(A, U, k)`.
fn singular_leading_submatrix(a: &Mat, u: &Mat, k: usize) -> (f64, Vec<f64>) {
    let n = a.rows;
    let delta: f64 = (0..k - 1).map(|i| u.at(i, k - 1).powi(2)).sum::<f64>() - a.at(k - 1, k - 1);
    let mut v = vec![0.0; n];
    v[k - 1] = 1.0;
    if k != 1 {
        let rhs: Vec<f64> = (0..k - 1).map(|i| -u.at(i, k - 1)).collect();
        let solved = solve_upper(u, &rhs);
        v[..k - 1].copy_from_slice(&solved);
    }
    (delta, v)
}

/// `minimize(fun, x0, jac=True, hess=hess, method="trust-exact", options={"gtol": gtol,
/// "maxiter": maxiter})` (maxiter None → 200 · len(x0)). `fun` returns (value, gradient).
pub fn trust_exact(
    mut fun: impl FnMut(&[f64]) -> (f64, Vec<f64>),
    mut hess: impl FnMut(&[f64]) -> Mat,
    x0: Vec<f64>,
    gtol: f64,
    maxiter: Option<usize>,
) -> Result<TrustResult> {
    const INITIAL_TRUST_RADIUS: f64 = 1.0;
    const MAX_TRUST_RADIUS: f64 = 1000.0;
    const ETA: f64 = 0.15;
    let maxiter = maxiter.unwrap_or(200 * x0.len());
    let mut make = |x: Vec<f64>| {
        let (f, g) = fun(&x);
        let h = hess(&x);
        Subproblem::new(x, f, g, h)
    };
    let mut trust_radius = INITIAL_TRUST_RADIUS;
    let mut m = make(x0);
    let mut k = 0;
    let mut warnflag = 0;
    while m.jac_mag >= gtol {
        let (p, hits_boundary) = match m.solve(trust_radius) {
            Ok(step) => step,
            Err(_) => {
                warnflag = 3;
                break;
            }
        };
        let predicted_value = m.model(&p);
        let x_proposed: Vec<f64> = m.x.iter().zip(&p).map(|(x, p)| x + p).collect();
        let proposed = make(x_proposed);
        let actual_reduction = m.fun - proposed.fun;
        let predicted_reduction = m.fun - predicted_value;
        if predicted_reduction <= 0.0 {
            warnflag = 2;
            break;
        }
        let rho = actual_reduction / predicted_reduction;
        if rho < 0.25 {
            trust_radius *= 0.25;
        } else if rho > 0.75 && hits_boundary {
            trust_radius = (2.0 * trust_radius).min(MAX_TRUST_RADIUS);
        }
        if rho > ETA {
            m = proposed;
        }
        k += 1;
        if m.jac_mag < gtol {
            warnflag = 0;
            break;
        }
        if k >= maxiter {
            warnflag = 1;
            break;
        }
    }
    let message = match warnflag {
        0 => "Optimization terminated successfully.",
        1 => "Maximum number of iterations has been exceeded.",
        2 => "A bad approximation caused failure to predict improvement.",
        _ => "A linalg error occurred, such as a non-psd Hessian.",
    };
    Ok(TrustResult { x: m.x, fun: m.fun, success: warnflag == 0, message, nit: k })
}

// --- MINPACK lmder ------------------------------------------------------------------------

/// MINPACK's `enorm`: the Euclidean norm with its scaling against under- and overflow.
fn enorm(x: &[f64]) -> f64 {
    const RDWARF: f64 = 3.834e-20;
    const RGIANT: f64 = 1.304e19;
    let (mut s1, mut s2, mut s3) = (0.0f64, 0.0f64, 0.0f64);
    let (mut x1max, mut x3max) = (0.0f64, 0.0f64);
    let agiant = RGIANT / x.len() as f64;
    for &xi in x {
        let xabs = xi.abs();
        if xabs > RDWARF && xabs < agiant {
            s2 += xabs * xabs;
        } else if xabs <= RDWARF {
            if xabs > x3max {
                s3 = 1.0 + s3 * (x3max / xabs).powi(2);
                x3max = xabs;
            } else if xabs != 0.0 {
                s3 += (xabs / x3max).powi(2);
            }
        } else if xabs > x1max {
            s1 = 1.0 + s1 * (x1max / xabs).powi(2);
            x1max = xabs;
        } else {
            s1 += (xabs / x1max).powi(2);
        }
    }
    if s1 != 0.0 {
        x1max * (s1 + (s2 / x1max) / x1max).sqrt()
    } else if s2 != 0.0 {
        if s2 >= x3max {
            (s2 * (1.0 + (x3max / s2) * (x3max * s3))).sqrt()
        } else {
            (x3max * ((s2 / x3max) + (x3max * s3))).sqrt()
        }
    } else {
        x3max * s3.sqrt()
    }
}

/// Column-major m × n storage for the MINPACK routines (as Fortran keeps `fjac`).
struct ColMat {
    m: usize,
    a: Vec<f64>,
}

impl ColMat {
    #[inline]
    fn at(&self, i: usize, j: usize) -> f64 {
        self.a[j * self.m + i]
    }
    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.a[j * self.m + i] = v;
    }
    fn col(&self, j: usize, from: usize) -> &[f64] {
        &self.a[j * self.m + from..(j + 1) * self.m]
    }
}

/// MINPACK `qrfac` with column pivoting: a ← Householder vectors and R's strict upper part;
/// returns (ipvt, rdiag, acnorm).
fn qrfac(a: &mut ColMat, n: usize) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let m = a.m;
    let epsmch = f64::EPSILON;
    let acnorm: Vec<f64> = (0..n).map(|j| enorm(a.col(j, 0))).collect();
    let mut rdiag = acnorm.clone();
    let mut wa = rdiag.clone();
    let mut ipvt: Vec<usize> = (0..n).collect();
    for j in 0..m.min(n) {
        let mut kmax = j;
        for k in j..n {
            if rdiag[k] > rdiag[kmax] {
                kmax = k;
            }
        }
        if kmax != j {
            for i in 0..m {
                let temp = a.at(i, j);
                a.set(i, j, a.at(i, kmax));
                a.set(i, kmax, temp);
            }
            rdiag[kmax] = rdiag[j];
            wa[kmax] = wa[j];
            ipvt.swap(j, kmax);
        }
        let mut ajnorm = enorm(a.col(j, j));
        if ajnorm != 0.0 {
            if a.at(j, j) < 0.0 {
                ajnorm = -ajnorm;
            }
            for i in j..m {
                a.set(i, j, a.at(i, j) / ajnorm);
            }
            a.set(j, j, a.at(j, j) + 1.0);
            for k in j + 1..n {
                let mut sum = 0.0;
                for i in j..m {
                    sum += a.at(i, j) * a.at(i, k);
                }
                let temp = sum / a.at(j, j);
                for i in j..m {
                    a.set(i, k, a.at(i, k) - temp * a.at(i, j));
                }
                if rdiag[k] != 0.0 {
                    let temp = a.at(j, k) / rdiag[k];
                    rdiag[k] *= (1.0 - temp * temp).max(0.0).sqrt();
                    if 0.05 * (rdiag[k] / wa[k]).powi(2) <= epsmch {
                        rdiag[k] = enorm(a.col(k, j + 1));
                        wa[k] = rdiag[k];
                    }
                }
            }
        }
        rdiag[j] = -ajnorm;
    }
    (ipvt, rdiag, acnorm)
}

/// MINPACK `qrsolv`: solves (R; D) x ≈ (Qᵀb; 0) in the least-squares sense. `r` is n × n
/// column-major (upper triangle R; its strict lower triangle is overwritten).
fn qrsolv(
    r: &mut ColMat,
    n: usize,
    ipvt: &[usize],
    diag: &[f64],
    qtb: &[f64],
    sdiag: &mut [f64],
) -> Vec<f64> {
    let mut x = vec![0.0; n];
    let mut wa = vec![0.0; n];
    for j in 0..n {
        for i in j..n {
            r.set(i, j, r.at(j, i));
        }
        x[j] = r.at(j, j);
        wa[j] = qtb[j];
    }
    for j in 0..n {
        let l = ipvt[j];
        if diag[l] != 0.0 {
            for s in sdiag.iter_mut().skip(j) {
                *s = 0.0;
            }
            sdiag[j] = diag[l];
            let mut qtbpj = 0.0;
            for k in j..n {
                if sdiag[k] == 0.0 {
                    continue;
                }
                let (sin, cos);
                if r.at(k, k).abs() < sdiag[k].abs() {
                    let cotan = r.at(k, k) / sdiag[k];
                    sin = 0.5 / (0.25 + 0.25 * cotan * cotan).sqrt();
                    cos = sin * cotan;
                } else {
                    let tan = sdiag[k] / r.at(k, k);
                    cos = 0.5 / (0.25 + 0.25 * tan * tan).sqrt();
                    sin = cos * tan;
                }
                r.set(k, k, cos * r.at(k, k) + sin * sdiag[k]);
                let temp = cos * wa[k] + sin * qtbpj;
                qtbpj = -sin * wa[k] + cos * qtbpj;
                wa[k] = temp;
                for i in k + 1..n {
                    let temp = cos * r.at(i, k) + sin * sdiag[i];
                    sdiag[i] = -sin * r.at(i, k) + cos * sdiag[i];
                    r.set(i, k, temp);
                }
            }
        }
        sdiag[j] = r.at(j, j);
        r.set(j, j, x[j]);
    }
    let mut nsing = n;
    for j in 0..n {
        if sdiag[j] == 0.0 && nsing == n {
            nsing = j;
        }
        if nsing < n {
            wa[j] = 0.0;
        }
    }
    for j in (0..nsing).rev() {
        let mut sum = 0.0;
        for i in j + 1..nsing {
            sum += r.at(i, j) * wa[i];
        }
        wa[j] = (wa[j] - sum) / sdiag[j];
    }
    for j in 0..n {
        x[ipvt[j]] = wa[j];
    }
    x
}

/// MINPACK `lmpar`: the Levenberg–Marquardt parameter for radius `delta`; returns (par, x).
fn lmpar(
    r: &mut ColMat,
    n: usize,
    ipvt: &[usize],
    diag: &[f64],
    qtb: &[f64],
    delta: f64,
    mut par: f64,
) -> (f64, Vec<f64>) {
    const P1: f64 = 0.1;
    const P001: f64 = 0.001;
    let dwarf = f64::MIN_POSITIVE;
    let mut wa1 = vec![0.0; n];
    let mut wa2 = vec![0.0; n];
    let mut sdiag = vec![0.0; n];
    // The Gauss-Newton direction.
    let mut nsing = n;
    for j in 0..n {
        wa1[j] = qtb[j];
        if r.at(j, j) == 0.0 && nsing == n {
            nsing = j;
        }
        if nsing < n {
            wa1[j] = 0.0;
        }
    }
    for j in (0..nsing).rev() {
        wa1[j] /= r.at(j, j);
        let temp = wa1[j];
        for i in 0..j {
            wa1[i] -= r.at(i, j) * temp;
        }
    }
    let mut x = vec![0.0; n];
    for j in 0..n {
        x[ipvt[j]] = wa1[j];
    }
    let mut iter = 0;
    for j in 0..n {
        wa2[j] = diag[j] * x[j];
    }
    let mut dxnorm = enorm(&wa2);
    let mut fp = dxnorm - delta;
    if fp > P1 * delta {
        // Lower bound parl (zero if the Jacobian is rank-deficient).
        let mut parl = 0.0;
        if nsing >= n {
            for j in 0..n {
                let l = ipvt[j];
                wa1[j] = diag[l] * (wa2[l] / dxnorm);
            }
            for j in 0..n {
                let mut sum = 0.0;
                for i in 0..j {
                    sum += r.at(i, j) * wa1[i];
                }
                wa1[j] = (wa1[j] - sum) / r.at(j, j);
            }
            let temp = enorm(&wa1);
            parl = ((fp / delta) / temp) / temp;
        }
        // Upper bound paru.
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..=j {
                sum += r.at(i, j) * qtb[i];
            }
            wa1[j] = sum / diag[ipvt[j]];
        }
        let gnorm = enorm(&wa1);
        let mut paru = gnorm / delta;
        if paru == 0.0 {
            paru = dwarf / delta.min(P1);
        }
        par = par.max(parl).min(paru);
        if par == 0.0 {
            par = gnorm / dxnorm;
        }
        loop {
            iter += 1;
            if par == 0.0 {
                par = dwarf.max(P001 * paru);
            }
            let temp = par.sqrt();
            for j in 0..n {
                wa1[j] = temp * diag[j];
            }
            x = qrsolv(r, n, ipvt, &wa1, qtb, &mut sdiag);
            for j in 0..n {
                wa2[j] = diag[j] * x[j];
            }
            dxnorm = enorm(&wa2);
            let temp = fp;
            fp = dxnorm - delta;
            if fp.abs() <= P1 * delta || (parl == 0.0 && fp <= temp && temp < 0.0) || iter == 10 {
                break;
            }
            // Newton correction.
            for j in 0..n {
                let l = ipvt[j];
                wa1[j] = diag[l] * (wa2[l] / dxnorm);
            }
            for j in 0..n {
                wa1[j] /= sdiag[j];
                let temp = wa1[j];
                for i in j + 1..n {
                    wa1[i] -= r.at(i, j) * temp;
                }
            }
            let temp = enorm(&wa1);
            let parc = ((fp / delta) / temp) / temp;
            if fp > 0.0 {
                parl = parl.max(par);
            }
            if fp < 0.0 {
                paru = paru.min(par);
            }
            par = parl.max(par + parc);
        }
    }
    if iter == 0 {
        par = 0.0;
    }
    (par, x)
}

/// MINPACK `lmder` (mode 1: `diag` from the Jacobian's column norms; factor 100), as
/// `least_squares(method="lm", ftol, xtol, gtol, max_nfev)` calls it. `fcn(x)` returns
/// (residuals, Jacobian as m rows × n). Returns (x, info): info 1–4 converged
/// (`least_squares` success), 5 too many evaluations; 6–8 (tolerances too small) are
/// errors, as Python's `least_squares` has no status for them.
pub fn lmder(
    mut fcn: impl FnMut(&[f64]) -> (Vec<f64>, Mat),
    x0: &[f64],
    ftol: f64,
    xtol: f64,
    gtol: f64,
    maxfev: usize,
) -> Result<(Vec<f64>, i32)> {
    const FACTOR: f64 = 100.0;
    let epsmch = f64::EPSILON;
    let n = x0.len();
    let mut x = x0.to_vec();
    let (mut fvec, _) = fcn(&x);
    let m = fvec.len();
    if n == 0 || m < n || ftol < 0.0 || xtol < 0.0 || gtol < 0.0 || maxfev == 0 {
        bail!("lmder: improper input parameters");
    }
    let mut nfev = 1;
    let mut fnorm = enorm(&fvec);
    let mut par = 0.0;
    let mut iter = 1;
    let mut diag = vec![0.0; n];
    let (mut delta, mut xnorm) = (0.0, 0.0);
    let mut info = 0;
    'outer: loop {
        let (_, jac) = fcn(&x);
        let mut fjac = ColMat { m, a: vec![0.0; m * n] };
        for i in 0..m {
            for j in 0..n {
                fjac.set(i, j, jac.at(i, j));
            }
        }
        let (ipvt, rdiag, acnorm) = qrfac(&mut fjac, n);
        if iter == 1 {
            for j in 0..n {
                diag[j] = if acnorm[j] == 0.0 { 1.0 } else { acnorm[j] };
            }
            let scaled: Vec<f64> = (0..n).map(|j| diag[j] * x[j]).collect();
            xnorm = enorm(&scaled);
            delta = FACTOR * xnorm;
            if delta == 0.0 {
                delta = FACTOR;
            }
        }
        // Qᵀ fvec in its first n entries, R's diagonal back into fjac.
        let mut wa4 = fvec.clone();
        let mut qtf = vec![0.0; n];
        for j in 0..n {
            if fjac.at(j, j) != 0.0 {
                let mut sum = 0.0;
                for i in j..m {
                    sum += fjac.at(i, j) * wa4[i];
                }
                let temp = -sum / fjac.at(j, j);
                for i in j..m {
                    wa4[i] += fjac.at(i, j) * temp;
                }
            }
            fjac.set(j, j, rdiag[j]);
            qtf[j] = wa4[j];
        }
        // The scaled gradient's norm.
        let mut gnorm = 0.0f64;
        if fnorm != 0.0 {
            for j in 0..n {
                let l = ipvt[j];
                if acnorm[l] != 0.0 {
                    let mut sum = 0.0;
                    for i in 0..=j {
                        sum += fjac.at(i, j) * (qtf[i] / fnorm);
                    }
                    gnorm = gnorm.max((sum / acnorm[l]).abs());
                }
            }
        }
        if gnorm <= gtol {
            info = 4;
            break 'outer;
        }
        for j in 0..n {
            diag[j] = diag[j].max(acnorm[j]);
        }
        loop {
            // R (n × n, column-major) for lmpar.
            let mut r = ColMat { m: n, a: vec![0.0; n * n] };
            for j in 0..n {
                for i in 0..n {
                    r.set(i, j, fjac.at(i, j));
                }
            }
            let (new_par, step) = lmpar(&mut r, n, &ipvt, &diag, &qtf, delta, par);
            par = new_par;
            // Keep lmpar's overwritten strict lower triangle, as fjac shares it in MINPACK.
            for j in 0..n {
                for i in j + 1..n {
                    fjac.set(i, j, r.at(i, j));
                }
            }
            let wa1: Vec<f64> = step.iter().map(|s| -s).collect();
            let wa2: Vec<f64> = (0..n).map(|j| x[j] + wa1[j]).collect();
            let wa3: Vec<f64> = (0..n).map(|j| diag[j] * wa1[j]).collect();
            let pnorm = enorm(&wa3);
            if iter == 1 {
                delta = delta.min(pnorm);
            }
            let (trial, _) = fcn(&wa2);
            nfev += 1;
            let fnorm1 = enorm(&trial);
            let actred = if 0.1 * fnorm1 < fnorm { 1.0 - (fnorm1 / fnorm).powi(2) } else { -1.0 };
            let mut jp = vec![0.0; n];
            for j in 0..n {
                let temp = wa1[ipvt[j]];
                for i in 0..=j {
                    jp[i] += fjac.at(i, j) * temp;
                }
            }
            let temp1 = enorm(&jp) / fnorm;
            let temp2 = (par.sqrt() * pnorm) / fnorm;
            let prered = temp1 * temp1 + temp2 * temp2 / 0.5;
            let dirder = -(temp1 * temp1 + temp2 * temp2);
            let ratio = if prered != 0.0 { actred / prered } else { 0.0 };
            if ratio <= 0.25 {
                let mut temp =
                    if actred >= 0.0 { 0.5 } else { 0.5 * dirder / (dirder + 0.5 * actred) };
                if 0.1 * fnorm1 >= fnorm || temp < 0.1 {
                    temp = 0.1;
                }
                delta = temp * delta.min(pnorm / 0.1);
                par /= temp;
            } else if par == 0.0 || ratio >= 0.75 {
                delta = pnorm / 0.5;
                par *= 0.5;
            }
            if ratio >= 1e-4 {
                x = wa2;
                let scaled: Vec<f64> = (0..n).map(|j| diag[j] * x[j]).collect();
                fvec = trial;
                xnorm = enorm(&scaled);
                fnorm = fnorm1;
                iter += 1;
            }
            if actred.abs() <= ftol && prered <= ftol && 0.5 * ratio <= 1.0 {
                info = 1;
            }
            if delta <= xtol * xnorm {
                info = if info == 1 { 3 } else { 2 };
            }
            if info != 0 {
                break 'outer;
            }
            if nfev >= maxfev {
                info = 5;
            }
            if actred.abs() <= epsmch && prered <= epsmch && 0.5 * ratio <= 1.0 {
                info = 6;
            }
            if delta <= epsmch * xnorm {
                info = 7;
            }
            if gnorm <= epsmch {
                info = 8;
            }
            if info != 0 {
                break 'outer;
            }
            if ratio >= 1e-4 {
                break;
            }
        }
    }
    let _ = fvec;
    if info >= 6 {
        bail!("lmder: tolerance too small (MINPACK info {info})");
    }
    Ok((x, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_exact_minimizes_a_quadratic() {
        // f = (x0 − 1)² + 10 (x1 + 2)², from the origin.
        let fun = |x: &[f64]| {
            let f = (x[0] - 1.0).powi(2) + 10.0 * (x[1] + 2.0).powi(2);
            (f, vec![2.0 * (x[0] - 1.0), 20.0 * (x[1] + 2.0)])
        };
        let hess = |_: &[f64]| Mat { rows: 2, cols: 2, data: vec![2.0, 0.0, 0.0, 20.0] };
        let r = trust_exact(fun, hess, vec![0.0, 0.0], 1e-10, None).unwrap();
        assert!(r.success);
        assert!((r.x[0] - 1.0).abs() < 1e-12 && (r.x[1] + 2.0).abs() < 1e-12, "{:?}", r.x);
    }

    #[test]
    fn lmder_fits_an_exponential() {
        // y = 2 exp(−0.5 t), exact data: residuals a exp(b t) − y.
        let t: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = t.iter().map(|t| 2.0 * (-0.5 * t).exp()).collect();
        let fcn = |x: &[f64]| {
            let r: Vec<f64> = t.iter().zip(&y).map(|(t, y)| x[0] * (x[1] * t).exp() - y).collect();
            let mut j = Mat::zeros(t.len(), 2);
            for (i, t) in t.iter().enumerate() {
                j.set(i, 0, (x[1] * t).exp());
                j.set(i, 1, x[0] * t * (x[1] * t).exp());
            }
            (r, j)
        };
        let (x, info) = lmder(fcn, &[1.0, 0.0], 1e-12, 1e-12, 1e-12, 200).unwrap();
        assert!((1..=4).contains(&info), "info {info}");
        assert!((x[0] - 2.0).abs() < 1e-9 && (x[1] + 0.5).abs() < 1e-9, "{x:?}");
    }
}
