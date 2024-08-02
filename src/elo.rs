use std::f64::consts::{PI, SQRT_2};

fn erf_inv(x: f64) -> f64 {
    let a = 8.0 * (PI - 3.0) / (3.0 * PI * (4.0 - PI));
    let y = (1.0 - x * x).ln();
    let z = 2.0 / (PI * a) + y / 2.0;
    ((z * z - y / a).sqrt() - z).sqrt().copysign(x)
}

fn phi_inv(p: f64) -> f64 {
    SQRT_2 * erf_inv(2.0 * p - 1.0)
}

fn elo(score: f64) -> f64 {
    if score <= 0.0 || score >= 1.0 {
        return 0.0;
    }
    return -400.0 * f64::log10(1.0 / score - 1.0);
}

pub fn elo_wld(wins: u32, losses: u32, draws: u32) -> (f64, f64, f64) {
    let n = wins + losses + draws;
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }

    let p_w = wins as f64 / n as f64;
    let p_l = losses as f64 / n as f64;
    let p_d = draws as f64 / n as f64;

    let mu = p_w + p_d / 2.0;
    let stdev = (p_w * (1.0 - mu).powi(2) + p_l * (0.0 - mu).powi(2) + p_d * (0.5 - mu).powi(2))
        .sqrt()
        / (n as f64).sqrt();

    let mu_min = mu + phi_inv(0.025) * stdev;
    let mu_max = mu + phi_inv(0.975) * stdev;

    (elo(mu_min), elo(mu), elo(mu_max))
}
