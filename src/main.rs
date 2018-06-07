extern crate rand;

use std::f64::consts::PI;

enum WindowType {
    Rectangular,
    Hamming,
    BlackmanHarris
}

fn generate_window(window_type: WindowType, window_len: u32) -> Vec<f64> {
    assert!(window_len % 2 == 1);

    let samples = (1..window_len).map(|e| e as f64);
    let m = window_len as f64;

    let window_function = |n: f64| {
        match window_type {
            WindowType::Rectangular => 1.0,
            WindowType::Hamming =>
                0.54 - 0.46 * f64::cos((2.0*n*PI)/m),
            WindowType::BlackmanHarris =>
                0.35875 - 0.48829 * f64::cos((2.0*n*PI)/m)
                        + 0.14128 * f64::cos((4.0*n*PI)/m)
                        - 0.01168 * f64::cos((6.0*n*PI)/m)
        }
    };

    samples.map(window_function)
           .collect()
}

// Cutoffs are omega/pi => 0 is 0, 1 is nyquist
enum FilterType {
    LowPass(f64),
    HighPass(f64),
    BandPass(f64, f64),
    BandStop(f64, f64)
}

fn fir_design(filter_type: FilterType, window_type: WindowType, filter_len: u32) -> Vec<f64> {
    assert!(filter_len % 2 == 1);

    let samples = (1..filter_len).map(|e| e as f64);
    let m = filter_len as f64;
    let sinc = |x: f64| f64::sin(x)/x;

    let filter_fn = |n: f64| {
        let lowpass = |c: f64| c * sinc(c * PI * (n - m / 2.0));
        match filter_type {
            FilterType::LowPass(c) =>
                lowpass(c),
            FilterType::HighPass(c) =>
                lowpass(1.0) - lowpass(c),
            FilterType::BandPass(cl, ch) =>
                lowpass(ch) - lowpass(cl),
            FilterType::BandStop(cl, ch) =>
                lowpass(1.0) - lowpass(ch) + lowpass(cl)
        }
    };

    samples.map(filter_fn)
           .zip(generate_window(window_type, filter_len))
           .map(|(a, b)| a * b)
           .collect()
}

// Simple, O(n^2), real -> abs mag dft
fn dft(x: &[f64], n_samples: usize) -> Vec<f64> {

    let mut y_re = vec![0.0; n_samples];
    let mut y_im = vec![0.0; n_samples];

    let m = x.len();

    for k in 0..n_samples {
        for n in 0..(m-1) {
            y_re[k] += x[n] * f64::cos(2.0 * PI * k as f64 * n as f64 / n_samples as f64);
            y_im[k] -= x[n] * f64::sin(2.0 * PI * k as f64 * n as f64 / n_samples as f64);
        }
    }

    y_re.iter()
        .zip(y_im)
        .map(|(a, b)| f64::sqrt(a*a + b*b))
        .collect()
}

// Simple O(n^2) convolution
fn convolve(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let result_len = signal.len() + kernel.len() - 1;
    let mut y = vec![0.0; result_len];
    for n in 0..result_len {
        let kmin = if n >= kernel.len() - 1 {n - (kernel.len() - 1)} else {0};
        let kmax = if n < signal.len() - 1 {n} else {signal.len() - 1};
        for k in kmin..kmax {
            y[n] += signal[k] * kernel[n - k];
        }
    }
    y
}

// Applies convolution, but corrects for time delay and culls to signal length
fn filter(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let mut ret = convolve(signal, kernel).split_off(kernel.len()/2);
    ret.resize(signal.len(), 0.0);
    ret
}

// White noise, DC offset 0, amplitude measured from 0 -> extent
fn white_noise(n_samples: usize, amplitude: f64) -> Vec<f64> {
    use rand::Rng;
    let mut v = vec![0.0; n_samples];
    let mut rng = rand::thread_rng();
    for x in v.iter_mut() {
        *x = rng.gen();
        *x = 2.0 * (*x - 0.5) * amplitude;
    }
    v
}

fn main() {
    let f = fir_design(FilterType::LowPass(0.1), WindowType::Hamming, 51);
    //println!("{:?}", dft(&f, 256));
    let noise = white_noise(100, 1.0);
    println!("{:?}", noise);
    println!("{:?}", filter(&noise, &f));
    //println!("octave --eval \"freqz({:?}, 1)\" --persist", f);
}
