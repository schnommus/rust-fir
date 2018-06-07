extern crate rand;
extern crate raster;
extern crate byteorder;

use byteorder::*;

use std::fs::File;
use std::env;
use std::process;

use std::collections::VecDeque;
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

const Fs: f64 = 20e6;
const nyquist: f64 = Fs/2.0;
const n_lines: i32 = 625;
const n_columns: i32 = 700;
const line_freq_hz: f64 = 15.625e3;
const frame_freq_hz: f64 = 50.0;
const CHUNK_SIZE: usize = 1024*32;

use raster::Image;
use raster::Color;

struct CompositeProcessor {
    blocks: VecDeque<Vec<f64>>,
    sample_base: usize,
    hsync_sawtooth: f64,
    vsync_sawtooth: f64,
    sync_threshold: f64,
    white_level: f64,
    black_level: f64,
    canvas: Image
}

impl CompositeProcessor {

    pub fn new() -> CompositeProcessor {
        let blocks = VecDeque::new();
        let sample_base = 0;
        let hsync_sawtooth = 0.0;
        let vsync_sawtooth = 0.0;
        let sync_threshold = 0.05;
        let white_level = 1.0;
        let black_level = 0.2;
        let canvas = Image::blank(n_columns, n_lines);
        CompositeProcessor {
            blocks,
            sample_base,
            hsync_sawtooth,
            vsync_sawtooth,
            sync_threshold,
            white_level,
            black_level,
            canvas
        }
    }

    pub fn process_block(&mut self, block: &[f64]) {
        self.blocks.push_back(block.to_vec());
        if self.blocks.len() > 3 {
            self.blocks.pop_front();
            self.sample_base += block.len();
            self.process();
            /* TODO: return video frame if produced? */
        }
    }

    fn process(&mut self) {
        println!("Processing blocks starting at: {}", self.sample_base);

        let block_size = self.blocks[0].len();
        let mut whole_block: Vec<f64> = self.blocks[0].clone();
        for block in self.blocks.iter().skip(1) {
            whole_block.extend(block)
        }

        let intensity_filter =
            fir_design(FilterType::LowPass(2e6/nyquist), WindowType::Hamming, 101);
        let intensity_filtered =
            filter(&whole_block, &intensity_filter);

        for i in block_size..(block_size*2) {
            let s = whole_block[i];
            self.hsync_sawtooth += (n_columns as f64) * line_freq_hz/Fs;
            self.vsync_sawtooth += (n_lines as f64) * frame_freq_hz/Fs;
            if s < self.sync_threshold {
                self.hsync_sawtooth = 0.0;
            }
            if self.hsync_sawtooth < n_columns as f64 &&
               self.vsync_sawtooth < n_lines as f64 {
                let v: u8 =
                    if s < self.black_level { 0 }
                    else if s > self.white_level { 1 }
                    else {
                        (255.0 * ((s - self.black_level) / self.white_level)) as u8
                    };
                self.canvas.set_pixel(self.hsync_sawtooth as i32,
                                      self.vsync_sawtooth as i32,
                                      Color::rgba(v, v, v, 255)).unwrap();
            }
        }
    }

    fn save(&mut self) {
        raster::save(&self.canvas, "test_render.png");
    }
}

fn main() {

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: composite_decode file_name");
        process::exit(1);
    }

    let mut file = File::open(args[1].clone()).unwrap();

    let mut proc = CompositeProcessor::new();

    let mut buffer: Vec<f64> = vec![];
    while let Ok(value) = file.read_f32::<LittleEndian>() {
        buffer.push(value as f64);
        if buffer.len() == CHUNK_SIZE {
            proc.process_block(&buffer);
            buffer.clear()
        }
    }
    proc.save();
}
