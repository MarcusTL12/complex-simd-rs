use std::simd::{Simd, LaneCount, SupportedLaneCount};

use simd_math::SimdFloatMath;

use crate::SimdComplex;

impl<const LANES: usize> SimdComplex<Simd<f64, LANES>> where
    LaneCount<LANES>: SupportedLaneCount
{
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        
        Self { re: r * self.im.cos(), im: r * self.im.sin() }
    }
}

#[cfg(test)]
pub mod tests {
    pub fn print_array(a: &[f64]) {
        print!("[");
        let mut first = true;
        for x in a {
            print!("{}{:9.2e}", if first { "" } else { ", " }, x);
            first = false;
        }
        println!("]");
    }
}