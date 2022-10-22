#![feature(portable_simd, trait_alias)]
use std::{
    ops::{
        Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
    },
    simd::{LaneCount, Mask, Simd, SimdFloat, StdFloat, SupportedLaneCount},
};

mod simd_cmath;

pub trait FV = SimdFloat
    + StdFloat
    + Add<Output = Self>
    + AddAssign
    + Neg<Output = Self>
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign;

#[derive(Debug, Clone, Copy)]
pub struct SimdComplex<T: FV> {
    pub re: T,
    pub im: T,
}

pub trait SimdComplexSelect<T: FV> {
    fn cselect(
        self,
        trues: SimdComplex<T>,
        falses: SimdComplex<T>,
    ) -> SimdComplex<T>;
}

impl<const LANES: usize> SimdComplex<Simd<f32, LANES>>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    pub fn splat((re, im): (f32, f32)) -> Self {
        Self {
            re: Simd::splat(re),
            im: Simd::splat(im),
        }
    }
}

impl<const LANES: usize> SimdComplex<Simd<f64, LANES>>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    pub fn splat((re, im): (f64, f64)) -> Self {
        Self {
            re: Simd::splat(re),
            im: Simd::splat(im),
        }
    }
}

impl<T: FV> SimdComplex<T> {
    #[inline(always)]
    pub fn mul_add(self, a: Self, b: Self) -> Self {
        Self {
            re: self.im.mul_add(-a.im, self.re.mul_add(a.re, b.re)),
            im: self.re.mul_add(a.im, self.im.mul_add(a.re, b.im)),
        }
    }

    #[inline(always)]
    pub fn abssqr(self) -> T {
        self.im.mul_add(self.im, self.re * self.re)
    }

    #[inline(always)]
    pub fn abs(self) -> T {
        self.abssqr().sqrt()
    }
}

impl<T: FV> Add for SimdComplex<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: FV> Add<T> for SimdComplex<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: T) -> Self::Output {
        Self {
            re: self.re + rhs,
            im: self.im,
        }
    }
}

impl<T: FV> AddAssign for SimdComplex<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: FV> AddAssign<T> for SimdComplex<T> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<T: FV> Neg for SimdComplex<T> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T: FV> Sub for SimdComplex<T> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T: FV> Sub<T> for SimdComplex<T> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: T) -> Self::Output {
        Self {
            re: self.re - rhs,
            im: self.im,
        }
    }
}

impl<T: FV> SubAssign for SimdComplex<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: FV> SubAssign<T> for SimdComplex<T> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}

impl<T: FV> Mul for SimdComplex<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.im.mul_add(-rhs.im, self.re * rhs.re),
            im: self.im.mul_add(rhs.re, self.re * rhs.im),
        }
    }
}

impl<T: FV> Mul<T> for SimdComplex<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl<T: FV> MulAssign for SimdComplex<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: FV> MulAssign<T> for SimdComplex<T> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: FV> Div for SimdComplex<T> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.abssqr().recip();

        Self {
            re: self.im.mul_add(rhs.im, self.re * rhs.re) * denom,
            im: self.re.mul_add(-rhs.im, self.im * rhs.re) * denom,
        }
    }
}

impl<T: FV> Div<T> for SimdComplex<T> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: T) -> Self::Output {
        let r = rhs.recip();

        Self {
            re: self.re * r,
            im: self.im * r,
        }
    }
}

impl<T: FV> DivAssign for SimdComplex<T> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: FV> DivAssign<T> for SimdComplex<T> {
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<const LANES: usize> SimdComplexSelect<Simd<f64, LANES>>
    for Mask<i64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    fn cselect(
        self,
        trues: SimdComplex<Simd<f64, LANES>>,
        falses: SimdComplex<Simd<f64, LANES>>,
    ) -> SimdComplex<Simd<f64, LANES>> {
        SimdComplex {
            re: self.select(trues.re, falses.re),
            im: self.select(trues.im, falses.im),
        }
    }
}

impl<const LANES: usize> Add<SimdComplex<Simd<f64, LANES>>> for Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = SimdComplex<Simd<f64, LANES>>;

    #[inline(always)]
    fn add(self, rhs: SimdComplex<Simd<f64, LANES>>) -> Self::Output {
        rhs + self
    }
}

impl<const LANES: usize> Sub<SimdComplex<Simd<f64, LANES>>> for Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = SimdComplex<Simd<f64, LANES>>;

    #[inline(always)]
    fn sub(self, rhs: SimdComplex<Simd<f64, LANES>>) -> Self::Output {
        SimdComplex {
            re: self - rhs.re,
            im: -rhs.im,
        }
    }
}

impl<const LANES: usize> Mul<SimdComplex<Simd<f64, LANES>>> for Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = SimdComplex<Simd<f64, LANES>>;

    #[inline(always)]
    fn mul(self, rhs: SimdComplex<Simd<f64, LANES>>) -> Self::Output {
        rhs * self
    }
}

impl<const LANES: usize> Div<SimdComplex<Simd<f64, LANES>>> for Simd<f64, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = SimdComplex<Simd<f64, LANES>>;

    #[inline(always)]
    fn div(self, rhs: SimdComplex<Simd<f64, LANES>>) -> Self::Output {
        let denom = rhs.abssqr().recip();

        SimdComplex {
            re: self * rhs.re * denom,
            im: -self * rhs.im * denom,
        }
    }
}

impl<const LANES: usize> SimdComplexSelect<Simd<f32, LANES>>
    for Mask<i32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    fn cselect(
        self,
        trues: SimdComplex<Simd<f32, LANES>>,
        falses: SimdComplex<Simd<f32, LANES>>,
    ) -> SimdComplex<Simd<f32, LANES>> {
        SimdComplex {
            re: self.select(trues.re, falses.re),
            im: self.select(trues.im, falses.im),
        }
    }
}

impl<const LANES: usize> Add<SimdComplex<Simd<f32, LANES>>> for Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = SimdComplex<Simd<f32, LANES>>;

    #[inline(always)]
    fn add(self, rhs: SimdComplex<Simd<f32, LANES>>) -> Self::Output {
        rhs + self
    }
}

impl<const LANES: usize> Sub<SimdComplex<Simd<f32, LANES>>> for Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = SimdComplex<Simd<f32, LANES>>;

    #[inline(always)]
    fn sub(self, rhs: SimdComplex<Simd<f32, LANES>>) -> Self::Output {
        SimdComplex {
            re: self - rhs.re,
            im: -rhs.im,
        }
    }
}

impl<const LANES: usize> Mul<SimdComplex<Simd<f32, LANES>>> for Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = SimdComplex<Simd<f32, LANES>>;

    #[inline(always)]
    fn mul(self, rhs: SimdComplex<Simd<f32, LANES>>) -> Self::Output {
        rhs * self
    }
}

impl<const LANES: usize> Div<SimdComplex<Simd<f32, LANES>>> for Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = SimdComplex<Simd<f32, LANES>>;

    #[inline(always)]
    fn div(self, rhs: SimdComplex<Simd<f32, LANES>>) -> Self::Output {
        let denom = rhs.abssqr().recip();

        SimdComplex {
            re: self * rhs.re * denom,
            im: -self * rhs.im * denom,
        }
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
