use std::simd::{LaneCount, Simd, SupportedLaneCount};

use simd_math::SimdFloatMath;

use crate::SimdComplex;

impl<const LANES: usize> SimdComplex<Simd<f64, LANES>>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn exp(self) -> Self {
        let r = self.re.exp();

        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::simd::Simd;

    use crate::{SimdComplex, tests::print_array};

    #[test]
    fn test_exp() {
        let zre = [
            0.7106566581109507,
            -0.5461187890421945,
            0.9116817311810632,
            0.7227996696442331,
            -0.5594680991682339,
            0.5446688682446102,
            -1.0676808758300795,
            1.807434548760676,
        ];

        let zim = [
            0.639525660421031,
            0.33703744521117146,
            -1.126644506861303,
            0.512151433227362,
            -0.17440497081298031,
            0.24196845022164443,
            0.4392278213915166,
            0.29229476570717267,
        ];

        let z = SimdComplex {
            re: Simd::from(zre),
            im: Simd::from(zim),
        };

        let expz = z.exp();

        print_array(&expz.re.to_array());
        print_array(&expz.im.to_array());
    }
}
