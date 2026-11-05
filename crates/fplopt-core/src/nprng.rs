//! numpy's `np.random.default_rng(seed)`: PCG64 seeded through `SeedSequence`, and the
//! `Generator` methods the backtester uses (`integers`, `choice`, `permutation`, `random`),
//! draw for draw (numpy's `_pcg64`, `_seed_sequence`, `distributions.c` and `_generator`), so
//! random start states and bootstrap resamples match Python's exactly. Golden:
//! `tests/fixtures/rust_golden/nprng.json.gz`.

const MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

/// `SeedSequence(seed).generate_state(n_words)` (uint32 words).
fn seed_sequence_state(seed: u128, n_words: usize) -> Vec<u32> {
    // _int_to_uint32_array: little-endian 32-bit words, [0] for 0.
    let mut entropy = Vec::new();
    let mut n = seed;
    loop {
        entropy.push(n as u32);
        n >>= 32;
        if n == 0 {
            break;
        }
    }
    let mut hash_const = INIT_A;
    let mut hashmix = |value: u32| {
        let mut value = value ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        value = value.wrapping_mul(hash_const);
        value ^ (value >> XSHIFT)
    };
    let mix = |x: u32, y: u32| {
        let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
        r ^ (r >> XSHIFT)
    };
    let mut pool = [0u32; POOL_SIZE];
    for (i, slot) in pool.iter_mut().enumerate() {
        *slot = hashmix(entropy.get(i).copied().unwrap_or(0));
    }
    for src in 0..POOL_SIZE {
        for dst in 0..POOL_SIZE {
            if src != dst {
                let h = hashmix(pool[src]);
                pool[dst] = mix(pool[dst], h);
            }
        }
    }
    for &word in entropy.iter().skip(POOL_SIZE) {
        for slot in pool.iter_mut() {
            let h = hashmix(word);
            *slot = mix(*slot, h);
        }
    }
    let mut hash_const = INIT_B;
    (0..n_words)
        .map(|i| {
            let mut value = pool[i % POOL_SIZE] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash_const);
            value ^ (value >> XSHIFT)
        })
        .collect()
}

/// `np.random.Generator(PCG64(seed))`.
#[derive(Clone, Debug)]
pub struct Generator {
    state: u128,
    inc: u128,
    /// The upper half of the last 64-bit draw, kept for the next 32-bit draw.
    pending: Option<u32>,
}

impl Generator {
    /// `np.random.default_rng(seed)` for a non-negative integer seed.
    pub fn new(seed: u128) -> Generator {
        let w = seed_sequence_state(seed, 8);
        // generate_state(4, uint64): pairs of words, little endian.
        let words: Vec<u64> = w.chunks(2).map(|p| p[0] as u64 | ((p[1] as u64) << 32)).collect();
        let initstate = ((words[0] as u128) << 64) | words[1] as u128;
        let initseq = ((words[2] as u128) << 64) | words[3] as u128;
        let mut g = Generator { state: 0, inc: (initseq << 1) | 1, pending: None };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc);
    }

    /// `next_uint64`: XSL-RR output of the stepped state.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let (hi, lo) = ((self.state >> 64) as u64, self.state as u64);
        (hi ^ lo).rotate_right((hi >> 58) as u32)
    }

    /// `next_uint32`: the halves of a 64-bit draw, low first.
    pub fn next_u32(&mut self) -> u32 {
        if let Some(high) = self.pending.take() {
            return high;
        }
        let next = self.next_u64();
        self.pending = Some((next >> 32) as u32);
        next as u32
    }

    /// `random()`: a double in [0, 1).
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// One `integers(low, high)` draw (high exclusive; `random_bounded_uint64_fill`, Lemire's
    /// method without masking).
    pub fn integer(&mut self, low: i64, high: i64) -> i64 {
        assert!(high > low, "integers: low >= high");
        let rng = (high - low - 1) as u64;
        let draw = if rng == 0 {
            0
        } else if rng < 0xFFFF_FFFF {
            self.lemire32(rng as u32) as u64
        } else if rng == 0xFFFF_FFFF {
            self.next_u32() as u64
        } else if rng == u64::MAX {
            self.next_u64()
        } else {
            self.lemire64(rng)
        };
        low.wrapping_add(draw as i64)
    }

    /// `integers(low, high, size=n)`, row-major for a shaped size.
    pub fn integers(&mut self, low: i64, high: i64, n: usize) -> Vec<i64> {
        (0..n).map(|_| self.integer(low, high)).collect()
    }

    fn lemire32(&mut self, rng: u32) -> u32 {
        let excl = rng + 1;
        let mut m = self.next_u32() as u64 * excl as u64;
        let mut leftover = m as u32;
        if leftover < excl {
            let threshold = (u32::MAX - rng) % excl;
            while leftover < threshold {
                m = self.next_u32() as u64 * excl as u64;
                leftover = m as u32;
            }
        }
        (m >> 32) as u32
    }

    fn lemire64(&mut self, rng: u64) -> u64 {
        let excl = rng + 1;
        let mut m = self.next_u64() as u128 * excl as u128;
        let mut leftover = m as u64;
        if leftover < excl {
            let threshold = (u64::MAX - rng) % excl;
            while leftover < threshold {
                m = self.next_u64() as u128 * excl as u128;
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// `random_interval(max)`: a value in [0, max] by masked rejection.
    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for shift in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> shift;
        }
        loop {
            let value = if max <= 0xFFFF_FFFF {
                self.next_u32() as u64 & mask
            } else {
                self.next_u64() & mask
            };
            if value <= max {
                return value;
            }
        }
    }

    /// `permutation(n)`: `arange(n)` shuffled (Fisher–Yates from the end).
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut out: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.interval(i as u64) as usize;
            out.swap(i, j);
        }
        out
    }

    /// `choice(n, size=k, replace=True)` without `p`: indices.
    pub fn choice_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        (0..k).map(|_| self.integer(0, n as i64) as usize).collect()
    }

    /// One `choice(len(p), p=p)` draw: `cdf = p.cumsum(); cdf /= cdf[-1]`, then the first
    /// index whose cdf exceeds a `random()` draw (`searchsorted(side="right")`).
    pub fn choice_weighted(&mut self, p: &[f64]) -> usize {
        let mut cdf = Vec::with_capacity(p.len());
        let mut acc = 0.0;
        for &x in p {
            acc += x;
            cdf.push(acc);
        }
        let last = *cdf.last().expect("choice: empty p");
        for c in &mut cdf {
            *c /= last;
        }
        let u = self.random();
        cdf.partition_point(|&c| c <= u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_sequence_matches_numpy() {
        // np.random.SeedSequence(0).generate_state(4)
        assert_eq!(
            seed_sequence_state(0, 4),
            [2_968_811_710, 3_677_149_159, 745_650_761, 2_884_920_346]
        );
    }

    #[test]
    fn first_draws_match_numpy() {
        // np.random.default_rng(0).bit_generator.random_raw(2)
        let mut g = Generator::new(0);
        assert_eq!(g.next_u64(), 11_749_869_230_777_074_271);
        assert_eq!(g.next_u64(), 4_976_686_463_289_251_617);
    }
}
