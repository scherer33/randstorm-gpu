pub struct XorShift128Plus {
    s0: u64,
    s1: u64,
}

impl XorShift128Plus {
    pub fn new(seed: u64) -> Self {
        let mut s = XorShift128Plus { s0: 0, s1: 0 };
        s.seed(seed);
        s
    }

    fn seed(&mut self, seed: u64) {
        self.s0 = splitmix64(seed);
        self.s1 = splitmix64(self.s0);
        for _ in 0..10 {
            self.next_raw();
        }
    }

    fn next_raw(&mut self) -> u64 {
        let mut s1 = self.s0;
        let s0 = self.s1;
        self.s0 = s0;
        s1 ^= s1 << 23;
        s1 ^= s0;
        s1 ^= s1 >> 17;
        s1 ^= s0 >> 26;
        self.s1 = s1;
        s1.wrapping_add(s0)
    }

    pub fn next_math_random_u16(&mut self) -> u16 {
        let sum = self.next_raw();
        (sum >> 37) as u16
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
