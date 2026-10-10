pub struct XorShift128Plus {
    s0: u64,
    s1: u64,
}

impl XorShift128Plus {
    pub fn new(seed: u64) -> Self {
        let mut s = Self { s0: 0, s1: 0 };
        s.seed(seed);
        s
    }

    fn seed(&mut self, seed: u64) {
        self.s0 = Self::splitmix(seed);
        self.s1 = Self::splitmix(self.s0);
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

    pub fn next_u16(&mut self) -> u16 {
        (self.next_raw() >> 37) as u16
    }

    fn splitmix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
}

pub struct ArcFour {
    i: u8,
    j: u8,
    s: [u8; 256],
}

impl ArcFour {
    pub fn new(key: &[u8]) -> Self {
        let mut s = [0u8; 256];
        for (i, v) in s.iter_mut().enumerate() {
            *v = i as u8;
        }
        let mut j = 0u8;
        for i in 0..256u16 {
            j = j.wrapping_add(s[i as usize]).wrapping_add(key[i as usize % key.len()]);
            s.swap(i as usize, j as usize);
        }
        ArcFour { i: 0, j: 0, s }
    }

    pub fn next(&mut self) -> u8 {
        self.i = self.i.wrapping_add(1);
        self.j = self.j.wrapping_add(self.s[self.i as usize]);
        self.s.swap(self.i as usize, self.j as usize);
        let idx = self.s[self.i as usize].wrapping_add(self.s[self.j as usize]);
        self.s[idx as usize]
    }
}

pub struct SecureRandom {
    arc4: ArcFour,
}

impl SecureRandom {
    pub fn from_pool(pool: &[u8; 256], seed_time: u64) -> Self {
        let mut p = *pool;
        p[0] ^= seed_time as u8;
        p[1] ^= (seed_time >> 8) as u8;
        p[2] ^= (seed_time >> 16) as u8;
        p[3] ^= (seed_time >> 24) as u8;
        SecureRandom {
            arc4: ArcFour::new(&p),
        }
    }

    pub fn get_bytes(&mut self, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            *b = self.arc4.next();
        }
    }
}
