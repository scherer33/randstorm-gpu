pub struct ArcFour {
    i: u8,
    j: u8,
    s: [u8; 256],
}

impl ArcFour {
    pub fn init(key: &[u8]) -> Self {
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
        SecureRandom { arc4: ArcFour::init(&p) }
    }

    pub fn get_byte(&mut self) -> u8 {
        self.arc4.next()
    }

    pub fn get_bytes(&mut self, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            *b = self.get_byte();
        }
    }
}
