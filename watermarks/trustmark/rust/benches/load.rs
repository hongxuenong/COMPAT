// Copyright 2025 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use criterion::{criterion_group, criterion_main, Criterion};
use trustmark::{Trustmark, Variant, Version};

fn load_main(c: &mut Criterion) {
    c.bench_function("load", |b| {
        b.iter(|| {
            let _tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
        })
    });
}

criterion_group!(benches, load_main);
criterion_main!(benches);
