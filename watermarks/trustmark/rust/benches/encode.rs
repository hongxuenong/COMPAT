// Copyright 2025 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use criterion::{criterion_group, criterion_main, Criterion};
use trustmark::{Trustmark, Variant, Version};

fn encode_main(c: &mut Criterion) {
    let tm = Trustmark::new("./models", Variant::Q, Version::Bch5).unwrap();
    let input = image::open("../images/ufo_240.jpg").unwrap();
    c.bench_function("encode", |b| {
        b.iter(|| {
            let _ = tm.encode(
                "0100100100100001000101001010".to_owned(),
                input.clone(),
                0.95,
            );
        })
    });
}

criterion_group!(benches, encode_main);
criterion_main!(benches);
