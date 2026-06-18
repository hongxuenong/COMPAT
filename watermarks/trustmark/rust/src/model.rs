// Copyright 2025 Adobe
// All Rights Reserved.
//
// NOTICE: Adobe permits you to use, modify, and distribute this file in
// accordance with the terms of the Adobe license agreement accompanying
// it.

use std::{fmt::Display, str::FromStr};

use crate::Error;

/// The model variant to load.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Variant {
    B,
    C,
    P,
    Q,
}

impl Variant {
    pub(super) fn encoder_filename(&self) -> String {
        let suffix = match self {
            Variant::B => "B",
            Variant::C => "C",
            Variant::P => "P",
            Variant::Q => "Q",
        };

        format!("encoder_{suffix}.onnx")
    }

    pub(super) fn decoder_filename(&self) -> String {
        let suffix = match self {
            Variant::B => "B",
            Variant::C => "C",
            Variant::P => "P",
            Variant::Q => "Q",
        };

        format!("decoder_{suffix}.onnx")
    }

    pub(super) fn strength_multiplier(&self) -> f32 {
        match self {
            Variant::P => 1.25,
            _ => 1.,
        }
    }
}

impl Display for Variant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Variant::B => "B",
            Variant::C => "C",
            Variant::P => "P",
            Variant::Q => "Q",
        };

        f.write_str(s)
    }
}

impl FromStr for Variant {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "B" => Variant::B,
            "C" => Variant::C,
            "P" => Variant::P,
            "Q" => Variant::Q,
            _ => return Err(Error::InvalidModelVariant),
        })
    }
}
