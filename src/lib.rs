//! [![crates.io](https://img.shields.io/crates/v/microflow)](https://crates.io/crates/microflow)
//! [![docs.rs](https://img.shields.io/docsrs/microflow)](https://docs.rs/microflow)
//! [![github](https://img.shields.io/github/actions/workflow/status/matteocarnelos/microflow-rs/cargo.yml?branch=main)](https://github.com/matteocarnelos/microflow-rs/actions/workflows/cargo.yml)
//!
//! A robust and efficient TinyML inference engine for embedded systems.

#![no_std]

pub use microflow_macros::*;

pub mod activation;
pub mod buffer;
pub mod ops;
pub mod quantize;
pub mod tensor;

// Per-layer timing via ariel-os (enable with `features = ["ariel-os"]` on this crate).
// `::ariel_os::*` is resolved in the consumer crate, not here.

#[cfg(feature = "ariel-os")]
#[macro_export]
#[doc(hidden)]
macro_rules! __layer_start {
    () => {{
        ::ariel_os::time::Instant::now().as_micros()
    }};
}

#[cfg(not(feature = "ariel-os"))]
#[macro_export]
#[doc(hidden)]
macro_rules! __layer_start {
    () => {{}};
}

#[cfg(feature = "ariel-os")]
#[macro_export]
#[doc(hidden)]
macro_rules! __layer_end {
    ($name:expr, $index:expr, $start:expr) => {{
        let __end = ::ariel_os::time::Instant::now().as_micros();
        let __start: u64 = $start;
        ::ariel_os::debug::log::info!(
            "layer {}: {} took {} us",
            $index as u32,
            $name,
            __end.saturating_sub(__start)
        );
    }};
}

#[cfg(not(feature = "ariel-os"))]
#[macro_export]
#[doc(hidden)]
macro_rules! __layer_end {
    ($name:expr, $index:expr, $start:expr) => {{}};
}
