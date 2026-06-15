use core::array;

use nalgebra::Const;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::ops_options::max_pool_2d::MaxPool2DOptions;
use crate::quantize::Quantized;
use crate::tensor::{Tensor4D, TensorView};

/// Performs the MaxPool2D operation. Channel-wise max over the filter window.
pub fn max_pool_2d<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTER_ROWS: usize,
    const FILTER_COLS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    _filter_shape: (Const<FILTER_ROWS>, Const<FILTER_COLS>),
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: MaxPool2DOptions,
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, INPUT_CHANS, 1> {
    let output = [Buffer2D::from_fn(|i, j| {
        let view: TensorView<T, FILTER_ROWS, FILTER_COLS, INPUT_CHANS> =
            input.view((i, j), 0, options.view_padding, options.strides);
        array::from_fn(|c| {
            let y = view
                .buffer
                .iter()
                .map(|a| a[c])
                .max()
                .unwrap_or(input.zero_point[0]);
            match options.fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, output_scale[0], output_zero_point[0]),
            }
        })
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}
