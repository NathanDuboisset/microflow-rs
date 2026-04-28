use core::array;
use libm::roundf;

use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;

/// Performs the GlobalAveragePool2D operation.
/// Returns a [1, 1, 1, CH] tensor containing the result.
pub fn global_average_pool_2d<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    fused_activation: FusedActivation,
    constants: (f32, f32),
) -> Tensor4D<T, 1, 1, 1, INPUT_CHANS, 1> {
    let area = (INPUT_ROWS * INPUT_COLS) as f32;
    let output = [Buffer2D::from_fn(|_, _| {
        array::from_fn(|c| {
            let sum = input.buffer[0]
                .iter()
                .fold(0i32, |acc, e| acc + i32::from_subset(&e[c]));
            let x = sum as f32 / area;
            let y = T::from_superset_unchecked(&roundf(constants.0 * x + constants.1));
            match fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, output_scale[0], output_zero_point[0]),
            }
        })
    })];
    Tensor4D::new(output, output_scale, output_zero_point)
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;

    const INPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4],  [5,  6];
            [7, 8], [9, 10], [11, 12]
        ]],
        scale: [0.13],
        zero_point: [14],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.15];
    const OUTPUT_ZERO_POINT: [i8; 1] = [16];
    const CONSTANTS: (f32, f32) = (0.866_666_7, 3.866_666_6);
    const OUTPUT: Tensor4D<i8, 1, 1, 1, 2, 1> = Tensor4D {
        buffer: [matrix![[9, 10]]],
        scale: [0.15],
        zero_point: [16],
    };

    #[test]
    fn global_average_pool_2d_layer() {
        assert_eq!(
            global_average_pool_2d(
                INPUT,
                OUTPUT_SCALE,
                OUTPUT_ZERO_POINT,
                FusedActivation::None,
                CONSTANTS,
            ),
            OUTPUT
        );
    }
}
