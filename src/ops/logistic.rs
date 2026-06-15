use crate::quantize::{dequantize, quantize, Quantized};
use crate::tensor::Tensor2D;
use libm::expf;

/// Performs the Logistic (sigmoid) activation function as an operator.
/// Returns a 2-dimensional output tensor containing the result of the operation.
pub fn logistic<T: Quantized, const ROWS: usize, const COLS: usize>(
    input: Tensor2D<T, ROWS, COLS, 1>,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
) -> Tensor2D<T, ROWS, COLS, 1> {
    Tensor2D::new(
        input.buffer.map(|e| {
            let x = dequantize(e, input.scale[0], input.zero_point[0]);
            let s = 1.0 / (1.0 + expf(-x));
            quantize(s, output_scale[0], output_zero_point[0])
        }),
        output_scale,
        output_zero_point,
    )
}
