use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::tensor::Tensor4D;

use crate::streaming_ops::stream_op::StreamOp;

/// Executes a streaming pipeline over a full tensor
pub fn stream_pipeline<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const OUTPUT_ROWS: usize,
    const OUTPUT_COLS: usize,
    const OUTPUT_CHANS: usize,
    OP: StreamOp<T, INPUT_CHANS, OUTPUT_CHANS>,
>(
    input: Tensor4D<T, 1, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, 1>,
    mut op: OP,
) -> Tensor4D<T, 1, OUTPUT_ROWS, OUTPUT_COLS, OUTPUT_CHANS, 1> {
    let mut out_row = 0usize;
    let mut out_col = 0usize;

    let mut output = [Buffer2D::from_fn(|_, _| {
        [input.zero_point[0]; OUTPUT_CHANS]
    })];

    for i in 0..INPUT_ROWS {
        for j in 0..INPUT_COLS {
            let pixel = input.buffer[0][(i, j)];

            if let Some(y) = op.push(pixel) {
                output[0][(out_row, out_col)] = y;

                out_col += 1;
                if out_col == OUTPUT_COLS {
                    out_col = 0;
                    out_row += 1;
                }
            }
        }
    }

    Tensor4D::new(output, input.scale, input.zero_point)
}