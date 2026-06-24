use core::array;

use crate::activation::{relu, relu6, FusedActivation};
use crate::ops_options::max_pool_2d::MaxPool2DOptions;
use crate::quantize::Quantized;
use crate::streaming_ops::stream_op::StreamOp;

pub struct StreamingMaxPool2D<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTER_ROWS: usize,
    const FILTER_COLS: usize,
    const BUF_SIZE: usize,
> {
    pub input_zero_point: T,
    pub output_scale: [f32; 1],
    pub output_zero_point: [T; 1],
    pub options: MaxPool2DOptions,
    pub buffer: [[T; INPUT_CHANS]; BUF_SIZE],
    pub write_idx: usize,
    pub shift_rows: usize,
    pub shift_cols: usize,
    pub in_cycle: usize,
    pub out_cycle: usize,
}

impl<
        T: Quantized,
        const INPUT_ROWS: usize,
        const INPUT_COLS: usize,
        const INPUT_CHANS: usize,
        const FILTER_ROWS: usize,
        const FILTER_COLS: usize,
        const BUF_SIZE: usize,
    >
    StreamingMaxPool2D<T, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, FILTER_ROWS, FILTER_COLS, BUF_SIZE>
{
    pub fn new(
        input_zero_point: T,
        output_scale: [f32; 1],
        output_zero_point: [T; 1],
        options: MaxPool2DOptions,
    ) -> Self {
        let (shift_rows, shift_cols) = match options.view_padding {
            crate::tensor::TensorViewPadding::Same => ((FILTER_ROWS - 1) / 2, (FILTER_COLS - 1) / 2),
            crate::tensor::TensorViewPadding::Valid => (0, 0),
        };
        Self {
            input_zero_point,
            output_scale,
            output_zero_point,
            options,
            buffer: [[input_zero_point; INPUT_CHANS]; BUF_SIZE],
            write_idx: 0,
            shift_rows,
            shift_cols,
            in_cycle: 0,
            out_cycle: 0,
        }
    }

    fn sample(&self, src_row: isize, src_col: isize) -> Option<[T; INPUT_CHANS]> {
        if src_col < 0
            || src_col >= INPUT_COLS as isize
            || src_row < 0
            || src_row >= INPUT_ROWS as isize
        {
            return None;
        }
        let target_cycle = (src_row as usize) * INPUT_COLS + (src_col as usize);
        let total_input_cycles = INPUT_ROWS.saturating_mul(INPUT_COLS);
        let current_cycle = self
            .in_cycle
            .saturating_sub(1)
            .min(total_input_cycles.saturating_sub(1));
        if target_cycle > current_cycle {
            return None;
        }
        let age = current_cycle - target_cycle;
        if age < BUF_SIZE {
            let idx = (self.write_idx + BUF_SIZE - 1 - age) % BUF_SIZE;
            Some(self.buffer[idx])
        } else {
            None
        }
    }

    fn compute(&self, out_row: usize, out_col: usize) -> [T; INPUT_CHANS] {
        let center_row = out_row * self.options.strides.0;
        let center_col = out_col * self.options.strides.1;

        let mut maxes = [self.input_zero_point; INPUT_CHANS];
        let mut seen = false;
        for kh in 0..FILTER_ROWS {
            for kw in 0..FILTER_COLS {
                let src_row = center_row as isize + kh as isize - self.shift_rows as isize;
                let src_col = center_col as isize + kw as isize - self.shift_cols as isize;
                if src_row >= 0
                    && src_row < INPUT_ROWS as isize
                    && src_col >= 0
                    && src_col < INPUT_COLS as isize
                {
                    let x = self
                        .sample(src_row, src_col)
                        .unwrap_or([self.input_zero_point; INPUT_CHANS]);
                    if !seen {
                        maxes = x;
                        seen = true;
                    } else {
                        for c in 0..INPUT_CHANS {
                            if x[c] > maxes[c] {
                                maxes[c] = x[c];
                            }
                        }
                    }
                }
            }
        }

        array::from_fn(|c| match self.options.fused_activation {
            FusedActivation::None => maxes[c],
            FusedActivation::Relu => relu(maxes[c], self.output_zero_point[0]),
            FusedActivation::Relu6 => relu6(maxes[c], self.output_scale[0], self.output_zero_point[0]),
        })
    }

    fn out_dims(&self) -> (usize, usize) {
        let out_cols = match self.options.view_padding {
            crate::tensor::TensorViewPadding::Same => INPUT_COLS.div_ceil(self.options.strides.1),
            crate::tensor::TensorViewPadding::Valid => {
                (INPUT_COLS.saturating_sub(FILTER_COLS)) / self.options.strides.1 + 1
            }
        };
        let out_rows = match self.options.view_padding {
            crate::tensor::TensorViewPadding::Same => INPUT_ROWS.div_ceil(self.options.strides.0),
            crate::tensor::TensorViewPadding::Valid => {
                (INPUT_ROWS.saturating_sub(FILTER_ROWS)) / self.options.strides.0 + 1
            }
        };
        (out_rows, out_cols)
    }
}

impl<
        T: Quantized,
        const INPUT_ROWS: usize,
        const INPUT_COLS: usize,
        const INPUT_CHANS: usize,
        const FILTER_ROWS: usize,
        const FILTER_COLS: usize,
        const BUF_SIZE: usize,
    > StreamOp<T, INPUT_CHANS, INPUT_CHANS>
    for StreamingMaxPool2D<T, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, FILTER_ROWS, FILTER_COLS, BUF_SIZE>
{
    #[inline(always)]
    fn is_finished(&self) -> bool {
        let (out_rows, out_cols) = self.out_dims();
        self.out_cycle >= out_rows * out_cols
    }

    #[inline(always)]
    fn push(&mut self, pixel: [T; INPUT_CHANS]) -> Option<[T; INPUT_CHANS]> {
        let in_row = self.in_cycle / INPUT_COLS;
        if in_row < INPUT_ROWS && BUF_SIZE > 0 {
            self.buffer[self.write_idx] = pixel;
            self.write_idx = (self.write_idx + 1) % BUF_SIZE;
        }
        self.in_cycle += 1;

        let (out_rows, out_cols) = self.out_dims();
        if self.out_cycle >= out_rows * out_cols {
            return None;
        }

        let out_row = self.out_cycle / out_cols;
        let out_col = self.out_cycle % out_cols;

        let req_in_row =
            out_row * self.options.strides.0 + FILTER_ROWS.saturating_sub(1).saturating_sub(self.shift_rows);
        let req_in_col =
            out_col * self.options.strides.1 + FILTER_COLS.saturating_sub(1).saturating_sub(self.shift_cols);
        let req_cycles = req_in_row * INPUT_COLS + req_in_col + 1;

        if self.in_cycle >= req_cycles {
            let emit = self.compute(out_row, out_col);
            self.out_cycle += 1;
            Some(emit)
        } else {
            None
        }
    }

    #[inline(always)]
    fn output_scale(&self) -> [f32; 1] {
        self.output_scale
    }

    #[inline(always)]
    fn output_zero_point(&self) -> [T; 1] {
        self.output_zero_point
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use super::*;
    use crate::streaming_ops::stream_pipeline::stream_pipeline;
    use crate::tensor::Tensor4D;
    use crate::tensor::TensorViewPadding;

    const INPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4],  [5,  6];
            [7, 8], [9, 10], [11, 12]
        ]],
        scale: [0.13],
        zero_point: [14],
    };
    const OUTPUT_SCALE: [f32; 1] = [0.13];
    const OUTPUT_ZERO_POINT: [i8; 1] = [14];
    const OPTIONS: MaxPool2DOptions = MaxPool2DOptions {
        fused_activation: FusedActivation::None,
        view_padding: TensorViewPadding::Valid,
        strides: (1, 1),
    };
    const OUTPUT: Tensor4D<i8, 1, 1, 2, 2, 1> = Tensor4D {
        buffer: [matrix![[9, 10], [11, 12]]],
        scale: [0.13],
        zero_point: [14],
    };

    #[test]
    fn max_pool_2d_layer() {
        let op: StreamingMaxPool2D<i8, 2, 3, 2, 2, 2, 5> = StreamingMaxPool2D::new(
            INPUT.zero_point[0],
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            OPTIONS,
        );
        let result = stream_pipeline::<i8, 2, 3, 2, 1, 2, 2, _>(&INPUT, op);
        assert_eq!(result, OUTPUT);
    }
}
