use core::array;
use libm::roundf;

use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::quantize::Quantized;
use crate::ops_options::average_pool_2d::AveragePool2DOptions;
use crate::streaming_ops::stream_op::StreamOp;

pub struct StreamingAveragePool2D<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTER_ROWS: usize,
    const FILTER_COLS: usize,
> {
    pub input_zero_point: T,
    pub output_scale: [f32; 1],
    pub output_zero_point: [T; 1],
    pub options: AveragePool2DOptions,
    pub constants: (f32, f32),
    pub row_head: usize,
    pub rows: [[[T; INPUT_CHANS]; INPUT_COLS]; FILTER_ROWS],
    
    // Precomputed offsets
    pub shift_rows: usize,
    pub shift_cols: usize,
    
    // Decoupled hardware clocks
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
> StreamingAveragePool2D<T, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, FILTER_ROWS, FILTER_COLS> {
    pub fn new(
        input_zero_point: T,
        output_scale: [f32; 1],
        output_zero_point: [T; 1],
        options: AveragePool2DOptions,
        constants: (f32, f32),
    ) -> Self {
        let (shift_rows, shift_cols) = match options.view_padding {
            crate::tensor::TensorViewPadding::Same => {
                ((FILTER_ROWS - 1) / 2, (FILTER_COLS - 1) / 2)
            }
            crate::tensor::TensorViewPadding::Valid => (0, 0),
        };

        Self {
            input_zero_point,
            output_scale,
            output_zero_point,
            options,
            constants,
            row_head: 0,
            rows: [[[input_zero_point; INPUT_CHANS]; INPUT_COLS]; FILTER_ROWS],
            shift_rows,
            shift_cols,
            in_cycle: 0,
            out_cycle: 0,
        }
    }

    fn sample(&self, src_row: isize, src_col: isize) -> Option<[T; INPUT_CHANS]> {
        if src_col < 0 || src_col >= INPUT_COLS as isize || src_row < 0 || src_row >= INPUT_ROWS as isize {
            return None; // Out of bounds pixels are simply ignored in AveragePool
        }

        let src_row = src_row as usize;
        let src_col = src_col as usize;

        let in_row = (self.in_cycle.saturating_sub(1)) / INPUT_COLS; 
        let active_row = in_row.min(INPUT_ROWS - 1);

        if src_row > active_row {
            return None;
        }

        let delta = active_row - src_row;
        if delta >= FILTER_ROWS {
            return None;
        }

        let row_idx = (self.row_head + FILTER_ROWS - delta) % FILTER_ROWS;
        Some(self.rows[row_idx][src_col])
    }

    fn compute(&self, out_row: usize, out_col: usize) -> [T; INPUT_CHANS] {
        let mut sums = [0i32; INPUT_CHANS];
        let mut valid_count = 0usize;

        let center_row = out_row * self.options.strides.0;
        let center_col = out_col * self.options.strides.1;

        // Loop over the pooling window
        for kh in 0..FILTER_ROWS {
            for kw in 0..FILTER_COLS {
                let src_row = center_row as isize + kh as isize - self.shift_rows as isize;
                let src_col = center_col as isize + kw as isize - self.shift_cols as isize;

                // TFLite Only averages over valid pixels!
                if src_row >= 0 && src_row < INPUT_ROWS as isize && src_col >= 0 && src_col < INPUT_COLS as isize {
                    valid_count += 1;
                    let x = self.sample(src_row, src_col).unwrap_or([self.input_zero_point; INPUT_CHANS]);
                    for c in 0..INPUT_CHANS {
                        sums[c] += i32::from_subset(&x[c]);
                    }
                }
            }
        }

        let inv_len = 1.0 / (valid_count as f32);

        array::from_fn(|c| {
            // Apply TFLite AveragePool math
            let x = inv_len * (sums[c] as f32);
            let y = T::from_superset_unchecked(&roundf(self.constants.0 * x + self.constants.1));

            match self.options.fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, self.output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, self.output_scale[0], self.output_zero_point[0]),
            }
        })
    }
}

impl<
        T: Quantized,
        const INPUT_ROWS: usize,
        const INPUT_COLS: usize,
        const INPUT_CHANS: usize,
        const FILTER_ROWS: usize,
        const FILTER_COLS: usize,
    > StreamOp<T, INPUT_CHANS, INPUT_CHANS>
    for StreamingAveragePool2D<T, INPUT_ROWS, INPUT_COLS, INPUT_CHANS, FILTER_ROWS, FILTER_COLS>
{
    #[inline(always)]
    fn push(&mut self, pixel: [T; INPUT_CHANS]) -> Option<[T; INPUT_CHANS]> {
        let in_row = self.in_cycle / INPUT_COLS;
        let in_col = self.in_cycle % INPUT_COLS;

        // 1. Advance ring buffer on new row (preserving data for right-edge math)
        if in_col == 0 && in_row > 0 && in_row < INPUT_ROWS {
            self.row_head = (self.row_head + 1) % FILTER_ROWS;
        }

        // 2. Write pixel
        if in_row < INPUT_ROWS {
            self.rows[self.row_head][in_col] = pixel;
        }

        self.in_cycle += 1;

        // 3. Determine Output Dimension Targets
        let out_cols = match self.options.view_padding {
            crate::tensor::TensorViewPadding::Same => (INPUT_COLS + self.options.strides.1 - 1) / self.options.strides.1,
            crate::tensor::TensorViewPadding::Valid => (INPUT_COLS.saturating_sub(FILTER_COLS)) / self.options.strides.1 + 1,
        };
        let out_rows = match self.options.view_padding {
            crate::tensor::TensorViewPadding::Same => (INPUT_ROWS + self.options.strides.0 - 1) / self.options.strides.0,
            crate::tensor::TensorViewPadding::Valid => (INPUT_ROWS.saturating_sub(FILTER_ROWS)) / self.options.strides.0 + 1,
        };

        if self.out_cycle >= out_rows * out_cols {
            return None; 
        }

        let out_row = self.out_cycle / out_cols;
        let out_col = self.out_cycle % out_cols;

        // 4. Calculate required clock cycle for this output
        let req_in_row = out_row * self.options.strides.0 + FILTER_ROWS.saturating_sub(1).saturating_sub(self.shift_rows);
        let req_in_col = out_col * self.options.strides.1 + FILTER_COLS.saturating_sub(1).saturating_sub(self.shift_cols);
        let req_cycles = req_in_row * INPUT_COLS + req_in_col + 1;

        // 5. Emit
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
    use crate::tensor::TensorViewPadding;
    use crate::streaming_ops::stream_pipeline::stream_pipeline;
    use crate::tensor::Tensor4D;

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
    const OPTIONS: AveragePool2DOptions = AveragePool2DOptions {
        fused_activation: FusedActivation::None,
        view_padding: TensorViewPadding::Same,
        strides: (1, 1),
    };
    const CONSTANTS: (f32, f32) = (0.866_666_7, 3.866_666_6);
    const OUTPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [8,  9],  [9,  10], [10, 11];
            [11, 12], [12, 13], [13, 13]
        ]],
        scale: [0.15],
        zero_point: [16],
    };

    #[test]
    fn average_pool_2d_layer() {
        let op: StreamingAveragePool2D<i8, 2, 3, 2, 2, 3> = StreamingAveragePool2D::new(
            INPUT.zero_point[0],
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            OPTIONS,
            CONSTANTS,
        );
        let result = stream_pipeline::<
            i8,
            2, // INPUT_ROWS
            3, // INPUT_COLS
            2, // INPUT_CHANS
            2, // OUTPUT_ROWS
            3, // OUTPUT_COLS
            2, // OUTPUT_CHANS
            _
        >(INPUT, op);
        assert_eq!(result, OUTPUT);
    }
}
