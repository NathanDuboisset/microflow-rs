use core::array;
use libm::roundf;

use simba::scalar::SupersetOf;


use crate::activation::{relu, relu6, FusedActivation};
use crate::buffer::Buffer2D;
use crate::quantize::Quantized;
use crate::streaming_ops::stream_op::StreamOp;
use crate::tensor::Tensor4D;

use crate::ops_options::conv_2d::Conv2DOptions;

/// Performs the Conv2D operation with streaming.
/// When pushing a data point, return optionally a result, 
/// corresponding to the computation of a resulting data point.
///
/// # Arguments
/// * `input` - The 4-dimensional input tensor
/// * `filters` - The 4-dimensional tensor representing the filters of the operator
/// * `output_scale` - The scale of the resulting output tensor
/// * `output_zero_point` - The zero point of the resulting output tensor
/// * `options` - Operator's options as an [`Conv2DOptions`] struct
/// * `constants` - Constant values coming from the pre-processing phase
///

pub struct StreamingConv2D<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
    const FILTERS_BATCHES: usize,
    const FILTERS_ROWS: usize,
    const FILTERS_COLS: usize,
    const FILTERS_QUANTS: usize,
    const BUF_SIZE: usize,
> {
    pub buffer: [[T; INPUT_CHANS]; BUF_SIZE],
    pub write_idx: usize,

    input_zero_point: T,
    filters: Tensor4D<
        T,
        FILTERS_BATCHES,
        FILTERS_ROWS,
        FILTERS_COLS,
        INPUT_CHANS,
        FILTERS_QUANTS,
    >,
    output_scale: [f32; 1],
    output_zero_point: [T; 1],
    options: Conv2DOptions,
    constants: (
        Buffer2D<f32, FILTERS_BATCHES, 1>,
        Buffer2D<f32, FILTERS_QUANTS, 1>,
    ),
    filter_sums: [i32; FILTERS_BATCHES],

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
        const FILTERS_BATCHES: usize,
        const FILTERS_ROWS: usize,
        const FILTERS_COLS: usize,
        const FILTERS_QUANTS: usize,
        const BUF_SIZE: usize,
    > StreamingConv2D<
        T,
        INPUT_ROWS,
        INPUT_COLS,
        INPUT_CHANS,
        FILTERS_BATCHES,
        FILTERS_ROWS,
        FILTERS_COLS,
        FILTERS_QUANTS,
        BUF_SIZE,
    >
{
    pub fn new(
        input_zero_point: T,
        filters: Tensor4D<
            T,
            FILTERS_BATCHES,
            FILTERS_ROWS,
            FILTERS_COLS,
            INPUT_CHANS,
            FILTERS_QUANTS,
        >,
        output_scale: [f32; 1],
        output_zero_point: [T; 1],
        options: Conv2DOptions,
        constants: (
            Buffer2D<f32, FILTERS_BATCHES, 1>,
            Buffer2D<f32, FILTERS_QUANTS, 1>,
        ),
    ) -> Self {
        let filter_sums = array::from_fn(|b| {
            filters.buffer[b].iter().fold(0i32, |acc, row| {
                acc + row
                    .iter()
                    .fold(0i32, |acc, e| acc + i32::from_subset(e))
            })
        });

        let (shift_rows, shift_cols) = match options.view_padding {
            crate::tensor::TensorViewPadding::Same => {
                ((FILTERS_ROWS - 1) / 2, (FILTERS_COLS - 1) / 2)
            }
            crate::tensor::TensorViewPadding::Valid => (0, 0),
        };

        Self {
            buffer: [[input_zero_point; INPUT_CHANS]; BUF_SIZE],
            write_idx: 0,
            input_zero_point,
            filters,
            output_scale,
            output_zero_point,
            options,
            constants,
            filter_sums,
            shift_rows,
            shift_cols,
            in_cycle: 0,
            out_cycle: 0,
        }
    }


    fn sample(&self, src_row: isize, src_col: isize) -> Option<[T; INPUT_CHANS]> {
        if src_col < 0 || src_col >= INPUT_COLS as isize || src_row < 0 || src_row >= INPUT_ROWS as isize {
            return match self.options.view_padding {
                crate::tensor::TensorViewPadding::Same => Some([self.input_zero_point; INPUT_CHANS]),
                crate::tensor::TensorViewPadding::Valid => None,
            };
        }

        let target_cycle = (src_row as usize) * INPUT_COLS + (src_col as usize);
        let total_input_cycles = INPUT_ROWS.saturating_mul(INPUT_COLS);
        let current_cycle = self
            .in_cycle
            .saturating_sub(1)
            .min(total_input_cycles.saturating_sub(1));

        if target_cycle > current_cycle {
            return match self.options.view_padding {
                crate::tensor::TensorViewPadding::Same => Some([self.input_zero_point; INPUT_CHANS]),
                crate::tensor::TensorViewPadding::Valid => None,
            };
        }

        let age = current_cycle - target_cycle;
        if age < BUF_SIZE {
            let idx = (self.write_idx + BUF_SIZE - 1 - age) % BUF_SIZE;
            Some(self.buffer[idx])
        } else {
            Some([self.input_zero_point; INPUT_CHANS])
        }
    }

    fn compute(&self, out_row: usize, out_col: usize) -> [T; FILTERS_BATCHES] {
        array::from_fn(|b| {
            let input_zero_point = i32::from_subset(&self.input_zero_point);
            let filters_zero_point = i32::from_subset(
                &self.filters.zero_point.get(b).copied().unwrap_or(self.filters.zero_point[0]),
            );

            let mut dot = 0i32;
            let mut sum_input = 0i32;

            // Compute center of kernel purely from output coordinates
            let center_row = out_row * self.options.strides.0;
            let center_col = out_col * self.options.strides.1;

            for kh in 0..FILTERS_ROWS {
                for kw in 0..FILTERS_COLS {
                    let src_row = center_row as isize + kh as isize - self.shift_rows as isize;
                    let src_col = center_col as isize + kw as isize - self.shift_cols as isize;

                    let x = self.sample(src_row, src_col).unwrap_or([self.input_zero_point; INPUT_CHANS]);

                    for c in 0..INPUT_CHANS {
                        let xv = i32::from_subset(&x[c]);
                        let fv = i32::from_subset(&self.filters.buffer[b][(kh, kw)][c]);
                        dot += xv * fv;
                        sum_input += xv;
                    }
                }
            }

            // ... (Keep your exact constants and activation math here, it is completely correct) ...
            let constants = (
                self.constants.0[b],
                self.constants.1.get(b).copied().unwrap_or(self.constants.1[0]),
                input_zero_point * self.filter_sums[b],
                (FILTERS_ROWS * FILTERS_COLS * INPUT_CHANS) as i32 * input_zero_point * filters_zero_point,
            );

            let y = T::from_superset_unchecked(&roundf(
                f32::from_subset(&self.output_zero_point[0])
                    + constants.0
                    + constants.1 * f32::from_subset(&(dot - sum_input * filters_zero_point - constants.2 + constants.3)),
            ));

            match self.options.fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, self.output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, self.output_scale[0], self.output_zero_point[0]),
            }
        })
    }

    pub fn push(&mut self, pixel: [T; INPUT_CHANS]) -> Option<[T; FILTERS_BATCHES]> {
        let in_row = self.in_cycle / INPUT_COLS;
        if in_row < INPUT_ROWS && BUF_SIZE > 0 {
            self.buffer[self.write_idx] = pixel;
            self.write_idx = (self.write_idx + 1) % BUF_SIZE;
        }

        self.in_cycle += 1;
        
        // Determine target output dimensions
        let out_cols = match self.options.view_padding {
            crate::tensor::TensorViewPadding::Same => (INPUT_COLS + self.options.strides.1 - 1) / self.options.strides.1,
            crate::tensor::TensorViewPadding::Valid => (INPUT_COLS.saturating_sub(FILTERS_COLS)) / self.options.strides.1 + 1,
        };
        let out_rows = match self.options.view_padding {
            crate::tensor::TensorViewPadding::Same => (INPUT_ROWS + self.options.strides.0 - 1) / self.options.strides.0,
            crate::tensor::TensorViewPadding::Valid => (INPUT_ROWS.saturating_sub(FILTERS_ROWS)) / self.options.strides.0 + 1,
        };

        if self.out_cycle >= out_rows * out_cols {
            return None; // Fully complete!
        }

        let out_row = self.out_cycle / out_cols;
        let out_col = self.out_cycle % out_cols;

        // 3. Find the bottom-right input coordinate required for this output
        let req_in_row = out_row * self.options.strides.0 + FILTERS_ROWS.saturating_sub(1).saturating_sub(self.shift_rows);
        let req_in_col = out_col * self.options.strides.1 + FILTERS_COLS.saturating_sub(1).saturating_sub(self.shift_cols);
        
        // Convert that coordinate back into the linear clock cycle
        let req_cycles = req_in_row * INPUT_COLS + req_in_col + 1;

        // 4. Emit if the required pixels have arrived (or if the padding flush has clocked past them)
        if self.in_cycle >= req_cycles {
            let emit = self.compute(out_row, out_col);
            self.out_cycle += 1;
            Some(emit)
        } else {
            None
        }
    }
}


impl<
        T: Quantized,
        const INPUT_ROWS: usize,
        const INPUT_COLS: usize,
        const INPUT_CHANS: usize,
        const FILTERS_BATCHES: usize,
        const FILTERS_ROWS: usize,
        const FILTERS_COLS: usize,
        const FILTERS_QUANTS: usize,
        const BUF_SIZE: usize,
    > StreamOp<T, INPUT_CHANS, FILTERS_BATCHES>
    for StreamingConv2D<
        T,
        INPUT_ROWS,
        INPUT_COLS,
        INPUT_CHANS,
        FILTERS_BATCHES,
        FILTERS_ROWS,
        FILTERS_COLS,
        FILTERS_QUANTS,
        BUF_SIZE,
    >
{
    #[inline(always)]
    fn is_finished(&self) -> bool {
        let out_cols = match self.options.view_padding {
            crate::tensor::TensorViewPadding::Same => {
                (INPUT_COLS + self.options.strides.1 - 1) / self.options.strides.1
            }
            crate::tensor::TensorViewPadding::Valid => {
                (INPUT_COLS.saturating_sub(FILTERS_COLS)) / self.options.strides.1 + 1
            }
        };
        let out_rows = match self.options.view_padding {
            crate::tensor::TensorViewPadding::Same => {
                (INPUT_ROWS + self.options.strides.0 - 1) / self.options.strides.0
            }
            crate::tensor::TensorViewPadding::Valid => {
                (INPUT_ROWS.saturating_sub(FILTERS_ROWS)) / self.options.strides.0 + 1
            }
        };
        self.out_cycle >= out_rows * out_cols
    }

    #[inline(always)]
    fn push(&mut self, input: [T; INPUT_CHANS]) -> Option<[T; FILTERS_BATCHES]> {
        StreamingConv2D::push(self, input)
    }

    fn output_scale(&self) -> [f32; 1] {
        self.output_scale
    }

    fn output_zero_point(&self) -> [T; 1] {
        self.output_zero_point
    }
}


#[cfg(test)]
mod tests {
    use nalgebra::matrix;

    use crate::buffer::Buffer2D;
    use crate::ops_options::conv_2d::Conv2DOptions;
    use crate::streaming_ops::stream_pipeline::stream_pipeline;
    use crate::activation::FusedActivation;
    use crate::tensor::{Tensor4D, TensorViewPadding};

    use super::*; // your StreamingConv2D

    const INPUT: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [1, 2], [3, 4],  [5,  6];
            [7, 8], [9, 10], [11, 12]
        ]],
        scale: [0.13],
        zero_point: [14],
    };

    const FILTERS: Tensor4D<i8, 2, 2, 3, 2, 2> = Tensor4D {
        buffer: [
            matrix![
                [15, 16], [17, 18], [19, 20];
                [21, 22], [23, 24], [25, 26]
            ],
            matrix![
                [27, 28], [29, 30], [31, 32];
                [33, 34], [35, 36], [37, 38]
            ],
        ],
        scale: [0.39, 0.40],
        zero_point: [41, 42],
    };

    const OUTPUT_SCALE: [f32; 1] = [0.49];
    const OUTPUT_ZERO_POINT: [i8; 1] = [50];

    const OPTIONS: Conv2DOptions = Conv2DOptions {
        fused_activation: FusedActivation::None,
        view_padding: TensorViewPadding::Same,
        strides: (1, 1),
    };

    const CONSTANTS: (Buffer2D<f32, 2, 1>, Buffer2D<f32, 2, 1>) = (
        matrix![-3.673_469_4; -3.755_102],
        matrix![0.103_469_39; 0.106_122_45],
    );

    const OUTPUT_REF: Tensor4D<i8, 1, 2, 3, 2, 1> = Tensor4D {
        buffer: [matrix![
            [127, 116], [127, 127], [127, 113];
            [98,  74],  [114, 84],  [82,  67]
        ]],
        scale: [0.49],
        zero_point: [50],
    };

    #[test]
    fn streaming_conv2d_matches_reference() {
        let op: StreamingConv2D<i8, 2, 3, 2, 2, 2, 3, 2, 6> = StreamingConv2D::new(
            INPUT.zero_point[0],
            FILTERS,
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
        >(&INPUT, op);

        assert_eq!(result, OUTPUT_REF);
    }
}