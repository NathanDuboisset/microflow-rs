use core::array;
use libm::roundf;

use simba::scalar::SupersetOf;

use crate::activation::{relu, relu6, FusedActivation};
use crate::quantize::Quantized;
use crate::streaming_ops::stream_op::StreamOp;

pub struct StreamingGlobalAveragePool2D<
    T: Quantized,
    const INPUT_ROWS: usize,
    const INPUT_COLS: usize,
    const INPUT_CHANS: usize,
> {
    pub input_zero_point: T,
    pub output_scale: [f32; 1],
    pub output_zero_point: [T; 1],
    pub fused_activation: FusedActivation,
    pub constants: (f32, f32),
    pub sums: [i32; INPUT_CHANS],
    pub in_cycle: usize,
    pub out_cycle: usize,
}

impl<T: Quantized, const INPUT_ROWS: usize, const INPUT_COLS: usize, const INPUT_CHANS: usize>
    StreamingGlobalAveragePool2D<T, INPUT_ROWS, INPUT_COLS, INPUT_CHANS>
{
    pub fn new(
        input_zero_point: T,
        output_scale: [f32; 1],
        output_zero_point: [T; 1],
        fused_activation: FusedActivation,
        constants: (f32, f32),
    ) -> Self {
        Self {
            input_zero_point,
            output_scale,
            output_zero_point,
            fused_activation,
            constants,
            sums: [0i32; INPUT_CHANS],
            in_cycle: 0,
            out_cycle: 0,
        }
    }

    fn compute(&self) -> [T; INPUT_CHANS] {
        let area = (INPUT_ROWS * INPUT_COLS) as f32;
        array::from_fn(|c| {
            let x = self.sums[c] as f32 / area;
            let y = T::from_superset_unchecked(&roundf(self.constants.0 * x + self.constants.1));
            match self.fused_activation {
                FusedActivation::None => y,
                FusedActivation::Relu => relu(y, self.output_zero_point[0]),
                FusedActivation::Relu6 => relu6(y, self.output_scale[0], self.output_zero_point[0]),
            }
        })
    }
}

impl<T: Quantized, const INPUT_ROWS: usize, const INPUT_COLS: usize, const INPUT_CHANS: usize>
    StreamOp<T, INPUT_CHANS, INPUT_CHANS>
    for StreamingGlobalAveragePool2D<T, INPUT_ROWS, INPUT_COLS, INPUT_CHANS>
{
    #[inline(always)]
    fn is_finished(&self) -> bool {
        self.out_cycle >= 1
    }

    #[inline(always)]
    fn push(&mut self, pixel: [T; INPUT_CHANS]) -> Option<[T; INPUT_CHANS]> {
        let input_len = INPUT_ROWS * INPUT_COLS;

        if self.in_cycle < input_len {
            for (c, value) in pixel.iter().enumerate() {
                self.sums[c] += i32::from_subset(value);
            }
        }

        self.in_cycle += 1;

        if self.in_cycle == input_len && self.out_cycle == 0 {
            self.out_cycle = 1;
            Some(self.compute())
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
    fn streaming_global_average_pool_2d_layer() {
        let op: StreamingGlobalAveragePool2D<i8, 2, 3, 2> = StreamingGlobalAveragePool2D::new(
            INPUT.zero_point[0],
            OUTPUT_SCALE,
            OUTPUT_ZERO_POINT,
            FusedActivation::None,
            CONSTANTS,
        );
        let result = stream_pipeline::<
            i8,
            2, // INPUT_ROWS
            3, // INPUT_COLS
            2, // INPUT_CHANS
            1, // OUTPUT_ROWS
            1, // OUTPUT_COLS
            2, // OUTPUT_CHANS
            _,
        >(&INPUT, op);
        assert_eq!(result, OUTPUT);
    }
}
