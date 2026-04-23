use crate::quantize::Quantized;

pub trait StreamOp<T: Quantized, const IN: usize, const OUT: usize> {
    fn push(&mut self, input: [T; IN]) -> Option<[T; OUT]>;
    
    // NEW: Expose the output metadata so the pipeline can read it
    fn output_scale(&self) -> [f32; 1];
    fn output_zero_point(&self) -> [T; 1];
}

/// Identity (useful for building pipelines)
pub struct Identity<T: Quantized> {
    pub scale: [f32; 1],
    pub zero_point: [T; 1],
}

impl<T: Quantized, const C: usize> StreamOp<T, C, C> for Identity<T> {
    #[inline(always)]
    fn push(&mut self, input: [T; C]) -> Option<[T; C]> {
        Some(input)
    }

    fn output_scale(&self) -> [f32; 1] {
        self.scale
    }

    fn output_zero_point(&self) -> [T; 1] {
        self.zero_point
    }
}

/// Chain two operators: A -> B
pub struct Chain<const MID: usize, A, B> {
    pub a: A,
    pub b: B,
}

impl<T, const IN: usize, const MID: usize, const OUT: usize, A, B>
    StreamOp<T, IN, OUT> for Chain<MID, A, B>
where
    T: Quantized,
    A: StreamOp<T, IN, MID>,
    B: StreamOp<T, MID, OUT>,
{
    #[inline(always)]
    fn push(&mut self, input: [T; IN]) -> Option<[T; OUT]> {
        if let Some(mid) = self.a.push(input) {
            self.b.push(mid)
        } else {
            None
        }
    }

    // The final metadata of a chained pipeline is dictated by the very last operator
    fn output_scale(&self) -> [f32; 1] {
        self.b.output_scale()
    }

    fn output_zero_point(&self) -> [T; 1] {
        self.b.output_zero_point()
    }
}

/// Helper to chain more cleanly
pub trait ChainExt<T: Quantized, const IN: usize, const MID: usize>:
    StreamOp<T, IN, MID> + Sized
{
    fn then<const OUT: usize, B>(self, b: B) -> Chain<MID, Self, B>
    where
        B: StreamOp<T, MID, OUT>,
    {
        Chain { a: self, b }
    }
}

impl<T, const IN: usize, const MID: usize, A> ChainExt<T, IN, MID> for A
where
    T: Quantized,
    A: StreamOp<T, IN, MID>,
{
}