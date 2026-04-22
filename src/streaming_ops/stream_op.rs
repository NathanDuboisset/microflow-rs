
use crate::quantize::Quantized;

pub trait StreamOp<T: Quantized, const IN: usize, const OUT: usize> {
    fn push(&mut self, input: [T; IN]) -> Option<[T; OUT]>;
}

/// Identity (useful for building pipelines)
pub struct Identity;

impl<T: Quantized, const C: usize> StreamOp<T, C, C> for Identity {
    #[inline(always)]
    fn push(&mut self, input: [T; C]) -> Option<[T; C]> {
        Some(input)
    }
}

/// Chain two operators: A -> B
pub struct Chain<A, B> {
    pub a: A,
    pub b: B,
}

impl<T, const IN: usize, const MID: usize, const OUT: usize, A, B>
    StreamOp<T, IN, OUT> for Chain<A, B>
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
}

/// Helper to chain more cleanly
pub trait ChainExt<T: Quantized, const IN: usize, const MID: usize>:
    StreamOp<T, IN, MID> + Sized
{
    fn then<const OUT: usize, B>(self, b: B) -> Chain<Self, B>
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