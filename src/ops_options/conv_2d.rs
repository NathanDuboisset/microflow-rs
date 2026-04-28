use crate::activation::FusedActivation;
use crate::tensor::TensorViewPadding;

pub struct Conv2DOptions {
    pub fused_activation: FusedActivation,
    pub view_padding: TensorViewPadding,
    pub strides: (usize, usize),
}
