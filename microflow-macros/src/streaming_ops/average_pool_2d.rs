use super::StreamingNode;
use crate::activation::TokenFusedActivation;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor4D, TokenTensorViewPadding};
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro_error::abort_call_site;
use quote::{format_ident, quote};
use simba::scalar::SupersetOf;

/// Represents the tokenized version of a `StreamingAveragePool2D` operator.
pub(crate) struct TokenStreamingAveragePool2D<T: TokenQuantized> {
    pub(crate) input: TokenTensor4D<T>,
    pub(crate) filter_shape: (usize, usize),
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) view_padding: TokenTensorViewPadding,
    pub(crate) strides: (usize, usize),
    pub(crate) constants: (f32, f32),
    pub(crate) index: usize,
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    index: usize,
) -> StreamingNode {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => TokenStreamingAveragePool2D::<i8>::new(operator, tensors, index)
            .to_streaming_node(quote! { i8 }),
        TensorType::UINT8 => TokenStreamingAveragePool2D::<u8>::new(operator, tensors, index)
            .to_streaming_node(quote! { u8 }),
        input_type => abort_call_site!(
            "StreamingAveragePool2D supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

impl<T: TokenQuantized> TokenStreamingAveragePool2D<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        index: usize,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor4D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator.builtin_options_as_pool_2_doptions().unwrap();
        let constants = Self::preprocess(&input, &output);
        Self {
            input,
            filter_shape: (
                options.filter_height() as usize,
                options.filter_width() as usize,
            ),
            output,
            fused_activation: options.fused_activation_function().into(),
            view_padding: options.padding().into(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
            constants,
            index,
        }
    }

    fn preprocess(input: &TokenTensor4D<T>, output: &TokenTensor4D<T>) -> (f32, f32) {
        (
            input.scale[0] / output.scale[0],
            f32::from_subset(&output.zero_point[0])
                - (input.scale[0] * f32::from_subset(&input.zero_point[0])) / output.scale[0],
        )
    }

    /// Converts the parsed data into a delayed StreamingNode.
    pub(crate) fn to_streaming_node(&self, type_tokens: proc_macro2::TokenStream) -> StreamingNode {
        let op_ident = format_ident!("stream_op_{}", self.index);
        let input_zp = &self.input.zero_point[0];
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let in_r = self.input.shape[1];
        let in_c = self.input.shape[2];
        let in_ch = self.input.shape[3];
        let f_r = self.filter_shape.0;
        let f_c = self.filter_shape.1;
        let buf_size = if f_r == 0 {
            0usize
        } else {
            (f_r - 1) * in_c + f_c
        };
        let (constants_0, constants_1) = self.constants;

        let setup_tokens = quote! {
            let mut #op_ident = microflow::streaming_ops::StreamingAveragePool2D::<
                #type_tokens, #in_r, #in_c, #in_ch, #f_r, #f_c, #buf_size
            >::new(
                    #input_zp,
                    [#(#output_scale),*],
                    [#(#output_zero_point),*],
                    microflow::ops_options::AveragePool2DOptions {
                        fused_activation: #fused_activation,
                        view_padding: #view_padding,
                        strides: (#strides_0, #strides_1),
                    },
                    (#constants_0, #constants_1)
                );
        };

        StreamingNode {
            setup_tokens,
            op_ident,
            input_type: type_tokens,
            in_shape: self.input.shape.clone(),
            out_shape: self.output.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;

    fn setup() -> TokenStreamingAveragePool2D<i8> {
        TokenStreamingAveragePool2D {
            input: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.5],
                zero_point: vec![6],
            },
            filter_shape: (2, 3),
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.1],
                zero_point: vec![2],
            },
            fused_activation: TokenFusedActivation::None,
            view_padding: TokenTensorViewPadding::Same,
            strides: (1, 1),
            constants: (3., 4.),
            index: 0,
        }
    }

    #[test]
    fn streaming_average_pool_2d_preprocess() {
        let layer = setup();
        let constants = TokenStreamingAveragePool2D::preprocess(&layer.input, &layer.output);
        assert_eq!(constants.0, 5.);
        assert_eq!(constants.1, -28.);
    }

    #[test]
    fn streaming_average_pool_2d_to_streaming_node_tokens() {
        let layer = setup();
        let fused_activation = layer.fused_activation;
        let view_padding = layer.view_padding;
        let node = layer.to_streaming_node(quote! { i8 });

        assert_eq!(node.op_ident.to_string(), "stream_op_0");
        assert_eq!(node.input_type.to_string(), quote! { i8 }.to_string());
        assert_eq!(node.in_shape, vec![1, 2, 3, 2]);
        assert_eq!(node.out_shape, vec![1, 2, 3, 2]);
        assert_eq!(
            node.setup_tokens.to_string(),
            quote! {
                let mut stream_op_0 = microflow::streaming_ops::StreamingAveragePool2D::<
                    i8, 2usize, 3usize, 2usize, 2usize, 3usize, 6usize
                >::new(
                        6i8,
                        [0.1f32],
                        [2i8],
                        microflow::ops_options::AveragePool2DOptions {
                            fused_activation: #fused_activation,
                            view_padding: #view_padding,
                            strides: (1usize, 1usize),
                        },
                        (3f32, 4f32)
                    );
            }
            .to_string()
        );
    }
}
