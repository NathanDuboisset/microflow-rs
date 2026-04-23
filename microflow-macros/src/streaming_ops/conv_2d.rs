use super::StreamingNode;
use crate::activation::TokenFusedActivation;
use crate::buffer::TokenBuffer2D;
use crate::quantize::TokenQuantized;
use crate::tensor::{TokenTensor2D, TokenTensor4D, TokenTensorViewPadding};
use crate::tflite_flatbuffers::tflite::{Buffer, Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use nalgebra::DMatrix;
use proc_macro_error::abort_call_site;
use quote::{format_ident, quote};

/// Represents the tokenized version of a `StreamingConv2D` operator.
pub(crate) struct TokenStreamingConv2D<T: TokenQuantized> {
    pub(crate) input: TokenTensor4D<T>,
    pub(crate) filters: TokenTensor4D<T>,
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) view_padding: TokenTensorViewPadding,
    pub(crate) strides: (usize, usize),
    pub(crate) constants: (TokenBuffer2D<f32>, TokenBuffer2D<f32>),
    pub(crate) index: usize,
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
    buffers: Vector<ForwardsUOffset<Buffer>>,
    index: usize,
) -> StreamingNode {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => {
            TokenStreamingConv2D::<i8>::new(operator, tensors, buffers, index)
                .to_streaming_node(quote! { i8 })
        }
        TensorType::UINT8 => {
            TokenStreamingConv2D::<u8>::new(operator, tensors, buffers, index)
                .to_streaming_node(quote! { u8 })
        }
        input_type => abort_call_site!(
            "StreamingConv2D supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

impl<T: TokenQuantized> TokenStreamingConv2D<T> {
    pub(crate) fn new(
        operator: Operator,
        tensors: Vector<ForwardsUOffset<Tensor>>,
        buffers: Vector<ForwardsUOffset<Buffer>>,
        index: usize,
    ) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor4D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let filters =
            TokenTensor4D::from_buffered_tensor(tensors.get(inputs.get(1) as usize), buffers);
        let biases =
            TokenTensor2D::from_buffered_tensor(tensors.get(inputs.get(2) as usize), buffers);
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let options = operator.builtin_options_as_conv_2_doptions().unwrap();
        let constants = Self::preprocess(&input, &filters, &biases, &output);
        Self {
            input,
            filters,
            output,
            fused_activation: options.fused_activation_function().into(),
            view_padding: options.padding().into(),
            strides: (options.stride_h() as usize, options.stride_w() as usize),
            constants,
            index,
        }
    }

    fn preprocess(
        input: &TokenTensor4D<T>,
        filters: &TokenTensor4D<T>,
        biases: &TokenTensor2D<i32>,
        output: &TokenTensor4D<T>,
    ) -> (TokenBuffer2D<f32>, TokenBuffer2D<f32>) {
        (
            TokenBuffer2D::from(DMatrix::from_fn(filters.shape[0], 1, |b, _| {
                biases.scale.get(b).copied().unwrap_or(biases.scale[0]) / output.scale[0]
                    * (biases.buffer[b]
                        - biases
                            .zero_point
                            .get(b)
                            .copied()
                            .unwrap_or(biases.zero_point[0])) as f32
            })),
            TokenBuffer2D::from(DMatrix::from_fn(filters.scale.len(), 1, |b, _| {
                input.scale[0] * filters.scale[b] / output.scale[0]
            })),
        )
    }

    /// Converts the parsed data into a delayed StreamingNode.
    pub(crate) fn to_streaming_node(&self, type_tokens: proc_macro2::TokenStream) -> StreamingNode {
        let op_ident = format_ident!("stream_op_{}", self.index);
        let filters_ident = format_ident!("filters_{}", self.index);
        let filters_type = self.filters.type_tokens();
        let filters = &self.filters;
        
        let input_zp = &self.input.zero_point[0];
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let (constants_0, constants_1) = &self.constants;
        let in_r = self.input.shape[1];
        let in_c = self.input.shape[2];
        let in_ch = self.input.shape[3];
        let f_b = self.filters.shape[0];
        let f_r = self.filters.shape[1];
        let f_c = self.filters.shape[2];
        let f_q = self.filters.scale.len();

        let setup_tokens = quote! {
            const #filters_ident: #filters_type = #filters;
            let mut #op_ident = microflow::streaming_ops::StreamingConv2D::<
                #type_tokens, #in_r, #in_c, #in_ch, #f_b, #f_r, #f_c, #f_q
            >::new(
                #input_zp,
                #filters_ident,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops_options::Conv2DOptions {
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
    use crate::buffer::{TokenBuffer2D, TokenBuffer4D};
    use nalgebra::dmatrix;

    fn setup() -> TokenStreamingConv2D<i8> {
        TokenStreamingConv2D {
            input: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.35],
                zero_point: vec![36],
            },
            filters: TokenTensor4D {
                buffer: TokenBuffer4D::from(vec![
                    dmatrix![
                        vec![1, 2], vec![3,  4],  vec![5,  6];
                        vec![7, 8], vec![9,  10], vec![11, 12]
                    ],
                    dmatrix![
                        vec![13, 14], vec![15, 16], vec![17, 18];
                        vec![19, 20], vec![21, 22], vec![23, 24]
                    ],
                ]),
                shape: vec![2, 2, 3, 2],
                scale: vec![0.25, 0.26],
                zero_point: vec![27, 28],
            },
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 2, 3, 2],
                scale: vec![0.29],
                zero_point: vec![30],
            },
            fused_activation: TokenFusedActivation::Relu6,
            view_padding: TokenTensorViewPadding::Same,
            strides: (1, 1),
            constants: (
                TokenBuffer2D::from(dmatrix![31., 32.]),
                TokenBuffer2D::from(dmatrix![33., 34.]),
            ),
            index: 0,
        }
    }

    #[test]
    fn streaming_conv_2d_preprocess() {
        let layer = setup();
        let biases = TokenTensor2D {
            buffer: TokenBuffer2D::from(dmatrix![
                37;
                38
            ]),
            shape: vec![2, 1],
            scale: vec![0.39, 0.40],
            zero_point: vec![41, 42],
        };
        let constants =
            TokenStreamingConv2D::preprocess(&layer.input, &layer.filters, &biases, &layer.output);
        assert_eq!(constants.0 .0, Some(dmatrix![-5.37931; -5.5172415]));
        assert_eq!(constants.1 .0, Some(dmatrix![0.30172414; 0.3137931]));
    }

    #[test]
    fn streaming_conv_2d_to_streaming_node_tokens() {
        let layer = setup();
        let filters = &layer.filters;
        let fused_activation = layer.fused_activation;
        let view_padding = layer.view_padding;
        let (constants_0, constants_1) = &layer.constants;

        let node = layer.to_streaming_node(quote! { i8 });
        assert_eq!(node.op_ident.to_string(), "stream_op_0");
        assert_eq!(node.input_type.to_string(), quote! { i8 }.to_string());
        assert_eq!(node.in_shape, vec![1, 2, 3, 2]);
        assert_eq!(node.out_shape, vec![1, 2, 3, 2]);
        assert_eq!(
            node.setup_tokens.to_string(),
            quote! {
                const filters_0: microflow::tensor::Tensor4D<i8, 2usize, 2usize, 3usize, 2usize, 2usize> = #filters;
                let mut stream_op_0 = microflow::streaming_ops::StreamingConv2D::<
                    i8, 2usize, 3usize, 2usize, 2usize, 2usize, 3usize, 2usize
                >::new(
                        36i8,
                        filters_0,
                        [0.29f32],
                        [30i8],
                        microflow::ops_options::Conv2DOptions {
                            fused_activation: #fused_activation,
                            view_padding: #view_padding,
                            strides: (1usize, 1usize),
                        },
                        (#constants_0, #constants_1)
                    );
            }
            .to_string()
        );
    }
}
