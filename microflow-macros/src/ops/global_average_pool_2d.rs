use crate::activation::TokenFusedActivation;
use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor4D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{quote, ToTokens};
use simba::scalar::SupersetOf;

/// Represents the tokenized version of the `GlobalAveragePool2D` operator.
pub(crate) struct TokenGlobalAveragePool2D<T: TokenQuantized> {
    pub(crate) output: TokenTensor4D<T>,
    pub(crate) fused_activation: TokenFusedActivation,
    pub(crate) constants: (f32, f32),
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenGlobalAveragePool2D::<i8>::new(operator, tensors)),
        TensorType::UINT8 => Box::new(TokenGlobalAveragePool2D::<u8>::new(operator, tensors)),
        input_type => abort_call_site!(
            "GlobalAveragePool2D supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

impl<T: TokenQuantized> TokenGlobalAveragePool2D<T> {
    pub(crate) fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> Self {
        let inputs = operator.inputs().unwrap();
        let input = TokenTensor4D::from_empty_tensor(tensors.get(inputs.get(0) as usize));
        let output = TokenTensor4D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        let constants = Self::preprocess(&input, &output);
        Self {
            output,
            fused_activation: TokenFusedActivation::None,
            constants,
        }
    }

    fn preprocess(input: &TokenTensor4D<T>, output: &TokenTensor4D<T>) -> (f32, f32) {
        (
            input.scale[0] / output.scale[0],
            f32::from_subset(&output.zero_point[0])
                - (input.scale[0] * f32::from_subset(&input.zero_point[0])) / output.scale[0],
        )
    }
}

impl<T: TokenQuantized> ToTokens for TokenGlobalAveragePool2D<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        // The runtime always returns Tensor4D<T,1,1,1,C,1>. tflite usually
        // stores the output as rank-2 [batch, C] (a MEAN op that folds the
        // spatial dims). In that case emit a Tensor2D type annotation so the
        // next op — typically FULLY_CONNECTED — sees a rank-2 tensor; we
        // recover it via the existing Tensor4D → Tensor2D Into impl.
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let (constants_0, constants_1) = self.constants;
        let call = quote! {
            microflow::ops::global_average_pool_2d(
                input,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                #fused_activation,
                (#constants_0, #constants_1)
            )
        };

        let ts = if self.output.shape.len() == 2 {
            let output_shape = &self.output.shape;
            quote! {
                let input: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> =
                    Into::into(#call);
            }
        } else {
            let mut output_shape = self.output.shape.clone();
            while output_shape.len() < 4 {
                output_shape.insert(1, 1);
            }
            quote! {
                let input: microflow::tensor::Tensor4D<_, #(#output_shape),*, 1usize> = #call;
            }
        };
        ts.to_tokens(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::TokenBuffer4D;

    fn setup() -> TokenGlobalAveragePool2D<i8> {
        TokenGlobalAveragePool2D {
            output: TokenTensor4D {
                buffer: TokenBuffer4D::new(),
                shape: vec![1, 1, 1, 2],
                scale: vec![0.1],
                zero_point: vec![2],
            },
            fused_activation: TokenFusedActivation::None,
            constants: (3., 4.),
        }
    }

    #[test]
    fn global_average_pool_2d_preprocess() {
        let layer = setup();
        let input = TokenTensor4D {
            buffer: TokenBuffer4D::new(),
            shape: vec![1, 2, 3, 2],
            scale: vec![0.5],
            zero_point: vec![6],
        };
        let constants = TokenGlobalAveragePool2D::preprocess(&input, &layer.output);
        assert_eq!(constants.0, 5.);
        assert_eq!(constants.1, -28.);
    }

    #[test]
    fn global_average_pool_2d_to_tokens() {
        let layer = setup();
        let fused_activation = layer.fused_activation;
        assert_eq!(
            layer.to_token_stream().to_string(),
            quote! {
                let input: microflow::tensor::Tensor4D<_, 1usize, 1usize, 1usize, 2usize, 1usize> =
                    microflow::ops::global_average_pool_2d(
                        input,
                        [0.1f32],
                        [2i8],
                        #fused_activation,
                        (3f32, 4f32)
                );
            }
            .to_string()
        );
    }
}
