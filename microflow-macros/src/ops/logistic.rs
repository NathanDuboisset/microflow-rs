use crate::quantize::TokenQuantized;
use crate::tensor::TokenTensor2D;
use crate::tflite_flatbuffers::tflite::{Operator, Tensor, TensorType};
use flatbuffers::{ForwardsUOffset, Vector};
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::{quote, ToTokens};

pub(crate) struct TokenLogistic<T: TokenQuantized> {
    pub(crate) output: TokenTensor2D<T>,
}

pub(crate) fn parse(
    operator: Operator,
    tensors: Vector<ForwardsUOffset<Tensor>>,
) -> Box<dyn ToTokens> {
    let inputs = operator.inputs().unwrap();
    let input_type = tensors.get(inputs.get(0) as usize).type_();
    match input_type {
        TensorType::INT8 => Box::new(TokenLogistic::<i8>::new(operator, tensors)),
        TensorType::UINT8 => Box::new(TokenLogistic::<u8>::new(operator, tensors)),
        input_type => abort_call_site!(
            "Logistic supports only INT8/UINT8 input tensors, got {:?}",
            input_type
        ),
    }
}

impl<T: TokenQuantized> TokenLogistic<T> {
    pub(crate) fn new(operator: Operator, tensors: Vector<ForwardsUOffset<Tensor>>) -> Self {
        let output = TokenTensor2D::from_empty_tensor(
            tensors.get(operator.outputs().unwrap().get(0) as usize),
        );
        Self { output }
    }
}

impl<T: TokenQuantized> ToTokens for TokenLogistic<T> {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let output_shape = &self.output.shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;

        let ts = quote! {
            let input: microflow::tensor::Tensor2D<_, #(#output_shape),*, 1usize> =
                microflow::ops::logistic(input, [#(#output_scale),*], [#(#output_zero_point),*]);
        };
        ts.to_tokens(tokens);
    }
}
