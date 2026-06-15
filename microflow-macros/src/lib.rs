//! [![crates.io](https://img.shields.io/crates/v/microflow-macros)](https://crates.io/crates/microflow-macros)
//! [![docs.rs](https://img.shields.io/docsrs/microflow-macros)](https://docs.rs/microflow-macros)
//! [![github](https://img.shields.io/github/actions/workflow/status/matteocarnelos/microflow-rs/cargo.yml?branch=main)](https://github.com/matteocarnelos/microflow-rs/actions/workflows/cargo.yml)
//!
//! Macro crate of the [MicroFlow](https://github.com/matteocarnelos/microflow-rs) inference engine, namely, the MicroFlow compiler.

extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro_error::{abort_call_site, proc_macro_error};
use std::fs;

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::{parse_macro_input, ItemStruct};

use crate::tflite_flatbuffers::tflite::TensorType;
use ops::*;
use streaming_ops::pipeline::StreamPipeline;
use structmeta::StructMeta;
use syn::{LitBool, LitStr};
use tflite_flatbuffers::tflite::{root_as_model, BuiltinOperator};

mod activation;
mod buffer;
mod ops;
mod quantize;
mod streaming_ops;
mod tensor;
#[path = "../flatbuffers/tflite_generated.rs"]
#[allow(unused_imports)]
#[allow(clippy::all)]
mod tflite_flatbuffers;

#[derive(StructMeta)]
struct Args {
    #[struct_meta(unnamed)]
    path: LitStr,
    enable_kernel_streaming: Option<LitBool>,
    enable_timing: Option<LitBool>,
}

/// The entry point of MicroFlow.
/// This attribute-like procedural macro can be placed on `structs` to implement the `predict()`
/// function based on the given model.
/// The macro takes as input the path of the model, which must be in the TensorFlow Lite format
/// (`.tflite`).
#[proc_macro_error]
#[proc_macro_attribute]
pub fn model(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as Args);
    let item = parse_macro_input!(item as ItemStruct);
    let enable_kernel_streaming = args
        .enable_kernel_streaming
        .map(|lit| lit.value())
        .unwrap_or(false);
    let enable_timing = args.enable_timing.map(|lit| lit.value()).unwrap_or(false);

    let buf = fs::read(args.path.value()).unwrap_or_else(|_| {
        abort_call_site!(
            "couldn't find '{}', please provide a valid path",
            &args.path.value()
        )
    });
    let model = root_as_model(&buf).unwrap_or_else(|_| {
        abort_call_site!("invalid model, please provide a valid TensorFlow Lite model")
    });

    let ident = &item.ident;

    let subgraph = model.subgraphs().unwrap().get(0);
    let tensors = subgraph.tensors().unwrap();
    let buffers = model.buffers().unwrap();

    let input = tensors.get(subgraph.inputs().unwrap().get(0) as usize);
    let mut input_shape: Vec<_> = input.shape().unwrap().iter().map(|e| e as usize).collect();
    if input_shape.len() == 1 {
        input_shape.insert(0, 1);
    }
    let input_rank = input_shape.len();
    let input_type = match input.type_() {
        TensorType::INT8 => quote!(i8),
        TensorType::UINT8 => quote!(u8),
        input_type => abort_call_site!(
            "unsupported input tensor type: {:?}. Supported input types are INT8 and UINT8",
            input_type
        ),
    };
    let input_tensor = match input_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        rank => abort_call_site!(
            "unsupported input tensor rank: {} (shape {:?}). Supported ranks are 2 and 4",
            rank,
            input_shape
        ),
    };
    let input_buffer = match input_shape.len() {
        2 => quote!(Buffer2D),
        4 => quote!(Buffer4D),
        rank => abort_call_site!(
            "unsupported input tensor rank for buffer mapping: {} (shape {:?}). Supported ranks are 2 and 4",
            rank,
            input_shape
        ),
    };
    let input_scale: Vec<_> = input
        .quantization()
        .unwrap()
        .scale()
        .unwrap()
        .iter()
        .map(|e| e.to_token_stream())
        .collect();
    let input_zero_point: Vec<_> = match input.type_() {
        TensorType::INT8 => input
            .quantization()
            .unwrap()
            .zero_point()
            .unwrap()
            .iter()
            .map(|e| (e as i8).to_token_stream())
            .collect(),
        TensorType::UINT8 => input
            .quantization()
            .unwrap()
            .zero_point()
            .unwrap()
            .iter()
            .map(|e| (e as u8).to_token_stream())
            .collect(),
        input_type => abort_call_site!(
            "unsupported input zero-point tensor type: {:?}. Supported types are INT8 and UINT8",
            input_type
        ),
    };

    let operators = subgraph.operators().unwrap();
    let mut layers = TokenStream2::new();
    // Imported once so multiple flushed pipelines don't each re-`use ChainExt`
    // and hit E0252.
    if enable_kernel_streaming {
        layers.extend(quote! {
            use microflow::streaming_ops::stream_op::ChainExt;
        });
    }
    let mut stream_pipeline = StreamPipeline::new();
    for (index, operator) in operators.iter().enumerate() {
        let opcode = BuiltinOperator(
            model
                .operator_codes()
                .unwrap()
                .get(operator.opcode_index() as usize)
                .deprecated_builtin_code() as i32,
        );

        if enable_kernel_streaming && streaming_ops::is_streaming_operator(opcode) {
            let node = match opcode {
                BuiltinOperator::CONV_2D => {
                    streaming_ops::conv_2d::parse(operator, tensors, buffers, index)
                }
                BuiltinOperator::AVERAGE_POOL_2D => {
                    streaming_ops::average_pool_2d::parse(operator, tensors, index)
                }
                BuiltinOperator::MEAN => {
                    streaming_ops::global_average_pool_2d::parse(operator, tensors, index)
                }
                _ => unreachable!("non-streaming opcode reached streaming branch"),
            };
            stream_pipeline.push(node);
            continue;
        }

        if enable_kernel_streaming {
            flush_pipeline(&mut stream_pipeline, &mut layers, enable_timing);
        }

        let layer: Box<dyn ToTokens> = match opcode {
            BuiltinOperator::FULLY_CONNECTED => {
                fully_connected::parse(operator, tensors, buffers, index)
            }
            BuiltinOperator::DEPTHWISE_CONV_2D => {
                depthwise_conv_2d::parse(operator, tensors, buffers, index)
            }
            BuiltinOperator::CONV_2D => conv_2d::parse(operator, tensors, buffers, index),
            BuiltinOperator::AVERAGE_POOL_2D => average_pool_2d::parse(operator, tensors),
            BuiltinOperator::MEAN => global_average_pool_2d::parse(operator, tensors),
            BuiltinOperator::SOFTMAX => softmax::parse(operator, tensors),
            BuiltinOperator::RESHAPE => Box::new(reshape::parse(operator, tensors)),
            BuiltinOperator::TRANSPOSE => transpose::parse(operator, tensors, buffers),
            unsupported_op => abort_call_site!("unsupported operator: {:?}", unsupported_op),
        };

        if enable_timing {
            let mut layer_ts = TokenStream2::new();
            layer.to_tokens(&mut layer_ts);
            let opcode_name = format!("{:?}", opcode);
            let start_ident = format_ident!("__ml_start_{}", index);
            layers.extend(quote! {
                let #start_ident = ::microflow::__layer_start!();
                #layer_ts
                ::microflow::__layer_end!(#opcode_name, #index, #start_ident);
            });
        } else {
            layer.to_tokens(&mut layers);
        }
    }
    if enable_kernel_streaming {
        flush_pipeline(&mut stream_pipeline, &mut layers, enable_timing);
    }

    let output = tensors.get(subgraph.outputs().unwrap().get(0) as usize);
    let mut output_shape: Vec<_> = output.shape().unwrap().iter().map(|e| e as usize).collect();
    if output_shape.len() == 1 {
        output_shape.insert(0, 1);
    }
    let output_type = match output.type_() {
        TensorType::INT8 => quote!(i8),
        TensorType::UINT8 => quote!(u8),
        output_type => abort_call_site!(
            "unsupported output tensor type: {:?}. Supported output types are INT8 and UINT8",
            output_type
        ),
    };
    let output_tensor = match output_shape.len() {
        2 => quote!(Tensor2D),
        4 => quote!(Tensor4D),
        rank => abort_call_site!(
            "unsupported output tensor rank: {} (shape {:?}). Supported ranks are 2 and 4",
            rank,
            output_shape
        ),
    };
    let output_buffer = match output_shape.len() {
        2 => quote!(Buffer2D),
        4 => quote!(Buffer4D),
        rank => abort_call_site!(
            "unsupported output tensor rank for buffer mapping: {} (shape {:?}). Supported ranks are 2 and 4",
            rank,
            output_shape
        ),
    };

    let ts = quote! {
        #item
        impl #ident {
            pub const fn expose_input() -> ([f32; 1], [#input_type; 1], [usize; #input_rank]) {
                (
                    [#(#input_scale),*],
                    [#(#input_zero_point),*],
                    [#(#input_shape),*],
                )
            }

            pub fn predict(input: microflow::buffer::#input_buffer<f32, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::quantize(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                Self::predict_inner(input).dequantize()
            }

            pub fn predict_quantized(input: microflow::buffer::#input_buffer<#input_type, #(#input_shape),*>) -> microflow::buffer::#output_buffer<f32, #(#output_shape),*> {
                let input = microflow::tensor::#input_tensor::new(input, [#(#input_scale),*], [#(#input_zero_point),*]);
                Self::predict_inner(input).dequantize()
            }

            fn predict_inner(input: microflow::tensor::#input_tensor<#input_type, #(#input_shape),*, 1usize>) -> microflow::tensor::#output_tensor<#output_type, #(#output_shape),*, 1usize> {
                #layers
                input
            }
        }
    };

    fs::write("target/microflow-expansion.rs", ts.to_string()).ok();

    ts.into()
}

fn flush_pipeline(pipeline: &mut StreamPipeline, tokens: &mut TokenStream2, timing: bool) {
    if pipeline.is_empty() {
        return;
    }
    if timing {
        let pipe_index = pipeline.first_index().unwrap_or(0);
        let start_ident = format_ident!("__ml_start_stream_{}", pipe_index);
        let mut body = TokenStream2::new();
        pipeline.flush(&mut body);
        tokens.extend(quote! {
            let #start_ident = ::microflow::__layer_start!();
            #body
            ::microflow::__layer_end!("StreamingPipeline", #pipe_index, #start_ident);
        });
    } else {
        pipeline.flush(tokens);
    }
}
