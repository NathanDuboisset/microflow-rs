pub mod average_pool_2d;
pub mod conv_2d;
pub mod global_average_pool_2d;
pub mod max_pool_2d;
pub mod pipeline;

use crate::tflite_flatbuffers::tflite::BuiltinOperator;
use proc_macro2::{Ident, TokenStream as TokenStream2};

/// Represents a single operator that has been parsed and is ready to be
/// chained into a streaming pipeline.
pub struct StreamingNode {
    /// The code required to initialize the operator (e.g. `let mut stream_op_0 = ...;`)
    pub setup_tokens: TokenStream2,
    /// The identifier of the operator variable (e.g. `stream_op_0`)
    pub op_ident: Ident,
    /// The Rust type tokens for the input (e.g. `i8` or `u8`)
    pub input_type: TokenStream2,
    /// The shape of the input tensor entering this specific operator
    pub in_shape: Vec<usize>,
    /// The shape of the output tensor leaving this specific operator
    pub out_shape: Vec<usize>,
    /// The operator index within the model (used for timing labels).
    pub index: usize,
}

/// Returns true when the operator should be compiled through the streaming pipeline.
pub(crate) fn is_streaming_operator(opcode: BuiltinOperator) -> bool {
    matches!(
        opcode,
        BuiltinOperator::CONV_2D
            | BuiltinOperator::AVERAGE_POOL_2D
            | BuiltinOperator::MAX_POOL_2D
            | BuiltinOperator::MEAN
    )
}
