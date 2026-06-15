use super::StreamingNode;
use proc_macro2::TokenStream as TokenStream2;
use proc_macro_error::abort_call_site;
use quote::quote;

// The pipeline runtime always takes 4D dims. tflite stores MEAN outputs as
// rank-2 `[batch, C]`, so map that to `(1, 1, C)` here.
fn pad_4d(shape: &[usize]) -> (usize, usize, usize) {
    match shape.len() {
        4 => (shape[1], shape[2], shape[3]),
        2 => (1, 1, shape[1]),
        n => abort_call_site!(
            "streaming pipeline node shape rank {} unsupported (shape {:?})",
            n,
            shape
        ),
    }
}

/// Aggregates streaming operators and compiles them into a single chained pipeline.
pub struct StreamPipeline {
    nodes: Vec<StreamingNode>,
}

impl StreamPipeline {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a parsed streaming operator to the delayed buffer.
    pub fn push(&mut self, node: StreamingNode) {
        self.nodes.push(node);
    }

    /// Returns true if no streaming operator is currently buffered.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the operator index of the first buffered node, if any.
    pub fn first_index(&self) -> Option<usize> {
        self.nodes.first().map(|n| n.index)
    }

    /// Flushes the buffered streaming operators into the main token stream.
    /// This writes the setup code, the `.then()` chains, and the final pipeline executor.
    pub fn flush(&mut self, tokens: &mut TokenStream2) {
        if self.nodes.is_empty() {
            return;
        }

        for node in &self.nodes {
            let setup = &node.setup_tokens;
            tokens.extend(quote! { #setup });
        }

        let first_ident = &self.nodes[0].op_ident;
        let mut chain_expr = quote! { #first_ident };
        for node in self.nodes.iter().skip(1) {
            let op_ident = &node.op_ident;
            chain_expr = quote! { #chain_expr.then(#op_ident) };
        }

        let first_node = self.nodes.first().expect("streaming pipeline is non-empty");
        let last_node = self.nodes.last().expect("streaming pipeline is non-empty");

        let input_type = &first_node.input_type;
        let (in_r, in_c, in_ch) = pad_4d(&first_node.in_shape);
        let (out_r, out_c, out_ch) = pad_4d(&last_node.out_shape);
        let collapse_to_2d = last_node.out_shape.len() == 2;

        tokens.extend(quote! {
            let input = microflow::streaming_ops::stream_pipeline::<
                #input_type, #in_r, #in_c, #in_ch, #out_r, #out_c, #out_ch, _
            >(&input, #chain_expr);
        });

        // `stream_pipeline` always yields Tensor4D<T,1,1,1,C,1>; if the model
        // stores the output as rank-2 (a MEAN feeding a FULLY_CONNECTED),
        // reshape so the next op sees the right type.
        if collapse_to_2d {
            tokens.extend(quote! {
                let input: microflow::tensor::Tensor2D<#input_type, 1usize, #out_ch, 1usize> =
                    microflow::ops::reshape(input);
            });
        }

        self.nodes.clear();
    }
}
