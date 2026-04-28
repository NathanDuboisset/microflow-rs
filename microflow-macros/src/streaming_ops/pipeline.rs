use super::StreamingNode;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

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
        let (in_r, in_c, in_ch) = (
            first_node.in_shape[1],
            first_node.in_shape[2],
            first_node.in_shape[3],
        );
        let (out_r, out_c, out_ch) = (
            last_node.out_shape[1],
            last_node.out_shape[2],
            last_node.out_shape[3],
        );

        tokens.extend(quote! {
            use microflow::streaming_ops::stream_op::ChainExt;

            let input = microflow::streaming_ops::stream_pipeline::<
                #input_type, #in_r, #in_c, #in_ch, #out_r, #out_c, #out_ch, _
            >(&input, #chain_expr);
        });

        self.nodes.clear();
    }
}
