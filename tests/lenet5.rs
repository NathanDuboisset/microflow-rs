use microflow::buffer::Buffer2D;
use microflow_macros::model;
use nalgebra::matrix;

#[model("models/lenet5.tflite")]
struct Lenet5;

#[model("models/lenet5.tflite", enable_kernel_streaming = true)]
struct Lenet5Streaming;


#[test]
fn lenet5_model() {
    let input = [Buffer2D::from_element([1.0]); 1];
    let normal_input = Lenet5::predict(input.clone());
    let streaming_input = Lenet5Streaming::predict(input);

    let expected_output = matrix![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99609375, 0.0
    ];

    assert_eq!(normal_input, expected_output);
    assert_eq!(streaming_input, expected_output);
    
}
