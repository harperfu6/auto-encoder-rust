use auto_encoder_rust::{
    data::SinSource,
    hashmap,
    model::{Model, Sequential},
    optimizer::get_optimizer_with_defaults,
    plot::plot_vec,
};

use arrayfire::DType;

fn main() {
    let input_dims = 8;
    let hidden_dims = 2;
    let output_dims = 8;
    let num_train_samples = 128;
    let batch_size = 32;
    let optimizer_type = "SGD";
    let epochs = 5;

    let optimizer = get_optimizer_with_defaults(optimizer_type).unwrap();
    let mut model = Sequential::new(optimizer, "mse");

    model.add(
        "dense",
        hashmap![
            "activation" => "tanh".to_string()
            , "input_size" => input_dims.to_string()
            , "output_size" => output_dims.to_string()
            , "w_init" => "uniform".to_string()
            , "b_init" => "zeros".to_string()
        ],
    );
    // model.info();

    let source = SinSource::new(input_dims, batch_size, DType::F32, num_train_samples);
    let loss = model.fit::<SinSource>(&source, epochs, batch_size, None, true);

    // plot_vec(loss, "Loss vs. Iterations", 512, 512);
}
