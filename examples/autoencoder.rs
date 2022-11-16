use auto_encoder_rust::{
    hashmap,
    model::{Model, Sequential},
};

fn main() {
    let input_dims = 8;
    let hidden_dims = 2;
    let output_dims = 8;

    let mut model = Sequential::default();

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
    model.info();
}
