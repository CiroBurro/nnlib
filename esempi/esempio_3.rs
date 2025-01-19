fn main() {
    use nnlib::utils::{genera_biases, genera_weights, applica_bias, moltiplicazione_matrici};
    use nnlib::structures;
    use nnlib::functions::{activation_functions, cost_functions};

    //creo manualmente il modello
    let input_nodi = 3;
    let hidden_layers = vec![2, 3];
    let output_nodi = 1;
    let weights_hidden = vec![
        genera_weights(input_nodi, hidden_layers[0], (-1.0, 1.0)),
        genera_weights(hidden_layers[0], hidden_layers[1], (-1.0, 1.0)),
    ];
    let bias_hidden = vec![
        genera_biases(hidden_layers[0], (-1.0, 1.0)),
        genera_biases(hidden_layers[1], (-1.0, 1.0)),
    ];
    let weights_output = genera_weights(hidden_layers[1], output_nodi, (-1.0, 1.0));
    let bias_output = genera_biases(output_nodi, (-1.0, 1.0));

    //preparo il dataset
    let dataset = match structures::Dataset::initialize("dataset_esempio.json") {
        Ok(d) => d,
        Err(e) => panic!("Errore: {}", e),
    };
    let (x_train, _x_val, _x_test) = dataset.inputs();
    let (y_train, _y_val, _y_test) = dataset.targets();

    //addestro il modello manualmente
    for (i, x) in x_train.iter().enumerate() {

        //forward propagation
        let layer_1_output = activation_functions::relu_vectors(applica_bias(moltiplicazione_matrici(&weights_hidden[0], x), &bias_hidden[0]));
        let layer_2_output = activation_functions::relu_vectors(applica_bias(moltiplicazione_matrici(&weights_hidden[1], &layer_1_output), &bias_hidden[1]));
        let output = activation_functions::softmax_vectors(applica_bias(moltiplicazione_matrici(&weights_output, &layer_2_output), &bias_output));
        println!("{:?}", output);

        //La backpropagation non è stata pensata per essere eseguita manualmente, è comunque possibile farlo ma richiede molto lavoro
    }
}
