use nnlib::structures;
use structures::{Layer, NeuralNetwork};

fn main() {
    let mut nn = crea_rete(); //creo il modello
    //definisco alcuni parametri
    let learning_rate = 0.003;
    let cost_function = "mse";
    let epochs = 5;

    //preparo il dataset
    let dataset = match structures::Dataset::initialize("dataset_esempio.json") {
        Ok(d) => d,
        Err(e) => panic!("Errore: {}", e),
    };
    let (x_train, x_val, _x_test) = dataset.inputs();
    let (y_train, y_val, _y_test) = dataset.targets();

    //addestro e salvo il modello
    nn.struttura(); // OPZIONALE: serve per stampare la struttura della rete (layers, nodi, funzioni di attivazione)
    nn.addestra(x_train, y_train, x_val, y_val, epochs, cost_function, learning_rate);
    nn.salva("model.json");
}

//funzione per creare una rete neurale con iperparametri personalizzati
fn crea_rete() -> NeuralNetwork {
    let input_nodi = 3;
    let layer_1 = Layer::new(2, "relu", input_nodi);
    let layer_2 = Layer::new(3, "relu", layer_1.nodi);
    let output_layer = Layer::new(1, "linear", layer_2.nodi);

    NeuralNetwork::new(input_nodi, vec![layer_1, layer_2], output_layer)
}