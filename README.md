# NNLIB

`nnlib` è una libreria Rust per la creazione e l'addestramento di reti neurali. Questa libreria fornisce strumenti per costruire modelli di reti neurali, caricare e salvare modelli, e addestrarli utilizzando i propri dataset.

## Features

- **Creazione di reti neurali**: Definisci un modello con parametri personalizzati come numero di neuroni e di layer.
- **Addestramento**: Addestra le reti neurali utilizzando diversi algoritmi di ottimizzazione e funzioni di costo.
- **Caricamento e salvataggio**: Carica modelli pre-addestrati da file JSON e salva i modelli dopo l'allenamento.
- **Funzioni di attivazione**: Supporto per diverse funzioni di attivazione come ReLU, Sigmoid, Tanh, Softmax, ecc.
- **Funzioni di costo**: Supporto per diverse funzioni di costo come MSE, MAE, BCE, ecc.
- **Dataset**: Gestione dei dataset per l'addestramento, la validazione e il test.

## Installazione

Aggiungi questa riga al tuo file `Cargo.toml`:

```toml
[dependencies]
nnlib = { path = "path/to/nnlib" }
```

Oppure:

```toml
[dependencies]
nnlib = { git = "https://github.com/CiroBurro/nnlib" }
```

## Utilizzo

### Creazione di una rete neurale

```rust
use nnlib::structures::{Layer, NeuralNetwork};

fn crea_rete() -> NeuralNetwork {
    let input_nodi = 3;
    let layer_1 = Layer::new(2, "relu", input_nodi);
    let layer_2 = Layer::new(3, "relu", layer_1.nodi);
    let output_layer = Layer::new(1, "linear", layer_2.nodi);

    NeuralNetwork::new(input_nodi, vec![layer_1, layer_2], output_layer)
}
```

### Addestramento di una rete neurale

```rust
fn main() {
    let mut nn = crea_rete();
    let learning_rate = 0.003;
    let cost_function = "mse";
    let epochs = 5;

    let dataset = structures::Dataset::initialize("dataset_esempio.json").unwrap();
    let (x_train, x_val, _x_test) = dataset.inputs();
    let (y_train, y_val, _y_test) = dataset.targets();

    nn.addestra(x_train, y_train, x_val, y_val, epochs, cost_function, learning_rate);
    nn.salva("model.json");
}
```

### Caricamento di un modello pre-addestrato

```rust
fn main() {
    let mut nn = structures::NeuralNetwork::carica("model.json").unwrap();
    let inputs = vec![0.0, 1.0];
    let (output, _) = nn.forwardprop(inputs);
    println!("{:?}", output);
}
```
