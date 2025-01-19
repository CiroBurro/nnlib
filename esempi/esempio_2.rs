fn main() {
    use nnlib::structures;

    //carico un modello già addestrato
    let mut nn = match structures::NeuralNetwork::carica("model.json") {
        Ok(n) => n,
        Err(e) => panic!("Errore: {}", e),
    };

    //definisco gli input ed eseguo una predizione (ottengo un output dalla rete)
    let inputs = vec![0.0, 1.0];
    nn.struttura();
    let (output, _) = nn.forwardprop(inputs);
    println!("{:?}", output);
}