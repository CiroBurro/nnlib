use crate::structures::Matrice;
use rand::{thread_rng, Rng};

//funzione per generare i bias casualmente in un dato intervallo per un dato layer
pub fn genera_biases(layer_nodes: u32, range: (f64, f64)) -> Vec<f64> {
    let mut rng = thread_rng();
    let mut biases = Vec::new();
    for _ in 0..layer_nodes {
        biases.push(rng.gen_range(range.0..range.1));
    }
    biases
}

pub fn genera_weights(layer_1_nodes: u32, layer_2_nodes: u32, range: (f64, f64)) -> Matrice<f64> {
    let mut rng = thread_rng();
    let mut content: Vec<Vec<f64>> = Vec::new();

    for _ in 0..layer_2_nodes {
        let mut rows = Vec::new();
        for _ in 0..layer_1_nodes {
            rows.push(rng.gen_range(range.0..range.1));
        }
        content.push(rows);
    }

    Matrice { content }
}

pub fn moltiplicazione_matrici(weights: &Matrice<f64>, input: &[f64]) -> Vec<f64> {
    let mut output: Vec<f64> = Vec::new();

    for i in 0..weights.content.len() {
        let mut somma = 0.0;
        for (j, _) in input.iter().enumerate() {
            somma += weights.content[i][j] * input[j];
        }
        output.push(somma);
    }

    output
}

pub fn applica_bias(input: Vec<f64>, bias: &[f64]) -> Vec<f64> {
    let mut output = Vec::new();

    if input.len() != bias.len() {
        panic!("I vettori devono avere la stessa lunghezza");
    }

    for (i, _) in input.iter().enumerate() {
        let somma = input[i] + bias[i];
        output.push(somma);
    }
    output
}

pub fn max<T: std::cmp::PartialOrd>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

pub fn min<T: std::cmp::PartialOrd>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

pub fn dummy(_: Vec<f64>) -> Vec<f64> {
    vec![0.0]
}