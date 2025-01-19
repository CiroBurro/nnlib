pub mod activation_functions {
    //funzioni di attivazione più comuni
    //versione vectors: applica la determinata funzione a tutti gli elementi di un vettore di input
    //versione derivative: derivata della funzione per calcolare i gradienti
    //versione derivative_vectors: derivata della funzione applicata a tutti gli elementi di un vettore di input
    use crate::utils::max;
    use std::f64::consts::E;

    //funzione sigmoid
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }
    pub fn sigmoid_vectors(input: Vec<f64>) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();
        for x in input.iter() {
            output.push(sigmoid(*x));
        }
        output
    }
    pub fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }
    pub fn sigmoid_derivative_vectors(input: &Vec<f64>) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();
        for x in input.iter() {
            output.push(sigmoid_derivative(*x));
        }
        output
    }

    //funzione rectified linear
    pub fn relu(x: f64) -> f64 {
        max(0.0, x)
    }
    pub fn relu_vectors(input: Vec<f64>) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();
        for x in input.iter() {
            output.push(relu(*x));
        }
        output
    }
    pub fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
    pub fn relu_derivative_vectors(input: &Vec<f64>) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();
        for x in input.iter() {
            output.push(relu_derivative(*x));
        }
        output
    }

    //funzione Step (Heaviside)
    pub fn step(x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            1.0
        }
    }
    pub fn step_vectors(input: Vec<f64>) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();
        for x in input.iter() {
            output.push(step(*x));
        }
        output
    }
    pub fn step_derivative(_x: f64) -> f64 {
        0.0
    }
    pub fn step_derivative_vectors(input: &Vec<f64>) -> Vec<f64> {
        vec![0.0; input.len()]
    }

    //funzione tangente iperbolica
    pub fn tanh(x: f64) -> f64 {
        (E.powf(x) - E.powf(-x)) / (E.powf(x) + E.powf(-x))
    }
    pub fn tanh_vectors(input: Vec<f64>) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();
        for x in input.iter() {
            output.push(tanh(*x));
        }
        output
    }
    pub fn tanh_derivative(x: f64) -> f64 {
        1.0 - tanh(x).powf(2.0)
    }
    pub fn tanh_derivative_vectors(input: &Vec<f64>) -> Vec<f64> {
        let mut output: Vec<f64> = Vec::new();
        for x in input.iter() {
            output.push(tanh_derivative(*x));
        }
        output
    }

    //funzione lineare (nessuna funzione di attivazione)
    pub fn linear(x: f64) -> f64 { x }
    pub fn linear_vectors(input:Vec<f64>) -> Vec<f64> { input }
    pub fn linear_derivative(x: f64) -> f64 {
        if x >= 0.0 {
            1.0
        }
        else {
            -1.0
        }
    }
    pub fn linear_derivative_vectors(input: &Vec<f64>) -> Vec<f64> {
        let mut output = Vec::new();
        for x in input.iter() {
            output.push(linear_derivative(*x));
        }
        output
    } 

    //funzione softmax, accetta solo vettori come input
    pub fn softmax_vectors(input: Vec<f64>) -> Vec<f64> {
        let max_input = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max); // Trova il massimo valore
        let mut output = Vec::new();
        let mut somma = 0.0;
    
        for x in input.iter() {
            somma += E.powf(x - max_input); // Calcola la somma degli esponenziali normalizzati
        }
    
        for x in input.iter() {
            output.push(E.powf(x - max_input) / somma); // Normalizza ogni valore
        }
    
        output
    }
    pub fn softmax_derivative_vectors(output: &Vec<f64>) -> Vec<f64> {
        let mut derivatives = Vec::new();
        for i in 0..output.len() {
            // La derivata della softmax per ogni elemento è: Si * (1 - Si)
            // dove Si è l'output della softmax per quell'elemento
            derivatives.push(output[i] * (1.0 - output[i]));
        }
        derivatives
    }
}

pub mod cost_functions {
    //funzioni di perdita più comuni
    //versione vectors: da applicare a un vettore 
    //versione derivative: derivata della funzione di perdita per calcolare i gradienti
    //versione derivative_vectors: applica la derivata della funzione a un vettore
    use crate::utils::{max, min};

    //funzione binary cross entropy loss
    pub fn binary_ce(output: &f64, target: &f64) -> f64 {
        let h: f64 = 10_f64.powf(-7.0);
        let output = max(h, min(*output, 1.0 - h));

        -(target * output.ln() + (1.0 - target) * (1.0 - output).ln())
    }
    pub fn bce_vectors(outputs: &[f64], targets: &[f64]) -> f64 {
        if outputs.len() != targets.len() {
            panic!("gli input devono avere la stessa lunghezza")
        }

        let mut somma = 0.0;
        for (i, _) in outputs.iter().enumerate() {
            somma += binary_ce(&outputs[i], &targets[i]);
        }
        -(somma / outputs.len() as f64)
    }
    pub fn bce_derivative(output: &f64, target: &f64) -> f64 {
        -(target / output) + ((1.0 - target) / (1.0 - output))
    }
    pub fn bce_derivative_vectors(outputs: &[f64], targets: &[f64]) -> Vec<f64> {
        if outputs.len() != targets.len() {
            panic!("gli input devono avere la stessa lunghezza")
        }

        let mut errors = Vec::new();
        for (i, _) in outputs.iter().enumerate() {
            errors.push(bce_derivative(&outputs[i], &targets[i]));
        }
        errors
    }

    //funzione errore quadratico medio
    pub fn mean_squared_error(output: &f64, target: &f64) -> f64 {
        (target - output).powf(2.0)
    }
    pub fn mse_vectors(outputs: &[f64], targets: &[f64]) -> f64 {
        if outputs.len() != targets.len() {
            panic!("gli input devono avere la stessa lunghezza")
        }

        let mut somma = 0.0;
        for (i, _) in outputs.iter().enumerate() {
            somma += mean_squared_error(&outputs[i], &targets[i]);
        }
        somma / outputs.len() as f64
    }
    pub fn mse_derivative(output: &f64, target: &f64) -> f64 {
        -2.0 * (target - output)
    }
    pub fn mse_derivative_vectors(outputs: &[f64], targets: &[f64]) -> Vec<f64> {
        if outputs.len() != targets.len() {
            panic!("gli input devono avere la stessa lunghezza")
        }
        let mut errors = Vec::new();
        for (i, _) in outputs.iter().enumerate() {
            errors.push(
                mse_derivative(&outputs[i], &targets[i])
            );
        }
        errors
    }

    //funzione errore assoluto medio
    pub fn mean_absolute_error(output: &f64, target: &f64) -> f64 {
        (target - output).abs()
    }
    pub fn mae_vectors(outputs: &[f64], targets: &[f64]) -> f64 {
        if outputs.len() != targets.len() {
            panic!("gli input devono avere la stessa lunghezza")
        }

        let mut somma = 0.0;
        for (i, _) in outputs.iter().enumerate() {
            somma += mean_absolute_error(&outputs[i], &targets[i]);
        }
        somma / outputs.len() as f64
    }
    pub fn mae_derivative(output: &f64, target: &f64) -> f64 {
        if output > target {
            -1.0
        } else if output < target {
            1.0
        } else {
            0.0
        }
    }
    pub fn mae_derivative_vectors(outputs: &[f64], targets: &[f64]) -> Vec<f64> {
        if outputs.len() != targets.len() {
            panic!("gli input devono avere la stessa lunghezza")
        }
        let mut errors = Vec::new();
        for (i, _) in outputs.iter().enumerate() {
            errors.push(
                mae_derivative(&outputs[i], &targets[i])
            );
        }
        errors
    }

    //funzione errore radice della media quadratica
    pub fn root_mean_square(output: &f64, target: &f64) -> f64 {
        mean_squared_error(output, target).powf(1.0 / 2.0)
    }
    pub fn rms_vectors(outputs: &[f64], targets: &[f64]) -> f64 {
        mse_vectors(outputs, targets).powf(1.0 / 2.0)
    }
    pub fn rms_derivative(output: &f64, target: &f64) -> f64 {
        let rms = root_mean_square(output, target);
        if rms == 0.0 {
            0.0
        } else {
            mse_derivative(output, target) / (2.0 * rms)
        }
    }
    pub fn rms_derivative_vectors(outputs: &[f64], targets: &[f64]) -> Vec<f64> {
        let mut errors = Vec::new();
        for (i, _) in outputs.iter().enumerate() {
            errors.push(
                rms_derivative(&outputs[i], &targets[i])
            );
        }
        errors
    }
}

