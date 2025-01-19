use serde::{Deserialize, Serialize};
use crate::{functions::{activation_functions, cost_functions},utils::{applica_bias, genera_biases, genera_weights, moltiplicazione_matrici, dummy}};

#[derive(Debug, Serialize, Deserialize)]
pub struct Matrice<T> {
    pub content: Vec<Vec<T>>, //vettore bidimensionale
}

//  content: vec![             il vettore più esterno contiene le righe, ogni riga è rappresentato da un vettore, e gli elementi contenuti al suo interno appartengono ciascuno a una colonna
//      vec![w1, w3, w5]       le righe rappresentano i weights collegati a un neurone del layer successivo
//      vec![w2, w4, w6]       le colonne rappresentano i weights collegati a un neurone del layer precedente
//  ],                         ESEMPIO: content[0][1] restituisce w3 --> prima riga seconda colonna

impl<T: std::clone::Clone> Matrice<T> {
    //metodo per trasporre una matrice
    pub fn trasposta(&self) -> Matrice<T> {
        let mut content: Vec<Vec<T>> = Vec::new();
        for i in 0..self.content[0].len() {
            let mut row: Vec<T> = Vec::new();
            for j in 0..self.content.len() {
                row.push(self.content[j][i].clone());
            }
            content.push(row);
        }
        Matrice { content }
    }
}


//trait per convertire un vettore in una matrice
pub trait VectoMatrix<T> {
    fn to_matrix(&self) -> Matrice<T>;
}

//il vettore monodimensionale diventa una riga di una matrice, che avrà tante colonne quanti elementi del vettore, e una sola riga
impl<T: std::clone::Clone> VectoMatrix<T> for Vec<T> {
    fn to_matrix(&self) -> Matrice<T> {
        Matrice {
            content: vec![self.to_vec()],
        }
    }
}

//struttura dati per i dataset
#[derive(Debug, Serialize, Deserialize)]
pub struct Sample {
    pub inputs: Vec<f64>,
    pub targets: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Dataset {
    pub samples: Vec<Sample>,
}

//metodi per il dataset
impl Dataset {
    //metodo per inizializzare un dataset da un file json
    pub fn initialize(path: &str) -> Result<Self, String> {
        let serialized = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => return Err(format!("Errore nella lettura del dataset: {}", e))
        };
        
        match serde_json::from_str::<Vec<Sample>>(&serialized) {
            Ok(samples) => Ok(Dataset { samples }),
            Err(e) => Err(format!("Errore nella deserializzazione del dataset: {}", e))
        }
    }

    //metodi per dividere il dataset in training, validation e test set
    pub fn inputs(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let train_percentage = (self.samples.len() as f64 * 0.7) as usize;
        let val_percentage = (self.samples.len() as f64 * 0.1) as usize;
        let (mut x_train, mut x_val, mut x_test) = (Vec::new(), Vec::new(), Vec::new());
        for sample in self.samples.iter() {
            if x_train.len() < train_percentage {
                x_train.push(sample.inputs.clone());
            } else if x_val.len() < val_percentage {
                x_val.push(sample.inputs.clone());
            } else {
                x_test.push(sample.inputs.clone());
            }
        }
        (x_train, x_val, x_test)
    }

    pub fn targets(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let train_percentage = (self.samples.len() as f64 * 0.7) as usize;
        let val_percentage = (self.samples.len() as f64 * 0.1) as usize;
        let (mut y_train, mut y_val, mut y_test) = (Vec::new(), Vec::new(), Vec::new());
        for sample in self.samples.iter() {
            if y_train.len() < train_percentage {
                y_train.push(sample.targets.clone());
            } else if y_val.len() < val_percentage {
                y_val.push(sample.targets.clone());
            } else {
                y_test.push(sample.targets.clone());
            }
        }
        (y_train, y_val, y_test)
    }
}

//struttura dati per i layer della rete neurale (il layer di input non è incluso)
#[derive(Debug, Serialize, Deserialize)]
pub struct Layer {
    pub nodi: u32,
    pub activation: String, //nome identificativo della funzione di attiavazione
    pub bias: Vec<f64>,
    pub weights: Matrice<f64>,
}
 
impl Layer {
    //metodo per creare un nuovo layer 
    pub fn new(nodi: u32, activation: &str, nodi_precedenti: u32) -> Self {
       Layer {
           nodi,
           activation: activation.to_string(),
           bias: genera_biases(nodi, (-1.0, 1.0)),
           weights: genera_weights(nodi_precedenti, nodi, (-1.0, 1.0)),   
       }
    }
    
    //metodo per calcolare l'output di un layer
    pub fn output(&self, inputs: Vec<f64>) -> Vec<f64> {
       let output =match self.activation.to_lowercase().as_str() {
        //applica la funzione di attivazione ai risultati della moltiplicazione tra i weights e gli input, a cui vengono aggiunti i bias
            "relu" => activation_functions::relu_vectors(applica_bias(moltiplicazione_matrici(&self.weights, &inputs), &self.bias)),
            "sigmoid" => activation_functions::sigmoid_vectors(applica_bias(moltiplicazione_matrici(&self.weights, &inputs), &self.bias)),
            "step" => activation_functions::step_vectors(applica_bias(moltiplicazione_matrici(&self.weights, &inputs), &self.bias)),
            "tanh" => activation_functions::tanh_vectors(applica_bias(moltiplicazione_matrici(&self.weights, &inputs), &self.bias)),
            "linear" => activation_functions::linear_vectors(applica_bias(moltiplicazione_matrici(&self.weights, &inputs), &self.bias)),
            "softmax" => activation_functions::softmax_vectors(applica_bias(moltiplicazione_matrici(&self.weights, &inputs), &self.bias)),
            _ => panic!("Funzione di attivazione non supportata: relu, sigmoid, step, tanh"),
        };

        if output.len() != self.nodi as usize {
            panic!("Il numero di nodi del layer non corrisponde all'output");
        } else {
            output
        }
        
    }
}

//struttura dati per la rete neurale
#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub input_nodi: u32,
    pub hidden_layers: Vec<Layer>,
    pub output_layer: Layer
}

impl NeuralNetwork {
    pub fn new(input_nodi: u32, hidden_layers: Vec<Layer>, output_layer: Layer) -> Self {
        NeuralNetwork {
            input_nodi,
            hidden_layers,
            output_layer
        }
    }
 
    //metodo per calcolare l'output della rete neurale
    pub fn forwardprop(&mut self, inputs: Vec<f64>) -> (Vec<f64>, Vec<Vec<f64>>) {
       let mut hidden_outputs = inputs; 
       let mut layers_outputs: Vec<Vec<f64>> = vec![hidden_outputs.clone()];
       //calcola l'output di ogni layer nascosto che funge da input per il layer successivo (propagazione dell'input iniziale)
       for layer in self.hidden_layers.iter_mut() {
            hidden_outputs = layer.output(hidden_outputs);
            layers_outputs.push(hidden_outputs.clone());
       }
       (self.output_layer.output(hidden_outputs), layers_outputs)
    }

    //metodo per calcolare i gradienti e aggiornare i pesi della rete
    pub fn backprop(&mut self, outputs: Vec<f64>, targets: Vec<f64>, layers_outputs: Vec<Vec<f64>>, initial_inputs: Vec<f64>, learning_rate: f64, cost_function: &str) {
        
        //calcolo della derivata della funzione di attivazione dell'output layer
        let output_activation_derivative = match self.output_layer.activation.to_lowercase().as_str() {
            "relu" => activation_functions::relu_derivative_vectors(&outputs),
            "sigmoid" => activation_functions::sigmoid_derivative_vectors(&outputs),
            "step" => activation_functions::step_derivative_vectors(&outputs),
            "tanh" => activation_functions::tanh_derivative_vectors(&outputs),
            "linear" => activation_functions::linear_derivative_vectors(&outputs),
            "softmax" => activation_functions::softmax_derivative_vectors(&outputs),
            _ => panic!("Funzione di attivazione non supportata: relu, sigmoid, step, tanh"),     
        };

        //associazione della funzione di attivazione dei layer nascosti ad una variabile
        let hidden_activation_derivative = if self.hidden_layers.len() > 0 {
            match self.hidden_layers[0].activation.to_lowercase().as_str() {
                "relu" => activation_functions::relu_derivative_vectors,
                "sigmoid" => activation_functions::sigmoid_derivative_vectors,
                "step" => activation_functions::step_derivative_vectors,
                "tanh" => activation_functions::tanh_derivative_vectors,
                "linear" => activation_functions::linear_derivative_vectors,
                "softmax" => activation_functions::softmax_derivative_vectors,
                _ => panic!("Funzione di attivazione non supportata: relu, sigmoid, step, tanh"),     
            }
        } else {
            |v: &Vec<f64>| dummy(v.clone())
        };

        //calcolo della derivata della funzione di costo
        let cost_function_derivative = match cost_function.to_lowercase().as_str() {
            "mse" => cost_functions::mse_derivative_vectors(&outputs, &targets),
            "mae" => cost_functions::mae_derivative_vectors(&outputs, &targets),
            "rms" => cost_functions::rms_derivative_vectors(&outputs, &targets),
            "bce" => cost_functions::bce_derivative_vectors(&outputs, &targets),
            _ => panic!("Funzione di costo non supportata: mse, mae, rms, bce"),
        };

        //calcolo del gradiente dell'errore rispetto ai pesi e ai bias dell'output layer
        let mut delta_output = Vec::new();
        for i in 0..outputs.len() {
            delta_output.push(cost_function_derivative[i] * output_activation_derivative[i]);
        }
        //aggiornamento dei pesi dell'output layer
        for (i, weight) in self.output_layer.weights.content.iter_mut().enumerate() {
            for (j, w) in weight.iter_mut().enumerate() {
                *w -= learning_rate * delta_output[i] * layers_outputs.last().unwrap()[j];
            }
        }
        //aggiornamento dei bias dell'output layer
        for (i, bias) in self.output_layer.bias.iter_mut().enumerate() {
            *bias -= learning_rate * delta_output[i];
        }

        //calcolo gradienti dei layer nascosti, per farlo la formula richiede il gradiente dell'errore rispetto ai pesi e ai bias del layer successivo
        let mut delta_hidden = delta_output; //per l'ultimo layer nascosto (si va a ritroso) il layer successivo è l'output layer quindi il gradiente sarà delta_output
        let mut weights_next = &self.output_layer.weights; //stesso discorso per i weights 
        let hidden_layers_len = self.hidden_layers.len(); //calcola il numero di layer nascosti per un controllo successivo

        //la procedura avviene in un ciclo per applicarla a tutti i layer nascosti andando a ritroso
        for (i, layer) in self.hidden_layers.iter_mut().rev().enumerate() {

            //calcolo dei gradienti 
            let vec = moltiplicazione_matrici(&weights_next.trasposta(), &delta_hidden); 
            let activation_derivative = hidden_activation_derivative(&layers_outputs[layers_outputs.len() - 1 - i]);
            let mut delta_hidden_new = Vec::new();
            for (j, d) in vec.iter().enumerate() {
                delta_hidden_new.push(d * activation_derivative[j]);
            }

            //controllo per capire se il layer corrente è l'ultimo (primo) layer nascosto
            let inputs = if i == hidden_layers_len - 1 {
                &initial_inputs
            } else {
                &layers_outputs[layers_outputs.len() - 2 - i]
            };
            
            //aggiornamento dei pesi e dei bias del layer corrente
            for (i, weight) in layer.weights.content.iter_mut().enumerate() {
                for (j, w) in weight.iter_mut().enumerate() {
                    if j < inputs.len() {  // Aggiungo controllo sulla dimensione
                        *w -= learning_rate * delta_hidden_new[i] * inputs[j];
                    }
                }
            }
            for (j, bias) in layer.bias.iter_mut().enumerate() {
                *bias -= learning_rate * delta_hidden_new[j];
            }

            //finito il ciclo la procedura si propaga all'indietro al layer precedente, quindi il layer corrente diventerà il layer successivo per il futuro layer corrente
            delta_hidden = delta_hidden_new; //così i delta del layer corrente diventano i delta del layer successivo
            weights_next = &layer.weights; //e allo stesso modo si comportano i weights
        }
            
    }

    //metodo per addestrare la rete neurale
    pub fn addestra(&mut self, x_train: Vec<Vec<f64>>, y_train: Vec<Vec<f64>>, x_val: Vec<Vec<f64>>, y_val: Vec<Vec<f64>>,epochs: u32, cost_function: &str, learning_rate: f64) {
        //la procedura viene ripetuta per un dato numero di epoche
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (i, input) in x_train.iter().enumerate() {
                let (outputs, layers_outputs) = self.forwardprop(input.clone()); //calcola l'output della rete
                let loss = match cost_function { //calcola la loss
                    "mse" => cost_functions::mse_vectors(&outputs, &y_train[i]),
                    "mae" => cost_functions::mae_vectors(&outputs, &y_train[i]),
                    "rms" => cost_functions::rms_vectors(&outputs, &y_train[i]),
                    "bce" => cost_functions::bce_vectors(&outputs, &y_train[i]),
                    _ => panic!("Funzione di costo non supportata: mse, mae, rms, bce"),
                };
                total_loss += loss;
                
                self.backprop(outputs.clone(), y_train[i].clone(), layers_outputs, input.clone(), learning_rate, cost_function); //calcola i gradienti ed aggiorna i pesi

                if i % 100 == 0 {
                    println!("Input: {:?}, Output: {:?}, Loss: {}", input, outputs, loss);
                }
            }

            let mut val_loss = 0.0;
            for (i, input) in x_val.iter().enumerate() {
                let (outputs, _) = self.forwardprop(input.clone());
                let loss = match cost_function {
                    "mse" => cost_functions::mse_vectors(&outputs, &y_val[i]),
                    "mae" => cost_functions::mae_vectors(&outputs, &y_val[i]),
                    "rms" => cost_functions::rms_vectors(&outputs, &y_val[i]),
                    "bce" => cost_functions::bce_vectors(&outputs, &y_val[i]),
                    _ => panic!("Funzione di costo non supportata: mse, mae, rms, bce"),
                };
                val_loss += loss;

                if i % 100 == 0 {
                    println!("Input val: {:?}, Output val: {:?}, Loss val: {}", input, outputs, loss);
                }
            }

            println!("Epoch: {}, Loss: {}, Validation Loss: {}", epoch + 1, total_loss / x_train.len() as f64, val_loss / x_val.len() as f64); //stampa l'andamento dell'addestramento
        }
    }

    //metodo per stampare la struttura della rete neurale
    pub fn struttura(&self) {
        println!("Input layer: {}", self.input_nodi);
        for (i, layer) in self.hidden_layers.iter().enumerate() {
            println!("Hidden layer {}: nodi: {}, funzione di attivazione: {}", i+1, layer.nodi, layer.activation);
        }
        println!("Output layer: nodi: {}, funzione di attivazione: {}", self.output_layer.nodi, self.output_layer.activation);
    }

    //metodo per salvare la rete neurale in un file json
    pub fn salva(&self, path: &str) {
        let serialized = serde_json::to_string(&self).unwrap();
        match std::fs::write(path, serialized) {
            Ok(_) => println!("Rete salvata correttamente"),
            Err(e) => println!("Errore nel salvataggio della rete: {}", e),
            
        }
    }

    //metodo per caricare una rete neurale da un file json
    pub fn carica(path: &str) -> Result<Self, String> {
        // Legge il file JSON come stringa
        let serialized = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => return Err(format!("Errore nel caricamento della rete: {}", e))
        };

        // Deserializza la stringa JSON nella struttura NeuralNetwork
        match serde_json::from_str::<NeuralNetwork>(&serialized) {
            Ok(nn) => Ok(nn),
            Err(e) => Err(format!("Errore nella deserializzazione della rete: {}", e))
        }
    }

}