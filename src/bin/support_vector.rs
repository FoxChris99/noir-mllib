#![allow(unused)]
use noir::prelude::*;
use std::time::Instant;
use rand::Rng;
use serde::{Deserialize, Serialize};

use noir_ml::{basic_stat::get_moments, sample::Sample};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;


//State for SGD
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct StateSGD {
    //regression coefficients
    pub weights: Vec<f64>,
    //total gradient of the batch
    global_grad: Vec<f64>,
    //iterations over the dataset
    epoch: usize,
}

impl StateSGD {
    pub fn new() -> StateSGD {
        StateSGD {
            weights:  Vec::<f64>::new(),
            global_grad: Vec::<f64>::new(),
            epoch : 0,
        }}}


struct SupportVectorMachine {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    lambda_param: f64,
}

impl SupportVectorMachine {
    fn new(num_features: usize, learning_rate: f64, lambda_param: f64) -> SupportVectorMachine {
        let mut rng = rand::thread_rng();
        let weights = (0..num_features).map(|_| rng.gen_range(-1.0..1.0)).collect();

        SupportVectorMachine {
            weights,
            bias: 0.0,
            learning_rate,
            lambda_param,
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let mut activation = self.bias;
        for i in 0..self.weights.len() {
            activation += self.weights[i] * features[i];
        }
        if activation >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

//     fn train(&mut self, features: &[Vec<f64>], labels: &[f64], num_epochs: usize) {
//         for _ in 0..num_epochs {
//             for i in 0..features.len() {
//                 let prediction = self.predict(&features[i]);
//                 let error = labels[i] - prediction;

//                 for j in 0..self.weights.len() {
//                     self.weights[j] += self.learning_rate * (2.0 * self.lambda_param * self.weights[j]);
//                 }

//                 if error != 0.0 {
//                     for j in 0..self.weights.len() {
//                         self.weights[j] += self.learning_rate * (2.0 * self.lambda_param * self.weights[j] - features[i][j] * labels[i]);
//                     }
//                     self.bias += self.learning_rate * labels[i];
//                 }
//             }
//         }
//     }
// }

    fn fit(&mut self, path_to_data: &String, num_iters: usize, learning_rate: f64, lambda_param: f64, config: &EnvironmentConfig){
        let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let fit = env.stream(source.clone())
        .replay(
            num_iters,
            StateSGD::new(),

            move |s, state| 
            {
                //shuffle the samples
                s.shuffle()
                //each replica filter a number of samples equal to batch size and
                //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                .rich_filter_map(
                    move |mut x|{
                        let dim = x.0.len();
                        //each iteration just a fraction of data is considered
                        if rand::thread_rng().gen::<f64>() > (1.0 - data_fraction){
                            if normalization==true{
                                //scale the features and the target
                                x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                }
                            //the target is in the last element of each sample
                            let y: f64 = x.0.pop().unwrap(); 
                            //assign to the target -1 or 1 based on the class
                            

                            let mut current_weights = &state.get().weights;
                            let vec = vec![0.;dim];
                            if state.get().epoch == 0{
                                current_weights = &vec;
                            }

                            let prediction: f64 = x.0.iter().zip(current_weights.iter()).map(|(xi, wi)| xi * wi).sum() - self.bias;

                            let new_weights;
                            let new_bias;
                            if y * prediction >=1.{
                                new_weights = current_weights.iter().map(|wi| wi - 2. * learning_rate * lambda_param * wi).collect();
                            }
                            else{
                                new_weights = current_weights.iter().map(|wi| wi - 2. * learning_rate * lambda_param * wi - 
                                x.0.iter().map(|xi| xi * y).sum()).collect();
                                new_bias = learning_rate * y;
                            }

                            Some(Sample(sample_grad))}
                        else{
                            None
                        }
            })
                //the average of the gradients is computed and forwarded as a single value
                .group_by_avg(|_x| true, |x| x.clone()).drop_key()//.max_parallelism(1)
            },

            move |local_grad: &mut Sample, avg_grad| 
            {   
                if avg_grad.0.len()!=0{
                *local_grad = avg_grad;}
            },

            move |state, local_grad| 
            {   
                //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                if local_grad.0.len()!=0{
                state.global_grad = local_grad.0.clone();}
            },

            move|state| 
            {   
                //initialize
                if state.epoch==0{
                    state.weights = vec![0.;state.global_grad.len()]
                }
                //update iterations
                 state.epoch +=1;
                //update the weights (optional with weight decay)
                state.weights = state.weights.iter().zip(state.global_grad.iter()).map(|(wi,g)| wi - learn_rate*g).collect();
                if weight_decay==true{
                    state.weights = state.weights.iter().map(|wi| wi -  learn_rate * 0.002 * wi).collect();
                }
                //tolerance=gradient's L2 norm for the stop condition
                let tol: f64 = state.global_grad.iter().map(|v| v*v).sum();
                //reset the global gradient for the next iteration
                state.global_grad = vec![0.;state.weights.len()];
                //loop condition
                state.epoch < num_iters && tol.sqrt() > 1e-4
            },

        )
        .collect_vec();

    env.execute();

    let state = fit.get().unwrap()[0].clone();
    state
    }
}


fn main() {
    let (config, _args) = EnvironmentConfig::from_args();
    let training_set = "forest_fire.csv".to_string();
    let data_to_predict = "forest_fire.csv".to_string();
 
    let mut model = LinearRegression::new();
    
    //hyper_parameters for the iterative method model.fit
    let method = "SGD".to_string(); //"ADAM".to_string()
    let num_iters = 100;
    let learn_rate = 1e-1;
    let data_fraction = 0.001;
    let weight_decay = false;

    let normalize = true;

    let start = Instant::now();
    //return the trained model
    //model = model.fit(&training_set, method, num_iters, learn_rate, data_fraction, normalize, weight_decay, &config);

    //fitting with ols
    model = model.fit_ols(&training_set, false, &config);

    let elapsed = start.elapsed();

    let start_score = Instant::now();
    //compute the score over the training set
    let r2 = model.clone().score(&training_set, &config);
    let elapsed_score = start_score.elapsed();
    
    let start_pred = Instant::now();
    let predictions = model.clone().predict(&data_to_predict, &config);
    let elapsed_pred = start_pred.elapsed();
    

    print!("\nCoefficients: {:?}\n", model.features_coef);
    print!("Intercept: {:?}\n", model.intercept);  
    print!("\nR2 score: {:?}\n", r2);
    print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<f64>>());
    eprintln!("\nElapsed fit: {elapsed:?}");
    eprintln!("\nElapsed score: {elapsed_score:?}"); 
    eprintln!("\nElapsed pred: {elapsed_pred:?}");     

}