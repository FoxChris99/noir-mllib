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
struct StateSGD {
    //regression coefficients
    weights: Vec<f64>,
    //total gradient of the batch
    bias:f64,
    //iterations over the dataset
    epoch: usize,
}

impl StateSGD {
     fn new() -> StateSGD {
        StateSGD {
            weights:  Vec::<f64>::new(),
            bias: 0.,
            epoch : 0,

        }}}


#[derive(Clone, Debug)]
struct SupportVectorMachine {
    weights: Vec<f64>,
    bias: f64,
    train_mean: Vec<f64>,
    train_std: Vec<f64>,
    normalization: bool,
    fitted: bool
}

impl SupportVectorMachine {
    fn new() -> SupportVectorMachine {
        SupportVectorMachine {
            weights: Vec::<f64>::new(),
            bias: 0.0,
            train_mean:  Vec::<f64>::new(),
            train_std:  Vec::<f64>::new(),
            normalization: false,
            fitted: false
        }
    }


    fn fit(&mut self, path_to_data: &String, num_iters: usize, learning_rate: f64, lambda_param: f64, normalization: bool, config: &EnvironmentConfig)
        
        {

        if normalization==true{
            (self.train_mean, self.train_std) = get_moments(&config, &path_to_data);
        }

        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();

        let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let fit = env.stream(source.clone())
        .replay(
            num_iters,
            StateSGD::new(),

            move |s, state| 
            {
                s
                //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                //.shuffle()
                .rich_filter_map({
                    let mut flag = 0;
                    let mut new_weights = Vec::<f64>::new();
                    let mut new_bias= 0.;
                    let mut count = 1;
                    let mut count2 = 1;
                    move |mut x|{
                        //to give more randomness to the sequential update of the weights each iteration we consider 90% of data
                        if rand::thread_rng().gen::<f64>() > 0.1 {
                            let dim = x.0.len();
                            let mut y: f64 = x.0[dim-1];
                            //let lambda_param = 1.;
                                if normalization==true{
                                    //scale the features and the target
                                    x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                    }
                                //the target is in the last element of each sample
                                x.0.pop();

                                //assign to the target -1 or 1 based on the class
                                if y==0.{
                                    y = -1.;
                                }
                                
                                //in the first sample "iteration" of the stream we set the final weights of the last global iteration
                                if flag == 0{
                                    new_weights = state.get().weights.clone();
                                    new_bias = state.get().bias;
                                    flag = 1;
                                }
    
                                //each replica update its new_weights with each sample
                                let mut current_weights = new_weights.clone();
                                
                                //if it is the first global iteration -> initialize
                                if state.get().epoch == 0{
                                    current_weights = vec![0.;dim-1];
                                    count +=1;
                                }
                                else{
                                    count2+=1;
                                }

                                let prediction: f64 = x.0.iter().zip(current_weights.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + new_bias;

                                if y * prediction >=1.{
                                    new_weights = current_weights.iter().map(|wi| wi - 2. * learning_rate * lambda_param * wi).collect();
                                }
                                else{
                                    new_weights = current_weights.iter().zip(x.0.iter()).map(|(wi,xi)| wi - 2. * learning_rate * (lambda_param * wi - 
                                    xi * y)).collect();
                                    new_bias -= learning_rate * (-y);
                                }
                                let mut weights = new_weights.clone();
                                //print!("\nweights: {:?}\n", weights);
                                weights.push(new_bias);
                                //print!("\nweights: {:?}\n", weights);
                                if state.get().epoch == 0{
                                    Some(Sample(weights))}
                                else{
                                    if count2==count{
                                        count2 =1;
                                        flag=0;
                                    Some(Sample(weights))
                                }
                                else{
                                    None
                                }
                            }}
                        else{
                            None
                        }
            }})
                //the average of the gradients is computed and forwarded as a single value
                .group_by_avg(|_x| true, |x| x.clone()).drop_key()//.max_parallelism(1)
            },

            move |local_weights: &mut Sample, weights| 
            {   
                if weights.0.len()!=0{
                *local_weights = weights;}
            },

            move |state, local_weights| 
            {   
                //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                if local_weights.0.len()!=0{
                state.weights = local_weights.0.iter().take(local_weights.0.len()-1).cloned().collect();
                state.bias = local_weights.0[local_weights.0.len()-1];}
            },

            move|state| 
            {   
                //update iterations
                 state.epoch +=1;
                state.epoch < num_iters
            },

        )
        .collect_vec();

    env.execute();

    let state = fit.get().unwrap()[0].clone();
    self.weights = state.weights;
    self.bias = state.bias;
    self.fitted = true;
    self.normalization = normalization;

    }


    fn score(&self, path_to_data: &String, config: &EnvironmentConfig) -> f64{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}
        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        let weights = self.weights.clone();
        let bias = self.bias.clone();
        let normalization = self.normalization;
        let score = env.stream(source)
    
            .map(move |mut x| {
                let dim = x.0.len();   
                let mut class = x.0[dim-1];
                if normalization==true{
                    x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                    }               
                
                x.0.pop();
                //we need -1 or 1 as in the training 
                if class == 0.{
                    class = -1.;
                }

                let y_hat:f64 = x.0.iter().zip(weights.iter()).map(|(xi, wi)| xi * wi).sum::<f64>() + bias;

                if (y_hat>=0. && class==1.) || (y_hat<0. && class==-1.){
                    1.
                }        
                else {
                    0.
                }
            })    
            .group_by_avg(|&_k| true, |&v| v).drop_key()
            .collect_vec();
        
        env.execute();
            
        let mut evaluation_score: f64 = -999.;
        if let Some(res3) = score.get() {
            evaluation_score = res3[0];}
        
        evaluation_score
    }
    
    
    
    fn predict(&self, path_to_data: &String, config: &EnvironmentConfig) -> Vec<f64>{

        if self.fitted != true {panic!("Can't compute predictions before fitting the model!");}
        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        let weights = self.weights.clone();
        let bias = self.bias.clone();
        let normalization = self.normalization;

        let predictions = env.stream(source)    
            .map(move |mut x| {
                let pred:f64;
                let dim = x.0.len();
                
                if normalization==true{
                    x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                    pred = x.0.iter().zip(weights.iter()).map(|(xi,wi)| xi*wi).sum::<f64>() + bias;
                }   
                else{                  
                pred = x.0.iter().zip(weights.iter()).map(|(xi,wi)| xi*wi).sum::<f64>() + bias;
                }

                if pred >=0.{
                    1.
                }    
                else {
                    0.
                }       
            })
            .collect_vec();
        
        env.execute();
        
        predictions.get().unwrap()
    }}



fn main() {
    let (config, _args) = EnvironmentConfig::from_args();
    let training_set = "data/class_10milion_4features_multiclass.csv".to_string();
    let data_to_predict = "data/class_1milion_4features_multiclass.csv".to_string();
 
    let mut model = SupportVectorMachine::new();
    
    //hyper_parameters for the iterative method model.fit
    let num_iters = 100;
    let learn_rate = 1e-3;
    let lambda_param = 1e-1;

    let normalize = false;

    let start = Instant::now();
    //return the trained model
    model.fit(&training_set, num_iters, learn_rate,  lambda_param, normalize , &config);

    let elapsed = start.elapsed();

    let start_score = Instant::now();
    //compute the score over the training set
    let score = model.score(&training_set, &config);
    let elapsed_score = start_score.elapsed();
    
    let start_pred = Instant::now();
    let predictions = model.clone().predict(&data_to_predict, &config);
    let elapsed_pred = start_pred.elapsed();
    

    print!("\nCoefficients: {:?}\n", model.weights);
    print!("Intercept: {:?}\n", model.bias);  
    print!("\nscore: {:?}\n", score);
    print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<f64>>());
    eprintln!("\nElapsed fit: {elapsed:?}");
    eprintln!("\nElapsed score: {elapsed_score:?}"); 
    eprintln!("\nElapsed pred: {elapsed_pred:?}");     

}

