use noir::prelude::*;

use serde::{Deserialize, Serialize};

use std::{time::Instant};

mod sample;
use sample::Sample;


#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Serialize, Deserialize, Default)]
struct State {
    //regression coefficients
    weights: Vec<f64>,
    //total gradient of the batch
    global_grad: Vec<f64>,
    //iterations over the dataset
    epoch: usize,
}

impl State {
    fn new(n_features: usize) -> State {
        State {
            weights:  vec![0.;n_features+1],
            global_grad: vec![0.;n_features+1],
            epoch : 0,
        }}}


#[derive(Clone, Serialize, Deserialize, Default)]
struct StateAdam {
    //regression coefficients
    weights: Vec<f64>,
    //total gradient of the batch
    global_grad: Vec<f64>,
    //first gradient moment
    m: Vec<f64>,
    //second gradient moment
    v: Vec<f64>, 
    //iterations over the dataset
    epoch: usize,}

impl StateAdam {
    fn new(n_features: usize) -> StateAdam {
        StateAdam {
            weights:  vec![0.;n_features+1],
            global_grad: vec![0.;n_features+1],
            m: vec![0.;n_features+1],
            v: vec![0.;n_features+1],
            epoch : 0,
        }}}



#[derive(Clone, Debug)]
struct LinearRegression {
    coefficients: Vec<f64>,
    features_coef: Vec<f64>,
    intercept: f64,
    score: f64,
    normalization: bool,
    train_mean: Vec<f64>,
    train_std: Vec<f64>,
    num_features: usize,
    fitted: bool,
}


impl LinearRegression {
    fn new(num_features: usize) -> LinearRegression {

        LinearRegression {
            coefficients: vec![0.;num_features+1],
            features_coef: vec![0.;num_features],
            intercept: 0.,
            score: 0.,
            normalization: false,
            train_mean: vec![0.;num_features+1],
            train_std: vec![0.;num_features+1],
            num_features: num_features,
            fitted: false,
        }}
}


impl LinearRegression {
    fn fit(mut self, path_to_data: &String, method: String, num_iters:usize, learn_rate: f64, batch_size: usize, normalize: bool, l2_reg: bool, weight_decay: bool, config: &EnvironmentConfig)
    -> LinearRegression{
            
            self.fitted = true;

            if normalize==true{
            self.normalization = true;
            let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
            let mut env0 = StreamEnvironment::new(config.clone());
            env0.spawn_remote_workers();
            
            //get the mean of all the features + target and the second moment E[x^2]
            let features_mean = env0.stream(source)
            .map(move |mut x| 
                {
                    x.0.extend(x.0.iter().map(|xi| xi.powi(2)).collect::<Vec<f64>>());
                    x
                })
            .group_by_avg(|_x| true, |x| x.clone()).drop_key().collect_vec();

            env0.execute();
            
            let mut moments:Vec<f64> = vec![0.;2*self.num_features+2];
            if let Some(means_vector) = features_mean.get() {
                if moments.len() != means_vector[0].0.len(){panic!("Wrong number of features!");}
                moments = means_vector[0].0.clone();
            }
            
            self.train_mean= moments.iter().take(self.num_features+1).cloned().collect::<Vec<f64>>();
            
            self.train_std = moments.iter().skip(self.num_features+1).zip(self.train_mean.iter()).map(|(e2,avg)| (e2-avg.powi(2)).sqrt()).collect();
         }


        let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();

        match  method.as_str(){


            "ADAM" | "adam"  =>

            {    
                let beta1 = 0.9;
                let beta2 = 0.999;

                let fit = env.stream(source.clone())
                    .replay(
                    num_iters,
                    StateAdam::new(self.num_features),

                    move |s, state| 
                    {
                        //shuffle the samples
                        s.shuffle()
                        //each replica filter a number of samples equal to batch size and
                        //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                        .rich_filter_map({
                            let mut count = 0;
                            move |mut x|{
                                //at first iter (epoch=0) count goes from 0 to batch_size; at epoch=1 from batchsize to 2*batch_size...
                                if count < batch_size * (state.get().epoch+1) {
                                    count+=1;
                                    if self.normalization==true{
                                        //scale the features and the target
                                        x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                        }
                                    //the target is in the last element of each sample
                                    let y: f64 = x.0[self.num_features]; 
                                    //switch the target with a 1 for the intercept
                                    x.0[self.num_features] = 1.;
                                        let current_weights = &state.get().weights;
                                        let prediction: f64 = x.0.iter().zip(current_weights.iter()).map(|(xi, wi)| xi * wi).sum();
                                        let error = prediction - y;
                                        let sample_grad: Vec<f64> = x.0.iter().map(|xi| xi * error).collect();                            
                                        Some(Sample(sample_grad))
                                    }
                                else {None}}})
                        //the average of the gradients is computed and forwarded as a single value
                        .group_by_avg(|_x| true, |x| x.clone()).drop_key().max_parallelism(1)
                    },

                    move |local_grad: &mut Sample, avg_grad| 
                    {   
                        *local_grad = avg_grad;
                    },

                    move |state, local_grad| 
                    {   
                        //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                        if local_grad.0.len()==self.num_features+1{
                        state.global_grad = local_grad.0.clone();}
                    },

                    move|state| 
                    {   
                        //update iterations
                        state.epoch +=1;
                        //update the moments
                        state.m = state.m.iter().zip(state.global_grad.iter()).map(|(mi, gi)|beta1 * mi + ((1. - beta1) * gi)).collect();
                        state.v =  state.v.iter().zip(state.global_grad.iter()).map(|(vi, gi)|beta2 * vi + ((1. - beta2) * gi.powi(2))).collect();
                        let m_hat: Vec<f64> = state.m.iter().map(|mi|mi/(1. - beta1.powi(state.epoch as i32))).collect();
                        let v_hat: Vec<f64> = state.v.iter().map(|vi|vi/(1. - beta2.powi(state.epoch as i32))).collect();
                        let adam: Vec<f64> =  m_hat.iter().zip(v_hat.iter()).map(|(mi,vi)| mi/(vi.sqrt()+1e-6)).collect();
                        //update the weights (optional with weight decay)
                        state.weights = state.weights.iter().zip(adam.iter()).map(|(wi,a)| wi - learn_rate*a).collect();
                        if weight_decay==true{
                            state.weights = state.weights.iter().map(|wi| wi -  learn_rate * 0.002 * wi).collect();
                        }
                        //tolerance=gradient's L2 norm for the stop condition
                        let tol: f64 = adam.iter().map(|v| v*v).sum();
                        //reset the global gradient for the next iteration
                        state.global_grad = vec![0.;self.num_features+1];
                        //loop condition
                        state.epoch < num_iters && tol.sqrt() > 1e-4
                    },

                )
                .collect_vec();

        env.execute();

        if let Some(res) = fit.get() {
            let state = &res[0];
            self.coefficients = state.weights.clone();
            self.intercept = state.weights[self.num_features];}

            LinearRegression {
                coefficients: self.coefficients.clone(),
                features_coef: self.coefficients.iter().take(self.num_features).cloned().collect::<Vec::<f64>>(),
                intercept: self.intercept,
                score: 0.,
                normalization: self.normalization,
                train_mean: train_mean,
                train_std: train_std,
                num_features: self.num_features,
                fitted: self.fitted,
            }
            }
            "SGD" | _ => 

            {
                //return the weights computed with SGD thanks to the model.fit method
             let fit = env.stream(source.clone())
                .replay(
                    num_iters,
                    State::new(self.num_features),

                    move |s, state| 
                    {
                        //shuffle the samples
                        s.shuffle()
                        //each replica filter a number of samples equal to batch size and
                        //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                        .rich_filter_map({
                            let mut count = 0;
                            move |mut x|{
                                //at first iter (epoch=0) count goes from 0 to batch_size; at epoch=1 from batchsize to 2*batch_size...
                                if count < batch_size * (state.get().epoch+1) {
                                    count+=1;
                                    if self.normalization==true{
                                        //scale the features and the target
                                        x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                        }
                                    //the target is in the last element of each sample
                                    let y: f64 = x.0[self.num_features]; 
                                    //switch the target with a 1 for the intercept
                                    x.0[self.num_features] = 1.;
                                        let current_weights = &state.get().weights;
                                        let prediction: f64 = x.0.iter().zip(current_weights.iter()).map(|(xi, wi)| xi * wi).sum();
                                        let error = prediction - y;
                                        let mut sample_grad: Vec<f64> = x.0.iter().map(|xi| xi * error).collect();
                                        if l2_reg==true{
                                            sample_grad = sample_grad.iter().zip(current_weights.iter()).map(|(gi,wi)| gi + 0.001 * wi).collect()
                                        }
                                        Some(Sample(sample_grad))
                                    }
                                else {None}}})
                        //the average of the gradients is computed and forwarded as a single value
                        .group_by_avg(|_x| true, |x| x.clone()).drop_key().max_parallelism(1)
                    },

                    move |local_grad: &mut Sample, avg_grad| 
                    {   
                        if avg_grad.0.len() != self.num_features+1 {panic!("Wrong number of features!");}
                        *local_grad = avg_grad;
                    },

                    move |state, local_grad| 
                    {   
                        //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                        if local_grad.0.len()==self.num_features+1{
                        state.global_grad = local_grad.0.clone();}
                    },

                    move|state| 
                    {   
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
                        state.global_grad = vec![0.;self.num_features+1];
                        //loop condition
                        state.epoch < num_iters && tol.sqrt() > 1e-4
                    },

                )
                .collect_vec();

        env.execute();

        if let Some(res) = fit.get() {
            let state = &res[0];
            self.coefficients = state.weights.clone();
            self.intercept = state.weights[self.num_features];}

            LinearRegression {
                coefficients: self.coefficients.clone(),
                features_coef: self.coefficients.iter().take(self.num_features).cloned().collect::<Vec::<f64>>(),
                intercept: self.intercept,
                score: 0.,
                normalization: self.normalization,
                train_mean: train_mean,
                train_std: train_std,
                num_features: self.num_features,
                fitted: self.fitted,
            }

            }

           
        }
    

        

    }
    }

        
//score takes as input also the target, a dataset with row of length num_features+1
//can be used for evaluating both training and test set
impl LinearRegression {
    fn score(mut self, path_to_data: &String, config: &EnvironmentConfig) -> f64{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}

        let mut avg_y = 0.;
        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');

        if self.normalization == false
        {

            let mut env = StreamEnvironment::new(config.clone());
            env.spawn_remote_workers();
        
            let res2 = env.stream(source.clone())
            .group_by_avg(|_x| true, move|x| x.0[self.num_features]).drop_key().collect_vec();
        
            env.execute();
        
            if let Some(res2) = res2.get() {
            avg_y = res2[0];}
        }

        let mut env = StreamEnvironment::new(config.clone());

        env.spawn_remote_workers();
    
        //compute the residuals sums for the R2
        let score = env.stream(source)
    
            .map(move |mut x| {
                let mut mean_y = avg_y;
                //scale the features and the target   
                if self.normalization==true{
                    x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                    mean_y = 0.; 
                    }                     
                let y = x.0[self.num_features];
                x.0[self.num_features] = 1.;   
                let pred: f64 = x.0.iter().zip(self.coefficients.iter()).map(|(xi,wi)| xi*wi).sum();
                [(y-pred).powi(2),(y-mean_y).powi(2)]           
            })
    
            .fold_assoc([0.,0.],
                |acc,value| {acc[0]+=value[0];acc[1]+=value[1];}, 
                |acc,value| {acc[0]+=value[0];acc[1]+=value[1];})
            .collect_vec();
    
        
        env.execute();
            
        let mut r2 = -999.;
        if let Some(res3) = score.get() {
            r2 = 1.-(res3[0][0]/res3[0][1]);}
        self.score = r2;
        
        r2

    }
        }


//predict doesn't take as input the target, so a dataset with row of length num_features
impl LinearRegression {
    fn predict(self, path_to_data: &String, config: &EnvironmentConfig) -> Vec<f64>{

        if self.fitted != true {panic!("Can't compute predictions before fitting the model!");}

        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');

        let mut env = StreamEnvironment::new(config.clone());

        env.spawn_remote_workers();
        
        let y_mean = self.train_mean[self.num_features];
        let y_std = self.train_std[self.num_features];

        let prediction = env.stream(source)
    
            .map(move |mut x| {
                let pred:f64;
                x.0.push(1.); //push the intercept
                if self.normalization==true{
                    x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                    pred = x.0.iter().zip(self.coefficients.iter()).map(|(xi,wi)| xi*wi).sum::<f64>() * y_std + y_mean;
                }   
                else{                  
                pred = x.0.iter().zip(self.coefficients.iter()).map(|(xi,wi)| xi*wi).sum();
                }
                pred           
            })
            .collect_vec();
        
        env.execute();

        let mut pred = vec![0.;1];
        if let Some(res3) = prediction.get() {
            pred = res3;}
        
        pred

    }
        }




fn main() {
    
    let (config, _args) = EnvironmentConfig::from_args();

    let num_features = 5;
    let training_set = "housing_numeric.csv".to_string();
    //let test_set = "housing_numeric.csv".to_string();
    let data_to_predict = "housing_numeric.csv".to_string();

    let start = Instant::now();

    let mut model = LinearRegression::new(num_features);
    
    let method = "SGD".to_string(); //"ADAM".to_string()
    let num_iters = 1000;
    let learn_rate = 1e-2;
    let batch_size = 500;
    let normalize = true;
    let l2_reg = false;
    let weight_decay = false;

    model = model.fit(&training_set, method, num_iters, learn_rate, batch_size, normalize, l2_reg, weight_decay, &config);

    let r2 = model.clone().score(&training_set, &config);

    //let r2_test = model.clone().score(&test_set, &config);

    let predictions = model.clone().predict(&data_to_predict, &config);

    let elapsed = start.elapsed();

    print!("\nCoefficients: {:?}\n", model.features_coef);
    print!("Intercept: {:?}\n", model.intercept);  

    print!("\nR2 score: {:?}\n", r2);
    //print!("\nTest score: {:?}\n", r2_test);

    print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<f64>>());


    eprintln!("\nElapsed: {elapsed:?}");

    
}