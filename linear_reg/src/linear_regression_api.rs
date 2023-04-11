use noir::prelude::*;

use serde::{Deserialize, Serialize};

use std::{time::Instant};

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

#[derive(Clone, Debug)]
struct LinearRegression {
    coefficients: Vec<f64>,
    intercept: f64,
    score: f64,
    normalization: bool,
    train_mean: Vec<f64>,
    train_std: Vec<f64>,
    num_features: usize,
}


impl LinearRegression {
    fn new(num_features: usize) -> LinearRegression {

        LinearRegression {
            coefficients: vec![0.;num_features+1],
            intercept: 0.,
            score: 0.,
            normalization: false,
            train_mean: vec![0.;num_features+1],
            train_std: vec![0.;num_features+1],
            num_features: num_features,
        }}
}


impl LinearRegression {
    fn fit(mut self, path_to_data: String, num_iters:usize, learn_rate: f64, batch_size: usize, normalize: bool, l2_reg: bool, config:EnvironmentConfig)
    -> LinearRegression{

            self.normalization = true;

            if normalize==true{
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
                moments = means_vector[0].0.clone();}
            
                self.train_mean= moments.iter().take(self.num_features+1).cloned().collect::<Vec<f64>>();
            
                self.train_std = moments.iter().skip(self.num_features+1).zip(self.train_mean.iter()).map(|(e2,avg)| (e2-avg.powi(2)).sqrt()).collect();
         }


        let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
    
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
            self.coefficients = state.weights.iter().take(self.num_features).cloned().collect();
            self.intercept = state.weights[self.num_features];}

            LinearRegression {
                coefficients: self.coefficients,
                intercept: self.intercept,
                score: 0.,
                normalization: self.normalization,
                train_mean: train_mean,
                train_std: train_std,
                num_features: self.num_features,
            }
    }
    }

        

impl LinearRegression {
    fn score(mut self, path_to_data: String, config:EnvironmentConfig) -> f64{

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

        let mut env = StreamEnvironment::new(config);

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
