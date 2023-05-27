use noir::prelude::*;
use serde::{Deserialize, Serialize};
use rand::Rng;
use crate::sample::Sample;
use crate::basic_stat::sigmoid;

////////////////////
//LINEAR REGRESSION
////////////////////
//State for ADAM
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct StateAdam {
    //regression coefficients
    pub weights: Vec<f64>,
    //total gradient of the batch
    global_grad: Vec<f64>,
    //first gradient moment
    m: Vec<f64>,
    //second gradient moment
    v: Vec<f64>, 
    //iterations over the dataset
    epoch: usize,}

impl StateAdam {
    pub fn new() -> StateAdam {
        StateAdam {
            weights:  Vec::<f64>::new(),
            global_grad: Vec::<f64>::new(),
            m: Vec::<f64>::new(),
            v: Vec::<f64>::new(),
            epoch : 0,
        }}}


pub fn linear_adam(weight_decay: bool, learn_rate: f64, data_fraction: f64, num_iters: usize, 
    path_to_data: &String, normalization: bool, train_mean: Vec<f64>, train_std: Vec<f64>, config: &EnvironmentConfig, regularization: &str, lambda: f64) 
    -> StateAdam {

        let reg_flag;
        match regularization {
            "lasso" | "LASSO" => reg_flag = 1,
            "ridge" | "RIDGE" => reg_flag = 2,
            "elasitc-net" => reg_flag = 3,
            _ => reg_flag = 0,
        }

        let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let beta1 = 0.9;
        let beta2 = 0.999;

        let fit = env.stream(source.clone())
            .replay(
            num_iters,
            StateAdam::new(),

            move |s, state| 
            {
                //shuffle the samples
                s
                //each replica filter a number of samples equal to batch size and
                //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                .rich_filter_map({
                    move |mut x|{
                        let dim = x.0.len();
                        //at first iter (epoch=0) count goes from 0 to batch_size; at epoch=1 from batchsize to 2*batch_size...
                        if rand::thread_rng().gen::<f64>() > (1.0 - data_fraction) {
                            if normalization==true{
                                //scale the features and the target
                                x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                }
                            //the target is in the last element of each sample
                            let y: f64 = x.0[dim-1]; 
                            //switch the target with a 1 for the intercept
                            x.0[dim-1] = 1.;
                            
                            let mut current_weights = &state.get().weights;
                            let vec = vec![0.;dim];
                            if state.get().epoch == 0{
                                current_weights = &vec;
                            }

                            let prediction: f64 = x.0.iter().zip(current_weights.iter()).map(|(xi, wi)| xi * wi).sum();
                            let error = prediction - y;
                            let sample_grad: Vec<f64> = x.0.iter().map(|xi| xi * error).collect();                            
                            let grad; 
                            match reg_flag{
                                //lasso
                                1 => grad = Sample(current_weights.iter().zip(sample_grad.iter()).map(|(wi,gi)| gi + if *wi>=0. {lambda} else {-lambda}).collect()),
                                //ridge
                                2 => grad = Sample(current_weights.iter().zip(sample_grad.iter()).map(|(wi,gi)| gi + wi * lambda).collect()),
                                //elastic-net
                                3 => grad = Sample(current_weights.iter().zip(sample_grad.iter()).map(|(wi,gi)| gi + wi * lambda + if *wi>=0. {lambda} else {-lambda}).collect()),
                                //no regularization
                                _ => grad = Sample(sample_grad),
                            }
                            
                            Some(grad)     
                            }
                        else {None}}})
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
                    let dim = state.global_grad.len();
                    state.m = vec![0.;dim];
                    state.v = vec![0.;dim];
                    state.weights = vec![0.;dim];
                }
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
                state.global_grad = vec![0.;adam.len()];
                //loop condition
                state.epoch < num_iters && tol.sqrt() > 1e-4
            },

        )
        .collect_vec();

        env.execute();

        let state = fit.get().unwrap()[0].clone();
        state
}





/////////////////////
//LOGISTIC REGRESSION
/////////////////////
//State for logistic ADAM
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct StateAdamLogistic {
    //regression coefficients
    pub weights: Vec<Vec<f64>>,
    //total gradient of the batch
    global_grad: Vec<Vec<f64>>,
    //first gradient moment
    m: Vec<Vec<f64>>,
    //second gradient moment
    v: Vec<Vec<f64>>, 
    //iterations over the dataset
    epoch: usize,}

impl StateAdamLogistic {
    fn new() -> StateAdamLogistic {
        StateAdamLogistic {
            weights:  Vec::new(),
            global_grad: Vec::new(),
            m: Vec::new(),
            v: Vec::new(),
            epoch : 0,
        }}}


pub fn logistic_adam(num_classes: usize, weight_decay: bool, learn_rate: f64, data_fraction: f64, num_iters: usize, 
    path_to_data: &String, normalization: bool, train_mean: Vec<f64>, train_std: Vec<f64>, config: &EnvironmentConfig) 
    -> StateAdamLogistic {

        let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let beta1 = 0.9;
        let beta2 = 0.999;

        let fit = env.stream(source.clone())
                    .replay(
                    num_iters,
                    StateAdamLogistic::new(),

                    move |s, state| 
                    {
                        //shuffle the samples
                        s
                        //each replica filter a number of samples equal to batch size and
                        //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                        .rich_filter_map({
                            move |mut x|{
                                let dim = x.0.len();
                                //at first iter (epoch=0) count goes from 0 to batch_size; at epoch=1 from batchsize to 2*batch_size...
                                if rand::thread_rng().gen::<f64>() > (1.0 - data_fraction) {
                                    //the target is in the last element of each sample, y one hot encoding
                                    let mut y = vec![0;num_classes];
                                    y[x.0[dim-1] as usize] = 1; //assigned before normalization because it's a classification task
                                    if normalization==true{
                                        //scale the features and the target
                                        x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                        };
                                    //switch the target with a 1 for the intercept
                                    x.0[dim-1] = 1.;

                                    let mut current_weights = &state.get().weights;
                                    let vec = vec![vec![0.;dim]; num_classes];
                                    if state.get().epoch == 0{
                                        current_weights = &vec;
                                    }

                                    let mut prediction = vec![0.;num_classes];
                                    let mut sample_grad = Vec::new();
                                    for i in 0..num_classes{
                                        let y_hat:f64 = x.0.iter().zip(current_weights[i].iter()).map(|(xi, wi)| xi * wi).sum();
                                        prediction[i] = sigmoid(y_hat);
                                        sample_grad.extend(x.0.iter().map(|xi| xi * (prediction[i] - y[i] as f64)).collect::<Vec::<f64>>());
                                    }
                                     
                                    Some(Sample(sample_grad))
                                    }
                                else {None}}})
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
                            let dim = local_grad.0.len()/num_classes;
                            //we have to push every dim elements, since Sample type is a Vec and not Vec<vec<f64>>
                            for i in 0..num_classes{
                                state.global_grad.push(local_grad.0.iter().skip(i*dim).take(dim).cloned().collect());
                            }
                            // state.global_grad.extend(local_grad.0)??
                        }
                    },

                    move|state| 
                    {   
                        //initialize
                        if state.epoch==0{
                            let dim = state.global_grad[0].len();
                            state.weights = vec![vec![0.;dim];num_classes];
                            state.m = vec![vec![0.;dim];num_classes];
                            state.v = vec![vec![0.;dim];num_classes];
                        }
                        //update iterations
                        state.epoch +=1;
                        //update the weights (optional with weight decay)
                        for i in 0..num_classes{
                            state.m[i] = state.m[i].iter().zip(state.global_grad[i].iter()).map(|(mi, gi)|beta1 * mi + ((1. - beta1) * gi)).collect();
                            state.v[i] =  state.v[i].iter().zip(state.global_grad[i].iter()).map(|(vi, gi)|beta2 * vi + ((1. - beta2) * gi.powi(2))).collect();
                            let m_hat: Vec<f64> = state.m[i].iter().map(|mi|mi/(1. - beta1.powi(state.epoch as i32))).collect();
                            let v_hat: Vec<f64> = state.v[i].iter().map(|vi|vi/(1. - beta2.powi(state.epoch as i32))).collect();
                            let adam: Vec<f64> =  m_hat.iter().zip(v_hat.iter()).map(|(mi,vi)| mi/(vi.sqrt()+1e-6)).collect();
                            //update the weights (optional with weight decay)
                            state.weights[i] = state.weights[i].iter().zip(adam.iter()).map(|(wi,a)| wi - learn_rate*a).collect();
                            if weight_decay==true{
                                state.weights[i] = state.weights[i].iter().map(|wi| wi -  learn_rate * 0.002 * wi).collect();
                            }
                        }
                        //tolerance=gradient's L2 norm for the stop condition
                        //let tol: f64 = adam.iter().map(|v| v*v).sum();
                        //reset the global gradient for the next iteration
                        state.global_grad = Vec::new();
                        //loop condition
                        state.epoch < num_iters //&& tol.sqrt() > 1e-4
                    },

                )
                .collect_vec();

        env.execute();

        let state = fit.get().unwrap()[0].clone();
        state
}