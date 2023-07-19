use noir::prelude::*;

use serde::{Deserialize, Serialize};
use rand::Rng;
//use crate::sample::{Sample, SampleArray};
use crate::sample::Sample;
use crate::basic_stat::sigmoid;
//use ndarray::Array1;

////////////////////
//LINEAR REGRESSION
////////////////////
//State for SGD

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct StateSGD {
    //regression coefficients
    pub weights: Vec<f64>,
    //total gradient of the batch
    global_grad: Vec<f64>,
    //iterations over the dataset
    epoch: usize,
    //best square loss for early stopping
    best_loss: f64,
    //n_iter_no_change
    n_iter_early_stopping: usize,
    //best coefficients
    pub best_weights: Vec<f64>,
}

impl StateSGD {
    pub fn new() -> StateSGD {
        StateSGD {
            weights:  Vec::<f64>::new(),
            global_grad: Vec::<f64>::new(),
            epoch : 0,
            best_loss: f64::MAX,
            n_iter_early_stopping : 0,
            best_weights:  Vec::<f64>::new(),
        }}}



pub fn linear_batch_gd(weight_decay: bool, learn_rate: f64, data_fraction: f64, num_iters: usize, 
    path_to_data: &String, tol: f64, n_iter_no_change:usize, normalization: bool, train_mean: Vec<f64>, train_std: Vec<f64>, config: &EnvironmentConfig, regularization: &str, lambda: f64) 
    -> StateSGD {


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
        
        let fit = env.stream(source.clone())
        .replay(
            num_iters,
            StateSGD::new(),

            move |s, state| 
            {
                //shuffle the samples
                s
                //each replica filter a number of samples equal to batch size and
                //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                .rich_filter_map({
                    let mut flag_at_least_one = 0;
                    move |mut x|{
                        let dim = x.0.len();
                        //each iteration just a fraction of data is considered
                        if rand::thread_rng().gen::<f64>() > (1.0 - data_fraction) || flag_at_least_one == state.get().epoch{
                            //make sure at each iteration at least a sample is passed forward
                            if flag_at_least_one == state.get().epoch{
                                flag_at_least_one += 1;
                            }
                            
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
                            let mut grad; 
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
                            //early stopping: we need the loss
                            if tol!=0.{
                            grad.0.push(error.powi(2));} //push the square loss for early stopping
                            Some(grad)          

                        }
                        else{
                            None
                        }
            }})
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
                if state.epoch==0{
                    state.weights = vec![0.;state.global_grad.len()];
                }

                if tol!=0.{
                let loss = state.global_grad.pop().unwrap();
                //initialize

                //early stopping if for tot iters the loss doesn't improve
                if loss > state.best_loss - tol && tol!=0.{
                    state.n_iter_early_stopping+=1;
                }
                else{
                    state.n_iter_early_stopping=0;
                }

                if state.best_loss>loss{
                    state.best_loss = loss;
                    state.best_weights = state.weights.clone();
                }
                }
                //update iterations
                 state.epoch +=1;
                //update the weights (optional with weight decay)
                state.weights = state.weights.iter().zip(state.global_grad.iter()).map(|(wi,g)| wi - learn_rate*g).collect();
                if weight_decay==true{
                    state.weights = state.weights.iter().map(|wi| wi -  learn_rate * 0.002 * wi).collect();
                }
                //tolerance=gradient's L2 norm for the stop condition
                //let grad_tol: f64 = state.global_grad.iter().map(|v| v*v).sum();

                //loop condition
                if state.n_iter_early_stopping >= n_iter_no_change {
                    print!("Early Stopping at iter: {:?}", state.epoch);
                }

                state.epoch < num_iters && state.n_iter_early_stopping < n_iter_no_change 
            },

        )
        .collect_vec();

    env.execute();

    let state = fit.get().unwrap()[0].clone();
    state
}





pub fn linear_sgd(weight_decay: bool, learn_rate: f64, num_iters: usize, 
    path_to_data: &String, normalization: bool, train_mean: Vec<f64>, train_std: Vec<f64>, config: &EnvironmentConfig, regularization: &str, lambda: f64) 
    -> StateSGD {

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
        let fit = env.stream(source.clone())
        .replay(
            num_iters,
            StateSGD::new(),

            move |s, state| 
            {
                //shuffle the samples
                s
                //each replica filter a number of samples equal to batch size and
                //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                .rich_filter_map({
                    let mut count = 0;
                    let mut count2 = 0;
                    let mut weights = vec![0.;1];
                    let mut flag = 0;
                    move |mut x|{
                        let dim = x.0.len();
                        if state.get().epoch == 0{
                            count+=1;
                            weights = vec![0.;dim];

                            None
                        }
                        else{
                        //if rand::thread_rng().gen::<f64>() > 0.5{
                        if normalization==true{
                            //scale the features and the target
                            x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
                        }
                        

                        if count2==0 && flag!=0{
                            weights = state.get().weights.clone();         
                        }
                        if flag ==0{
                            flag = 1;
                            weights = vec![0.;dim];
                        }

                        count2+=1;
                        //the target is in the last element of each sample
                        let y: f64 = x.0[dim-1]; 
                        //switch the target with a 1 for the intercept
                        x.0[dim-1] = 1.;

                        let prediction: f64 = x.0.iter().zip(weights.iter()).map(|(xi, wi)| xi * wi).sum();
                        let error = prediction - y;
                        let sample_grad: Vec<f64> = x.0.iter().map(|xi| xi * error).collect();
                        let grad; 
                        match reg_flag{
                            //lasso
                            1 => grad = weights.iter().zip(sample_grad.iter()).map(|(wi,gi)| gi + if *wi>=0. {lambda} else {-lambda}).collect(),
                            //ridge
                            2 => grad = weights.iter().zip(sample_grad.iter()).map(|(wi,gi)| gi + wi * lambda).collect(),
                            //elastic-net
                            3 => grad = weights.iter().zip(sample_grad.iter()).map(|(wi,gi)| gi + wi * lambda + if *wi>=0. {lambda} else {-lambda}).collect(),
                            //no regularization
                            _ => grad = sample_grad,
                        }
                        weights = weights.iter().zip(grad).map(|(wi,g)| wi - learn_rate*g).collect();
                        
                        if count2==count{
                            count2 = 0;
                            Some(Sample(weights.clone()))

                        }
                        else{
                            None
                        }
                    //}
                    //else{
                        //None
                    //}
                            
                    }
            }})
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
                if local_grad.0.len()!=0 && state.epoch!=0{
                    state.weights = local_grad.0.clone();}
            },

            move|state| 
            {   

                if weight_decay==true{
                    state.weights = state.weights.iter().map(|wi| wi -  learn_rate * 0.002 * wi).collect();
                }

                 state.epoch +=1;

                state.epoch < num_iters
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
//State for logistic SGD
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct StateSgdLogistic {
    //regression coefficients
    pub weights: Vec<Vec<f64>>,
    //total gradient of the batch
    global_grad: Vec<Vec<f64>>,
    //iterations over the dataset
    epoch: usize,
}

impl StateSgdLogistic {
    fn new() -> StateSgdLogistic {
        StateSgdLogistic {
            weights:  Vec::new(),
            global_grad: Vec::new(),
            epoch : 0,
        }}}



pub fn logistic_sgd(num_classes: usize, weight_decay: bool, learn_rate: f64, data_fraction: f64, num_iters: usize, 
    path_to_data: &String, normalization: bool, train_mean: Vec<f64>, train_std: Vec<f64>, config: &EnvironmentConfig) 
    -> StateSgdLogistic {

    let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
    let mut env = StreamEnvironment::new(config.clone());
    env.spawn_remote_workers();
    let fit = env.stream(source.clone())
    .replay(
        num_iters,
        StateSgdLogistic::new(),

        move |s, state| 
        {
            s
            //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
            .rich_filter_map({
                let mut flag_at_least_one = 0;
                move |mut x|{
                    let dim = x.0.len();
                    //each iteration just a fraction of data is considered
                    if rand::thread_rng().gen::<f64>() > (1.0 - data_fraction) || flag_at_least_one == state.get().epoch{
                        //make sure at each iteration at least a sample is passed forward
                        if flag_at_least_one == state.get().epoch{
                            flag_at_least_one += 1;
                        }
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
            //print!("\nGRAD {:?}\n", local_grad);
            //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
            if local_grad.0.len()!=0{
                let dim = local_grad.0.len()/num_classes;
                //we have to push every dim elements, since Sample type is a Vec and not Vec<vec<f64>>
                for i in 0..num_classes{
                    state.global_grad.push(local_grad.0.iter().skip(i*dim).take(dim).cloned().collect());
                }
                // state.global_grad.extend(local_grad.0)??
            }
            //print!("\nGRAD {:?}\n", state.global_grad);
        },

        move|state| 
        {   
            //initialize
            if state.epoch==0{
                state.weights = vec![vec![0.;state.global_grad[0].len()];num_classes];
            }
            //update iterations
            state.epoch +=1;
            //update the weights (optional with weight decay)
            for i in 0..num_classes{
                state.weights[i] = state.weights[i].iter().zip(state.global_grad[i].iter()).map(|(wi,g)| wi - learn_rate*g).collect();
                if weight_decay==true{
                    state.weights[i] = state.weights[i].iter().map(|wi| wi -  learn_rate * 0.002 * wi).collect();
                }
            }
            //tolerance=gradient's L2 norm for the stop condition
            //let tol: f64 = state.global_grad.iter().map(|v| v*v).sum();
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








//Linear regression with ndarray

// #[derive(Clone, Serialize, Deserialize, Default)]
// pub struct StateSGDArray {
//     //regression coefficients
//     pub weights: Array1<f64>,
//     //total gradient of the batch
//     global_grad: Array1<f64>,
//     //iterations over the dataset
//     epoch: usize,
// }

// impl StateSGDArray {
//     pub fn new() -> StateSGDArray {
//         StateSGDArray {
//             weights:  Array1::default(0),
//             global_grad: Array1::default(0),
//             epoch : 0,
//         }}}


// pub fn linear_sgd_array(weight_decay: bool, learn_rate: f64, data_fraction: f64, num_iters: usize, 
//     path_to_data: &String, normalization: bool, train_mean: Vec<f64>, train_std: Vec<f64>, config: &EnvironmentConfig, regularization: &str, lambda: f64) 
//     -> StateSGDArray {

//         let reg_flag;
//         match regularization {
//             "lasso" | "LASSO" => reg_flag = 1,
//             "ridge" | "RIDGE" => reg_flag = 2,
//             "elasitc-net" => reg_flag = 3,
//             _ => reg_flag = 0,
//         }

//         let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
//         let mut env = StreamEnvironment::new(config.clone());
//         env.spawn_remote_workers();
//         let fit = env.stream(source.clone())
//         .replay(
//             num_iters,
//             StateSGDArray::new(),

//             move |s, state| 
//             {
//                 //shuffle the samples
//                 s
//                 //each replica filter a number of samples equal to batch size and
//                 //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
//                 .rich_filter_map({
//                     let mut flag_at_least_one = 0;
//                     move |x2|{
//                         let mut x = SampleArray(Array1::from_vec(x2.0));
//                         let dim = x.0.len();
//                         //each iteration just a fraction of data is considered
//                         if rand::thread_rng().gen::<f64>() > (1.0 - data_fraction) || flag_at_least_one == state.get().epoch{
//                             //make sure at each iteration at least a sample is passed forward
//                             if flag_at_least_one == state.get().epoch{
//                                 flag_at_least_one += 1;
//                             }
                            
//                             if normalization==true{
//                                 //scale the features and the target
//                                 //x = (x-train_mean)/train_std;
//                                 }
//                             //the target is in the last element of each sample
//                             let y: f64 = x.0[dim-1]; 
//                             //switch the target with a 1 for the intercept
//                             x.0[dim-1] = 1.;

//                             let current_weights = &state.get().weights;

//                             let mut prediction = 0.;
//                             if state.get().epoch != 0{
//                                 prediction = x.0.dot(current_weights);
//                             } 

//                             let error = prediction - y;
//                             let sample_grad = Sample((x.0*error).to_vec());
//                             //let grad; 
//                             // match reg_flag{
//                             //     //lasso
//                             //     1 => grad = Sample(current_weights.iter().zip(sample_grad.iter()).map(|(wi,gi)| gi + if *wi>=0. {lambda} else {-lambda}).collect()),
//                             //     //ridge
//                             //     2 => grad = Sample(current_weights.iter().zip(sample_grad.iter()).map(|(wi,gi)| gi + wi * lambda).collect()),
//                             //     //elastic-net
//                             //     3 => grad = Sample(current_weights.iter().zip(sample_grad.iter()).map(|(wi,gi)| gi + wi * lambda + if *wi>=0. {lambda} else {-lambda}).collect()),
//                             //     //no regularization
//                             //     _ => grad = Sample(sample_grad),
//                             // }

                            
//                             Some(sample_grad)          

//                         }
//                         else{
//                             None
//                         }
//             }})
//                 //the average of the gradients is computed and forwarded as a single value
//                 .group_by_avg(|_x| true, |x| x.clone()).drop_key().max_parallelism(1)
//             },

//             move |local_grad: &mut SampleArray, avg_grad| 
//             {   
//                 if avg_grad.0.len()!=0{
//                 *local_grad = SampleArray(Array1::from_vec(avg_grad.0));}
//             },

//             move |state, local_grad| 
//             {   
//                 //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
//                 if local_grad.0.len()!=0{
//                 state.global_grad = local_grad.0.clone();}
//             },

//             move|state| 
//             {   
//                 //initialize
//                 if state.epoch==0{
//                     state.weights = Array1::from_vec(vec![0.;state.global_grad.len()]);
//                 }
//                 //update iterations
//                  state.epoch +=1;
//                 //update the weights (optional with weight decay)
//                 state.weights = &state.weights-learn_rate*&state.global_grad;
//                 if weight_decay==true{
//                     state.weights = state.weights.clone() -  learn_rate * 0.002 * state.weights.clone();
//                 }
//                 //tolerance=gradient's L2 norm for the stop condition
//                 let tol: f64 = state.global_grad.iter().map(|v| v*v).sum();
//                 //reset the global gradient for the next iteration
//                 //state.global_grad = vec![0.;state.weights.len()];
//                 //loop condition
//                 state.epoch < num_iters && tol.sqrt() > 1e-4
//             },

//         )
//         .collect_vec();

//     env.execute();

//     let state = fit.get().unwrap()[0].clone();
//     state
// }
