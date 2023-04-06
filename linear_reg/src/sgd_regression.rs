use noir::prelude::*;

use serde::{Deserialize, Serialize};

use std::time::Instant;




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


fn main() {
    let (config, args) = EnvironmentConfig::from_args();

    //args: path_to_data, n_features, n_iters, learn_rate, batch_size,
    let path_to_data: String;
    let num_features: usize;
    let mut num_iters = 1000;
    let mut learn_rate= 1e-3;
    let mut batch_size= 100;
    

    match args.len() {

        2 => {path_to_data = args[0].parse().expect("Invalid file path");
             num_features = args[1].parse().expect("Invalid number of features");}

        3 => {path_to_data = args[0].parse().expect("Invalid file path");
             num_features = args[1].parse().expect("Invalid number of features");
             num_iters = args[2].parse().expect("Invalid number of iterations");}

        4 => {path_to_data = args[0].parse().expect("Invalid file path");
             num_features = args[1].parse().expect("Invalid number of features");
             num_iters = args[2].parse().expect("Invalid number of iterations");
             learn_rate = args[3].parse().expect("Invalid learning rate");}

        5 => {path_to_data = args[0].parse().expect("Invalid file path");
             num_features = args[1].parse().expect("Invalid number of features");
             num_iters = args[2].parse().expect("Invalid number of iterations");
             learn_rate = args[3].parse().expect("Invalid learning rate");
             batch_size = args[4].parse().expect("Invalid batch_size");}

        _ => panic!("Wrong number of arguments!"),
    }

    //read from csv source
    let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');

    //create the environment
    let mut env = StreamEnvironment::new(config);
    env.spawn_remote_workers();

    //return the weights computed with SGD (the model.fit method)
    let res = env.stream(source)
        .replay(
            num_iters,
            State::new(num_features),

            //BODY
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
                            //the target is in the last element of each sample
                            let y: f64 = x[x.len()-1]; 
                            //switch the target with a 1 for the intercept
                            x[x.len()-1] = 1.;
                                let current_weights = &state.get().weights;
                                let prediction: f64 = x.iter().zip(current_weights.iter()).map(|(xi, bi)| xi * bi).sum();
                                let error = prediction - y;
                                let sample_grad: Vec<f64> = x.iter().map(|xi| xi * error).collect();                            
                                sample_grad
                            }
                        else {None}}})
            },

            //LOCAL FOLD
            //the sample gradients vectors of each replica are pushed in a local_grad vector of vectors of the replica
            move |local_grad: &mut Vec<Vec<f64>>, sample_grad| 
            {
                local_grad.push(sample_grad);
            },

            //GLOBAL FOLD
            //the global gradient is computed as the average of the gradients of each replica
            move |state, local_grad| 
            {
                let num_replica = 10; //?
                //the local gradient of each replica is computed as the sum of the gradients of the samples in the replica
                let mut local_grad_sum: Vec<f64> = vec![0.;num_features+1];
                //the sum is performed over the columns (corresponding to a feature)
                for row in local_grad.iter(){
                    local_grad_sum = local_grad_sum.iter().zip(row.iter()).map(|(a, b)| (a + b)).collect();
                }
                //the average is computed as sum/(batch size * n_replica)
                state.global_grad = state.global_grad.iter().zip(local_grad_sum.iter()).map(|(a, b)| (a + b)/(batch_size * num_replica) as f64).collect();
                //print!("local Grad{:?}\n", local_grad);
            },

            //LOOP CONDITION
            move|state| 
            {   
                //print!("Grad{:?}\n", state.global_grad);
                //update iterations
                state.epoch +=1;
                //update the weights
                state.weights = state.weights.iter().zip(state.global_grad.iter()).map(|(beta, g)| beta - g * learn_rate).collect();
                let tol: f64 = state.global_grad.iter().map(|v| v*v).sum();
                //reset the global gradient for the next iteration
                state.global_grad = vec![0.;num_features+1];
                //loop condition
                //print!("TOL{}\n", tol);
                state.epoch < num_iters && tol.sqrt() * learn_rate > 1e-4
            },
        )
        .collect_vec();


    let start = Instant::now();
    env.execute();
    let elapsed = start.elapsed();

    if let Some(res) = res.get() {
        let state = &res[0];
        eprintln!("Weights: {:?}", state.weights);
        eprintln!("Epochs: {:?}",state.epoch);
    }
    eprintln!("Elapsed: {elapsed:?}");
}