use noir::prelude::*;

use serde::{Deserialize, Serialize};

use std::time::Instant;
use std::ops::{AddAssign,Div};


#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Serialize, Deserialize, Default)]
struct Sample(Vec<f64>);

// Implement AddAssign for MyVec
impl AddAssign for Sample {
    fn add_assign(&mut self, other: Self) {
        // Make sure the two vectors have the same length
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");

        // Iterate over each element and add them together
        for (i, element) in other.0.into_iter().enumerate() {
            self.0[i] += element;
        }
    }
}

impl Div for Sample {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        // Make sure the two vectors have the same length
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");

        // Create a new MyVec instance to store the result
        let mut result = Sample(vec![0.0; self.0.len()]);

        // Iterate over each element and divide them
        for (i, element) in self.0.into_iter().enumerate() {
            result.0[i] = element / other.0[i];
        }

        result
    }
}

impl Div<f64> for Sample {
    type Output = Self;

    fn div(self, other: f64) -> Self::Output {
        
        let mut result = Sample(vec![0.0; self.0.len()]);

        // Iterate over each element and divide them
        for (i, element) in self.0.into_iter().enumerate() {
            result.0[i] = element / other;
        }

        result
    }
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct State {
    //regression coefficients
    weights: Vec<f64>,
    //total gradient of the batch
    global_grad: Vec<f64>,
    m: Vec<f64>,
    v: Vec<f64>, 
    //iterations over the dataset
    epoch: usize,
}

impl State {
    fn new(n_features: usize) -> State {
        State {
            weights:  vec![0.;n_features+1],
            global_grad: vec![0.;n_features+1],
            m: vec![0.;n_features+1],
            v: vec![0.;n_features+1],
            epoch : 0,
        }}}


fn main() {
    let (config, args) = EnvironmentConfig::from_args();

    //args: path_to_data, n_features, n_iters, learn_rate, batch_size,
    let path_to_data: String;
    let num_features: usize;
    let mut num_iters = 100;
    let mut learn_rate= 1e-3;
    let mut batch_size= 16;
    

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


    let beta1 = 0.9;
    let beta2 = 0.999;

    //read from csv source
    let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');

    //create the environment
    let mut env = StreamEnvironment::new(config.clone());
    env.spawn_remote_workers();
    

    //return the weights computed with SGD thanks to the model.fit method
    let fit = env.stream(source.clone())
        .replay(
            num_iters,
            State::new(num_features),

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
                            let y: f64 = x.0[num_features]; 
                            //switch the target with a 1 for the intercept
                            x.0[num_features] = 1.;
                                let current_weights = &state.get().weights;
                                let prediction: f64 = x.0.iter().zip(current_weights.iter()).map(|(xi, bi)| xi * bi).sum();
                                let error = prediction - y;
                                let sample_grad: Vec<f64> = x.0.iter().map(|xi| xi * error).collect();                            
                                Some(Sample(sample_grad))
                            }
                        else {None}}})
                //the average of the gradients is computed and forwarded as a single value -> global gradient
                .group_by_avg(|_x| true, |x| x.clone()).drop_key()
            },

            move |local_grad: &mut Sample, sample_grad| 
            {   
                *local_grad = sample_grad;
            },

            move |state, local_grad| 
            {   
                //we don't want to read empty replica gradient
                //BAD COMMUNICATION!!!
                if local_grad.0.len()==num_features+1{
                state.global_grad = local_grad.0.clone();}
            },

            move|state| 
            {   
                //update iterations
                state.epoch +=1;
                //update the weights
                state.m = state.m.iter().zip(state.global_grad.iter()).map(|(mi, gi)|beta1 * mi + ((1. - beta1) * gi)).collect();
                state.v =  state.v.iter().zip(state.global_grad.iter()).map(|(vi, gi)|beta2 * vi + ((1. - beta2) * gi.powi(2))).collect();
                let m_hat: Vec<f64> = state.m.iter().map(|mi|mi/(1. - beta1.powi(state.epoch as i32))).collect();
                let v_hat: Vec<f64> = state.v.iter().map(|vi|vi/(1. - beta2.powi(state.epoch as i32))).collect();
                let adam: Vec<f64> =  m_hat.iter().zip(v_hat.iter()).map(|(mi,vi)| mi/(vi.sqrt()+1e-6)).collect();
                state.weights = state.weights.iter().zip(adam.iter()).map(|(wi,a)| wi - learn_rate*a).collect();
                
                let tol: f64 = adam.iter().map(|v| v*v).sum();
                //reset the global gradient for the next iteration
                state.global_grad = vec![0.;num_features+1];
                //loop condition
                state.epoch < num_iters && tol.sqrt() > 1e-4
            },

        )
        .collect_vec();

    let res2 = env.stream(source.clone())
    .group_by_avg(|_x| true, move|x| x.0[num_features]).drop_key().collect_vec();
    
    
    let start = Instant::now();

    env.execute();



    if let Some(res) = fit.get() {
        let state = &res[0];
        let weights = state.weights.clone();


    let mut avg_y = 0.;
    if let Some(res2) = res2.get() {
        avg_y = res2[0];}





                

    let mut env2 = StreamEnvironment::new(config);
    env2.spawn_remote_workers();


    let res3 = env2.stream(source)

        .map(move |mut x| {
            let y = x.0[num_features];
            x.0[num_features] = 1.;   
            let pred: f64 = x.0.iter().zip(weights.iter()).map(|(xi,wi)| xi*wi).sum();
            [(y-pred).powi(2),(y-avg_y).powi(2)]           
        })

        .fold_assoc([0.,0.],
            |acc,value| {acc[0]+=value[0];acc[1]+=value[1];}, 
            |acc,value| {acc[0]+=value[0];acc[1]+=value[1];})
        .collect_vec();

    
    env2.execute();


    let elapsed = start.elapsed();
        
    let mut r2 = -999.;
    if let Some(res3) = res3.get() {
        r2 = 1.-(res3[0][0]/res3[0][1]);}
    
    eprintln!("Weights: {:?}", state.weights);
    eprintln!("Epochs: {:?}",state.epoch);
    eprintln!("Mean: {:?}",avg_y);
    eprintln!("R2: {:?}",r2);
    eprintln!("Elapsed: {elapsed:?}");
}

}