use noir::prelude::*;

use serde::{Deserialize, Serialize};

use std::time::Instant;

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


fn main() {
    let (config, args) = EnvironmentConfig::from_args();

    //args: path_to_data, n_features, n_iters, learn_rate, batch_size,
    let path_to_data: String;
    let num_features: usize;
    let mut num_iters = 100;
    let mut learn_rate= 1e-3;
    let mut batch_size= 16;
    let mut l2_reg = false;
    let mut norm = false;
    

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

        6 => {path_to_data = args[0].parse().expect("Invalid file path");
            num_features = args[1].parse().expect("Invalid number of features");
            num_iters = args[2].parse().expect("Invalid number of iterations");
            learn_rate = args[3].parse().expect("Invalid learning rate");
            batch_size = args[4].parse().expect("Invalid batch_size");
            norm= args[5].parse().expect("Normalization must be set true or false");}
        
            7 => {path_to_data = args[0].parse().expect("Invalid file path");
            num_features = args[1].parse().expect("Invalid number of features");
            num_iters = args[2].parse().expect("Invalid number of iterations");
            learn_rate = args[3].parse().expect("Invalid learning rate");
            batch_size = args[4].parse().expect("Invalid batch_size");
            norm= args[5].parse().expect("Normalization must be set true or false");
            l2_reg= args[6].parse().expect("L2 regularization must be set true or false");}

        _ => panic!("Wrong number of arguments!"),
    }


    //read from csv source
    let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');


    //NORMALIZATION
    let mut mean: Vec<f64> = vec![0.;num_features+1];
    let mut std = vec![0.;num_features+1];
    
    if norm==true{

    let mut env0 = StreamEnvironment::new(config.clone());
    env0.spawn_remote_workers();
    
    //get the mean of all the features + target and the second moment E[x^2]
    let features_mean = env0.stream(source.clone())
    .map(move |mut x| 
        {
            x.0.extend(x.0.iter().map(|xi| xi.powi(2)).collect::<Vec<f64>>());
            x
        })
    .group_by_avg(|_x| true, |x| x.clone()).drop_key().collect_vec();

    env0.execute();
    
    let mut moments:Vec<f64> = vec![0.;2*num_features+2];
    if let Some(means_vector) = features_mean.get() {
        moments = means_vector[0].0.clone();}
    
    mean= moments.iter().take(num_features+1).cloned().collect::<Vec<f64>>();
    
    std = moments.iter().skip(num_features+1).cloned().collect::<Vec<f64>>();

    std = std.iter().zip(mean.iter()).map(|(e2,avg)| (e2-avg.powi(2)).sqrt()).collect();
    
    }
    let std_ = std.clone();
    let mean_ = mean.clone();


    //TRAINING
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
                            if norm==true{
                                //scale the features and the target
                                x.0 = x.0.iter().zip(mean_.iter().zip(std_.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                }
                            //the target is in the last element of each sample
                            let y: f64 = x.0[num_features]; 
                            //switch the target with a 1 for the intercept
                            x.0[num_features] = 1.;
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
                if local_grad.0.len()==num_features+1{
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
                state.global_grad = vec![0.;num_features+1];
                //loop condition
                state.epoch < num_iters && tol.sqrt() > 1e-4
            },

        )
        .collect_vec();
    
    let start = Instant::now();

    //get the mean of the targets
    //CAN BE REPLACED BY THE STREAM BEFORE WHICH COMPUTE ALL THE MOMENTS, 
    //but if they are not all necessary, this can compute just the target mean
    let mut avg_y = 0.;

    if norm == false
    {
    let res2 = env.stream(source.clone())
    .group_by_avg(|_x| true, move|x| x.0[num_features]).drop_key().collect_vec();

     env.execute();

    if let Some(res2) = res2.get() {
    avg_y = res2[0];}
    }

    //if normalization is true, avg_y = 0, we don't need to compute the mean again
    else {
        env.execute();
    }
    
    

    

    if let Some(res) = fit.get() {
        let state = &res[0];
        let weights = state.weights.clone();


   


    //SCORE
    //compute the score on the training set
    let mut env2 = StreamEnvironment::new(config);
    env2.spawn_remote_workers();

    //compute the residuals sums for the R2
    let score = env2.stream(source)

        .map(move |mut x| {
            let mut mean_y = avg_y;
            //scale the features and the target   
            if norm==true{
                x.0 = x.0.iter().zip(mean.iter().zip(std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                mean_y = 0.; 
                }                     
            let y = x.0[num_features];
            x.0[num_features] = 1.;   
            let pred: f64 = x.0.iter().zip(weights.iter()).map(|(xi,wi)| xi*wi).sum();
            [(y-pred).powi(2),(y-mean_y).powi(2)]           
        })

        .fold_assoc([0.,0.],
            |acc,value| {acc[0]+=value[0];acc[1]+=value[1];}, 
            |acc,value| {acc[0]+=value[0];acc[1]+=value[1];})
        .collect_vec();

    
    env2.execute();


    let elapsed = start.elapsed();
        
    let mut r2 = -999.;
    if let Some(res3) = score.get() {
        r2 = 1.-(res3[0][0]/res3[0][1]);}
    
    eprintln!("Weights: {:?}", state.weights);
    eprintln!("Epochs: {:?}",state.epoch);
    eprintln!("R2: {:?}",r2);
    eprintln!("Elapsed: {elapsed:?}");
}

}