use noir::prelude::*;

use serde::{Deserialize, Serialize};


#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Serialize, Deserialize, Default)]
struct State {
    //regression coefficients
    weights: Vec<f64>,
}

impl State {
    fn new(weights: Vec<f64>) -> State {
        State {
            weights,
        }}}


fn main() {
    let (config, args) = EnvironmentConfig::from_args();

    //args: path_to_data, n_features, n_iters, learn_rate, batch_size,
    if args.len() == 5 || args.len() != 3 {
        panic!("Wrong arguments!");
    }

    let mut num_features = 1;
    let mut num_iters = 1000;
    let mut learn_rate= 1e-3;
    let mut batch_size=16;
    let mut path_to_data = String::new();

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

    //initialize weights and global gradient to vectors of zeros
    let initial_state = State::new(vec![0.;num_features+1]);

    //read from csv source
    let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');

    //create the environment
    let mut env = StreamEnvironment::new(config);
    env.spawn_remote_workers();

    //this should return the weights in the model.fit method
    let res = env.stream(source)
        .replay(
            num_iters,
            initial_state,

            |s, state| 
            {
                //shuffle the samples
                s.shuffle()
                //each replica gets a number of samples equal to batch size
                .rich_filter_map({
                    let mut count = 0;
                    move |x|{
                        count+=1;
                        if count<=batch_size { 
                            Some(x) }
                        else{ 
                            None }
                    }})
                //for each sample in each replica the gradient of the loss is computed (a vector of length: n_features+1)
                .rich_map({
                    move |x|{
                        let sample_grad;
                        if let Some(y)=x.pop(){ 
                            x.push(1.); //add a column of ones for the intercept
                            let current_weights = &state.get().weights;
                            let prediction: f64 = x.iter().zip(current_weights.iter()).map(|(a, b)| a * b).sum();
                            let error = y-prediction;
                            sample_grad = x.iter().map(|v| v * error).collect();
                        }
                        sample_grad
                    }})
            },

            |local_grad: &mut Vec<f64>, sample_grad| 
            {
                //the local gradient of each replica is computed as the sum of the gradients of the samples in the replica
                local_grad.extend(vec![0.; sample_grad.len()]);
                *local_grad = local_grad.iter().zip(sample_grad.iter()).map(|(a, b)| (a + b)).collect();
            },

            move |state, mut local_grad| 
            {
                let num_replica = 1; //?????????????????
                let mut global_grad = vec![0.;num_features];
                global_grad = global_grad.iter().zip(local_grad.iter()).map(|(a, b)| (a + b)/(batch_size * num_replica) as f64).collect();
                state.weights = state.weights.iter().zip(global_grad.iter()).map(|(beta, g)| beta - g * learn_rate).collect();
            },

            |state| 
            {
                true
            },

        )
        .collect_vec();







    env.execute();
    /* 
    assert_eq!(features.len(), num_features);
    let initial_state = State::new(centroids);
    let source = CsvSource::<Point>::new(path).has_headers(true);
    let res = env
        .stream(source)
        .replay(
            num_iters,
            initial_state,
            |s, state| {
                s.map(move |point| (point, select_nearest(point, &state.get().centroids), 1))
                    .group_by_avg(|(_p, c, _n)| *c, |(p, _c, _n)| *p)
                    .drop_key()
            },
            |update: &mut Vec<Point>, p| update.push(p),
            move |state, mut update| {
                if state.changed {
                    state.changed = true;
                    state.old_centroids.clear();
                    state.old_centroids.append(&mut state.centroids);
                }
                state.centroids.append(&mut update);
            },
            |state| {
                state.changed = false;
                state.iter_count += 1;
                state.centroids.sort_unstable();
                state.old_centroids.sort_unstable();
                state.centroids != state.old_centroids
            },
        )
        .collect_vec();
    let start = Instant::now();
    env.execute();
    let elapsed = start.elapsed();
    if let Some(res) = res.get() {
        let state = &res[0];
        eprintln!("Iterations: {}", state.iter_count);
        eprintln!("Output: {:?}", state.centroids.len());
    }
    eprintln!("Elapsed: {elapsed:?}");
    */
}