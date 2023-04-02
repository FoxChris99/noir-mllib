//use std::cmp::Ordering;
//use std::hash::{Hash, Hasher};
//use std::ops::{AddAssign, Div};
//use std::time::Instant;
use std::fs::File;
use rand::distributions::{Distribution, Uniform};
use ndarray::{Array, Array2, Array1};
use std::error::Error;
use serde::{Deserialize, Serialize};

use noir::prelude::*;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/*fn data_from_csv(filename: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {

    let file = File::open(filename)?;
    let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut rows_feature: Vec<Vec<f64>> = vec![];
    let mut target: Vec<f64> = vec![];

    for result in reader.records() {

        let record = result?;
        let mut row: Vec<f64> = record.iter()
            .map(|value| value.parse::<f64>().unwrap())
            .collect();

        if let Some(t)=row.pop(){ 
            target.push(t);
        }
        else {
            println!("Missing target values found");
            target.push(999999.);
        }

        rows_feature.push(row);
    }

    let num_rows = rows_feature.len();
    let num_cols = rows_feature[0].len();
    let flat: Vec<f64> = rows_feature.into_iter().flatten().collect();
    let features = Array::from_shape_vec((num_rows, num_cols), flat)?;
    let target = Array::from_shape_vec(num_rows, target)?;

    Ok((features,target))
}*/

/*fn select_nearest(point: Point, old_centroids: &[Point]) -> Point {
    *old_centroids
        .iter()
        .map(|c| (c, point.distance_to(c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
}*/

#[derive(Clone, Serialize, Deserialize, Default)]
struct State {
    iter_count: i64,
    weights: Vec<f64>,
}

impl State {
    fn new(weights: Vec<f64>) -> State {
        State {
            weights,
            ..Default::default()
        }
    }
}

fn main() {
    let (config, args) = EnvironmentConfig::from_args();

    //args: n_features, n_iters, path
    if args.len() != 5 {
        panic!("Wrong arguments!");
    }
    let num_features: usize = args[0].parse().expect("Invalid number of features");
    let num_iters: usize = args[1].parse().expect("Invalid number of iterations");
    let batch_size: usize = args[2].parse().expect("Invalid batch size");
    let learn_rate: usize = args[3].parse().expect("Invalid learning rate");

    //get dataset path
    let path = &args[4];

    //create the environment
    let mut env = StreamEnvironment::new(config);
    env.spawn_remote_workers();

    //get data features + target
    //let (features, target) = data_from_csv(path).unwrap();
    //println!("Features: {:?}", features);
    //println!("Target: {:?}", target);
    
    //weights vector initialization
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(-0.1, 0.1);
    let init = (0..num_features+1).map(|_| uniform.sample(&mut rng)).collect();
    let initial_state = State::new(init);

    
    let source = CsvSource::<Vec<f64>>::new(path).has_headers(true).delimiter(b',');
    let res = env
        .stream(source)
        .replay(
            num_iters,
            initial_state,
            |s, state| {
               s.shuffle().rich_filter_map({
                let mut count = 0;
                move |x|{
                    count+=1;
                    if count<=batch_size { Some(x) }
                    else{ None }
                }})
               .rich_map({
                    move |x|{
                        let error;
                        if let Some(y)=x.pop(){ 
                            x.push(1.);
                            let current = &state.get().weights;
                            let prediction = x.iter().zip(current.iter()).map(|(a, b)| a * b).sum();
                            error = y-prediction;
                        }
                        error * x
                    }
                })
            },
            |update: &mut Vec<f64>, p| {
                update.iter().zip(p.iter()).map(|(a, b)| (a + b)/batch_size as f64).collect();
            },
            move |state, mut update| {
                
            }


        )
        .collect_vec();







    env.execute();
    print!("{:?}",res.get().unwrap());
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
