#![allow(unused)]
use itertools::Itertools;
use noir::prelude::*;
use noir_ml::sample::Sample;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs::File;
use std::num::ParseFloatError;
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Point {
    pub value: String,
    pub coords: Vec<f64>,
}

#[derive(Default, Clone, Debug)]
struct KNNModel {
    pub data: Vec<Point>,
    pub regression: bool,
}

fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> OrderedFloat<f64> {
    OrderedFloat(
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt(),
    )
}

fn k_nearest(dataset: &Vec<Point>, point: &Point, k: usize) -> Vec<Point> {
    //Initialize map of k-nearest points ordered by key
    let mut map = BTreeMap::new();

    //Compare current point with each point in dataset
    for p in dataset {
        let dist = euclidean_distance(&p.coords, &point.coords);
        //For first k iterations, simply add to map
        if map.len() < k {
            let arr: Vec<Point> = vec![p.clone()];
            map.insert(dist, arr);
        }
        //For remaining iterations, check if it's a k-nearest before adding to map
        else if let Some(max) = &map.clone().last_entry() {
            let val = *max.key();
            //Check if latest point is closer than current farthest point in k-nearest
            if dist < val {
                map.remove(&val);
                let mut arr: Vec<Point> = Vec::new();
                if let Some(mut curr) = map.get(&dist) {
                    arr.extend(curr.to_vec());
                }
                arr.push(p.clone());
                map.insert(dist, arr);
            }
        }
    }

    map.values().flatten().cloned().collect()
}

impl KNNModel {
    fn new(mode: bool) -> KNNModel {
        KNNModel {
            regression: mode,
            ..Default::default()
        }
    }

    //Train model with a CSV input file
    fn train(&mut self, path: String, config: &EnvironmentConfig) {
        //Initialize enviroment
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let mut source = CsvSource::<Point>::new(path)
            .has_headers(false)
            .delimiter(b',');

        //Parallel read training dataset
        let train_set = env.stream(source).collect_vec();

        let start = Instant::now();
        env.execute();
        let elapsed = start.elapsed();
        eprintln!("Training time: {elapsed:?}");

        //Update model data with the newly read points
        if let Some(data) = train_set.get() {
            self.data = data;
        }
    }

    //Perform prediction on the model for points read from a CSV input file
    fn predict(
        &self,
        path: String,
        config: &EnvironmentConfig,
        k: usize,
    ) -> StreamOutput<Vec<f64>> {
        //Initialize enviroment
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let source = CsvSource::<Point>::new(path)
            .has_headers(false)
            .delimiter(b',');
        let model = self.clone();

        //Parallel prediction operations for each input point
        let predict_set = env
            .stream(source)
            .rich_map(move |x| {
                //Compute k-nearest points
                let nearest = k_nearest(&model.data, &x, k);
                //Classification: find most common class in k-nearest and return 1 if equal to expected class, else 0
                if !model.regression {
                    let mut map: HashMap<String, usize> = HashMap::new();
                    let mut res = 0.;
                    //Count amount of nearest points for each class
                    for y in nearest {
                        *map.entry(y.value).or_default() += 1;
                    }
                    //Get class with maximum amount and compare it with expected one
                    if let Some(max) = map.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k) {
                        if x.value == max {
                            res = 1.;
                        }
                    }
                    res
                }
                //Regression: compute average of values of k-nearest, return (average - expected value)^2
                else {
                    let mut tot: f64 = 0.;
                    let mut temp: f64;
                    for y in &nearest {
                        temp = y.value.parse().unwrap();
                        tot += temp;
                    }
                    tot /= (nearest.len() as f64);
                    temp = x.value.parse().unwrap();
                    (tot - temp).powi(2)
                }
            })
            .group_by_avg(|&_k| true, |&n| n as f64)
            .drop_key()
            .collect_vec();

        let start = Instant::now();
        env.execute();
        let elapsed = start.elapsed();
        eprintln!("Test time: {elapsed:?}");

        predict_set
    }
}

fn main() {
    let (config, args) = EnvironmentConfig::from_args();

    let reg: bool;

    let train_path: String = args[0].parse().expect("Invalid train file path");
    let predict_path: String = args[1].parse().expect("Invalid test file path");
    let k = args[2].parse().expect("Invalid K value");

    if args.len() < 4 {
        reg = false;
    } else {
        reg = args[3].parse().expect("Invalid regression boolean value");
    }

    let mut model = KNNModel::new(reg);

    model.train(train_path, &config);
    let predict_set = model.predict(predict_path.clone(), &config, k);

    if let Some(result) = predict_set.get() {
        eprintln! {"Regression: {:?}, Score: {:?}",model.regression,result[0]}
    }
}
