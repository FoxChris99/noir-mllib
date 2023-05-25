#![allow(unused)]
use noir::prelude::*;
use std::time::Instant;
use noir_ml::sample::Sample;
use std::fs::File;
use std::num::ParseFloatError;
use std::collections::BTreeMap;
use std::collections::HashMap;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use itertools::Itertools;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Point{
    pub value: String,
    pub coords: Vec<f64>
}

#[derive(Default, Clone, Debug)]
struct KNNModel{            
    pub data: Vec<Point>,
    pub regression: bool   
}                           

fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> OrderedFloat<f64>{
    OrderedFloat(
        a.iter().zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>()
        .sqrt()
    )   
}


fn k_nearest(dataset: &Vec<Point>, point: &Point, k: usize) -> Vec<Point>{
    //Initialize map of k-nearest points ordered by key
    let mut map = BTreeMap::new();
     
        //Compare current point with each point in dataset    
        for p in dataset{        
                let dist = euclidean_distance(&p.coords, &point.coords);
                //For first k iterations, simply add to map
                if map.len()<k {
                    let arr: Vec<Point> = vec![p.clone()];
                    map.insert(dist, arr);
                }
                //For remaining iterations, check if it's a k-nearest before adding to map
                else{
                    if let Some(max)=&map.clone().last_entry(){
                        let val = *max.key();
                        //Check if latest point is closer than current farthest point in k-nearest
                        if dist<val{
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
        }
    
    map.values().cloned().into_iter().flatten().collect()
}


fn knn_predict(model: KNNModel, path: String, config: &EnvironmentConfig, k: usize) -> StreamOutput<Vec<f64>>{
    let mut env = StreamEnvironment::new(config.clone());
    let source = CsvSource::<Point>::new(path).has_headers(false).delimiter(b',');
    let predict_set = env.stream(source)
    .rich_map(
        move |x| {
            let nearest = k_nearest(&model.data, &x, k);
                if !model.regression {
                    let mut map: HashMap<String, usize> = HashMap::new();
                    let mut res = 0.;
                    for y in nearest {
                            *map.entry(y.value).or_default() += 1;
                    }
                    if let Some(max) = map.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k){
                        if x.value == max {
                            res = 1.;
                        }
                    }
                    res 
                }
                else {
                    let mut tot: f64=0.;
                    let mut temp: f64;
                    for y in &nearest {
                            temp = y.value.parse().unwrap();
                            tot+=temp;
                    }
                    tot/=(nearest.len() as f64);
                    temp = x.value.parse().unwrap();
                    (tot-temp).powi(2) 
                }
                        
        }
    )
    .group_by_avg(|&_k| true, |&n| n as f64)
    .drop_key()
    .collect_vec();

    let start = Instant::now();
    env.execute();
    let elapsed = start.elapsed();
    eprintln!("Test time: {elapsed:?}"); 

    return predict_set;
}

impl KNNModel {
    fn new(mode: bool) -> KNNModel {
        KNNModel { regression: mode, ..Default::default()
    }}

    fn train(&mut self, path: String, config: &EnvironmentConfig){
        //Initialize enviroment
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let mut source = CsvSource::<Point>::new(path).has_headers(false).delimiter(b',');

        //Parallel read training dataset
        let train_set = env.stream(source)
        .collect_vec();


        let start = Instant::now();
        env.execute();
        let elapsed = start.elapsed();
        eprintln!("Training time: {elapsed:?}"); 

        //Update model data with the newly read points
        if let Some(data) = train_set.get() {
            self.data = data;
        }    
    }
}


fn main() {
    let (config, args) = EnvironmentConfig::from_args();
    let train_path: String;
    let predict_path: String;
    let k: usize;

    train_path = args[0].parse().expect("Invalid train file path");
    predict_path = args[1].parse().expect("Invalid test file path");
    k = args[2].parse().expect("Invalid K value");

    let mut model = KNNModel::new(true);

    model.train(train_path, &config);
 
    //print results
    let predict_set = knn_predict(model, predict_path, &config, k);
    if let Some(result) = predict_set.get(){
        eprintln!{"Score: {:?}",result[0]}
    }
}
