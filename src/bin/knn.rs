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

fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> OrderedFloat<f64>{
    OrderedFloat(
        a.iter().zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2)).sum::<f64>()
        .sqrt()
    )   
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
struct Point {
    pub class: String,
    pub coords: Vec<f64>
}

#[derive(Default, Clone, Debug)]
struct KNeighborsClassifier {
    pub data: Vec<Point> 
}

impl KNeighborsClassifier {
    fn new() -> KNeighborsClassifier {KNeighborsClassifier {
        ..Default::default()
    }}}

impl KNeighborsClassifier {
    fn set(&mut self, new_data: Vec<Point>){
        self.data = new_data;
    }
}

//Return K nearest points, as a Vec with their respective class
impl KNeighborsClassifier {
    fn k_nearest(&self, point: &Vec<f64>, k: usize) -> Vec<String>{
        let mut map = BTreeMap::new();
        
        for p in &self.data{
            let dist = euclidean_distance(point, &p.coords);
            if map.len()<k {
                map.insert(dist, p.class.to_string());
            }
            else{
                if let Some(max)=&map.clone().last_entry(){
                    let val = *max.key();
                    if dist<val{
                        map.remove(&val);
                        map.insert(dist, p.class.to_string());
                    }
                }    
            }
        }
        map.values().cloned().collect()
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
 
    //********* TRAIN **********
    //initialize enviroment
    let mut env = StreamEnvironment::new(config.clone());
    env.spawn_remote_workers();
    let mut source = CsvSource::<Point>::new(train_path).has_headers(false).delimiter(b',');

    //parallel read CSV train file
    let train_set = env.stream(source)
    .collect_vec();
    env.execute();

    //use input points to create model
    let mut model = KNeighborsClassifier::new();
    if let Some(data) = train_set.get() {
        model.set(data);
    }
    

    //******** PREDICT *********
    //initialize new enviroment
    let mut env = StreamEnvironment::new(config);
    source = CsvSource::<Point>::new(predict_path).has_headers(false).delimiter(b',');

    //parallel read CSV predict file, for each point predict its class
    let predict_set = env.stream(source)
    .rich_map(
        move |x| {
            let nearest = model.k_nearest(&x.coords, k);
            let mut map: HashMap<String, usize> = HashMap::new();
            for x in nearest {
                *map.entry(x).or_default() += 1;
            }
            let max = map.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k);
            (x.class,max)
        }
    )
    .collect_vec();

    let start = Instant::now();
    env.execute();
    let elapsed = start.elapsed();

    //print results
    if let Some(result) = predict_set.get(){
        eprintln!{"{:#?}",result}
    }
    eprintln!("Elapsed: {elapsed:?}"); 

}
