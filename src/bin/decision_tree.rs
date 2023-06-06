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
use std::option::Option;

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
struct TreeNode {
    pub children: Vec<TreeNode>,
    pub leaf: bool,
    pub class: Option<String>,
    pub next_feature: usize,
    pub feature_value: String
}

impl TreeNode{
    fn new()->TreeNode {
        TreeNode { ..Default::default()}
    }
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
struct Entry{
    pub class: String,
    pub features: Vec<String>,
}

impl Entry{
    fn new()->Entry {
        Entry { ..Default::default()}
    }
}

impl TreeNode {
    fn train_all(&mut self, data: Vec<Entry>, feature: usize, val: Option<String>){
        self.next_feature = feature;
        if let Some(v) = val {
            self.feature_value = v;
        }
        
        if data[0].features.len()>0 {
            data.iter().into_group_map_by(|a| a.features[0].to_string()).into_iter().for_each(
                |a|
                {   let mut node = TreeNode::new();
                    let mut arr: Vec<Entry> = a.1.into_iter().cloned().collect();
                    arr.iter_mut().for_each(|b| {b.features.remove(0);});
                    node.train_all(arr, feature+1, Some(a.0));
                    self.children.push(node);
                }
            );
        }
        else{
            self.leaf = true;
            self.class = Some(data[0].class.to_string());
        }
    }
}


fn main() {
    let (config, args) = EnvironmentConfig::from_args();
    let train_path: String;
    let predict_path: String;
    let k: usize;


    train_path = args[0].parse().expect("Invalid train file path");
    //predict_path = args[1].parse().expect("Invalid test file path");
    //k = args[2].parse().expect("Invalid K value");
 
    //********* TRAIN **********
    //initialize enviroment
    let mut env = StreamEnvironment::new(config.clone());
    env.spawn_remote_workers();
    let mut source = CsvSource::<Entry>::new(train_path).has_headers(false).delimiter(b',');


    //parallel read CSV train file
    let train_set = env.stream(source)
    .collect_vec();
    env.execute();

    let mut tree: TreeNode = TreeNode::new();
    if let Some(data) = train_set.get() {
        //eprintln!("{:#?}",data);
        tree.train_all(data,0,None);
        eprintln!("{:#?}",tree);
    }
     
    /*let start = Instant::now();
    env.execute();
    let elapsed = start.elapsed();
    eprintln!("Training time: {elapsed:?}"); 


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
            let mut res = 0;
            for x in nearest {
                *map.entry(x).or_default() += 1;
            }
            if let Some(max) = map.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k){
                if x.class == max {
                    res = 1;
                }
            }
            res 
        }
    )
    .group_by_avg(|&_k| true, |&n| n as f64)
    .drop_key()
    .collect_vec();

    let start = Instant::now();
    env.execute();
    let elapsed = start.elapsed();

    //print results
    if let Some(result) = predict_set.get(){
        eprintln!{"Score: {:?}",result[0]}
    }
    eprintln!("Test time: {elapsed:?}"); 
    */ */
}
