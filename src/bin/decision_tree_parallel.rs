use noir::prelude::*;
use noir_ml::sample::Sample;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use rand::seq::SliceRandom;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Serialize, Debug, Deserialize, Default)]
struct DTree {
    root: Option<Node>,
}

#[derive(Clone, Serialize, Debug, Deserialize)]
enum Node {
    Split {
        feature_index: usize,
        split_value: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
    Leaf {
        prediction: f64,
    },
}

fn train_decision_tree(num_features: usize, max_features: usize, max_depth: usize, min_samples: usize, path: &String, config: EnvironmentConfig) -> DTree {

        let left_treshold_lower = vec![f64::MIN;num_features];
        let left_treshold_higher = vec![f64::MAX;num_features];
        let right_treshold_lower = vec![f64::MIN;num_features];
        let right_treshold_higher = vec![f64::MAX;num_features];
        let mut depth = 0;
        let root = build_tree_regression(max_features, max_depth, &mut depth, min_samples, config, path.clone());
        DTree { root: Some(root) }
    }


#[derive(Clone, Serialize, Deserialize, Default)]
struct StateSplit {
    best_feature: usize,
    best_mse: f64,
    best_split: f64,
    current_splits: Vec<f64>,
    best_left_idxs: Vec<usize>,
    best_right_idxs: Vec<usize>
}

impl StateSplit {
    fn new(starting_split: Vec<f64>) -> StateSplit {
        StateSplit {
            best_feature: 0,
            best_mse: f64::MAX,
            best_split: 0.,
            current_splits: starting_split,
            best_left_idxs: Vec::new(),
            best_right_idxs: Vec::new()
        }}}


fn build_tree_regression(max_features: usize, max_depth: usize, depth: &mut usize, min_samples: usize, config: EnvironmentConfig, path: String) -> Node {

    let source = CsvSource::<Sample>::new(path.clone()).has_headers(true).delimiter(b',');
    let mut env = StreamEnvironment::new(config.clone());
    env.spawn_remote_workers();

    //compute mean of each features and target
    let result = env.stream(source.clone())
    .group_by_avg(|_| true, |x| x.clone()).drop_key().collect_vec();
    
    let count_result = env.stream(source)
    .group_by_count(|_| true).drop_key().collect_vec();

    env.execute();
    
    let res = result.get().unwrap()[0].clone().0;
    let dim_features = res.len()-1;
    let mean_target = res[dim_features];
    let mean_features = res.iter().take(dim_features).cloned().collect::<Vec::<f64>>();
    let num_samples = count_result.get().unwrap()[0];

    let mut rng = rand::thread_rng();
    let range = (0..dim_features).collect::<Vec<_>>();

    let feature_indices: Vec<usize> = range
        .choose_multiple(&mut rng, max_features)
        .cloned()
        .collect();

    if feature_indices.is_empty() || *depth == max_depth || num_samples < min_samples{
        Node::Leaf {
            prediction: mean_target,
        }
    }
    else {
        //find_best_feature and best split value
        let source = CsvSource::<Vec<f64>>::new(path.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let feature_idx = feature_indices.clone();

        let result = env.stream(source.clone())
        .fold_assoc((vec![f64::MIN;dim_features],vec![f64::MAX;dim_features]), 
    move |(max, min), mut x| {
                x.pop();
                let mut j = 0;
                for &idx in &feature_idx{
                    if x[idx] > max[j]{
                        max[j] = x[idx];
                    }   
                    if x[idx] < min[j]{
                        min[j] = x[idx]
                    }
                    j+=1;
                }}, 
    |acc, (max,min)|{
                for (i, &elem) in max.iter().enumerate(){
                    if elem>acc.0[i]{
                        acc.0[i] = elem;
                    }
                    if elem<acc.1[i]{
                        acc.1[i] = elem;
                    }
                }
                for (i, &elem) in min.iter().enumerate(){
                    if elem>acc.0[i]{
                        acc.0[i] = elem;
                    }
                    if elem<acc.1[i]{
                        acc.1[i] = elem;
                    }
                }})
        .collect_vec();

        env.execute();
        
        let (max_vec, min_vec) = result.get().unwrap()[0].clone();

        let num_splits = 10;
        let upper_split_interval = mean_features.iter().zip(max_vec.iter()).map(|(mean, max)| (max - mean) / num_splits as f64).collect::<Vec<f64>>();
        let lower_split_interval = mean_features.iter().zip(min_vec.iter()).map(|(mean, min)| (mean - min) / num_splits as f64).collect::<Vec<f64>>();
        let starting_splits: Vec<f64> = min_vec.iter().zip(lower_split_interval.iter()).map(|(a ,b)| a+b).collect();
        //Find best split
        let mut state = StateSplit::new(starting_splits);

        for _ in 0..2*num_splits-1{
        
        let current_splits = state.clone().current_splits;
        let feature_idx = feature_indices.clone();
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        //compute the mean of the features based on their split left or right
        let result  = env.stream(source.clone())
        .fold_assoc(
            (vec![0.;dim_features],vec![0.;dim_features], vec![0;dim_features], vec![0;dim_features]),
            move |(left,right, count_left, count_right), mut x|{
            x.pop();
            let mut i = 0;
            for &idx in &feature_idx{
                if x[idx]<=current_splits[i]{
                    left[i] += x[idx];
                    count_left[i]+=1;
                }
                else{
                    right[i] += x[idx];
                    count_right[i]+=1;
                }
                i+=1;
            }
           },
            |(left,right, count_left, _), tuple|{
                *left = left.iter().zip(tuple.0.iter().zip(tuple.2.iter())).map(|(a,(b,c))| {if *c!= 0{a+ (b/ *c as f64)} else{0.}}).collect::<Vec<f64>>();
                *right = right.iter().zip(tuple.1.iter().zip(tuple.3.iter())).map(|(a,(b,c))| {if *c!= 0{a+ (b/ *c as f64)} else{0.}}).collect::<Vec<f64>>();
                //counter for the number of replica
                count_left[0]+=1;
            }
        ).collect_vec();

        env.execute();

        let res = result.get().unwrap()[0].clone();
        let left_means: Vec<f64> = res.0.iter().map(|a|a/res.2[0] as f64).collect();
        let right_means: Vec<f64> = res.1.iter().map(|a|a/res.2[0] as f64).collect();


        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let current_splits = state.clone().current_splits;
        let feature_idx = feature_indices.clone();

        let tuple = env.stream(source.clone()).fold_assoc(
            (vec![0.;dim_features],vec![0.;dim_features], vec![0;dim_features], vec![0;dim_features]),
            move |(left_mse,right_mse, count_left, count_right), mut x|{
            x.pop();
            let mut i = 0;
            for &idx in &feature_idx{
                if x[idx]<=current_splits[i]{
                    left_mse[i] += (x[idx]-left_means[i]).powi(2);
                    count_left[i]+=1;
                }
                else{
                    right_mse[i] += (x[idx]-right_means[i]).powi(2);
                    count_right[i]+=1;
                }
                i+=1;
            }
        },
            |(left,right, _, _), tuple|{
                *left = left.iter().zip(tuple.0.iter().zip(tuple.2.iter())).map(|(a,(b,c))| {if *c!= 0{a+ (b/ *c as f64)} else{0.}}).collect::<Vec<f64>>();
                *right = right.iter().zip(tuple.1.iter().zip(tuple.3.iter())).map(|(a,(b,c))| {if *c!= 0{a+ (b/ *c as f64)} else{0.}}).collect::<Vec<f64>>();
            }).collect_vec();
            
            env.execute();
            
            let tuple = tuple.get().unwrap()[0].clone();

            let mse_vec: Vec<f64> = tuple.0.iter().zip(tuple.1.iter()).map(|(a,b)| (a+b)/2.).collect();
            
            let (feat_idx, mse) = mse_vec.iter().enumerate().fold((0, std::f64::MAX), |acc, (index, &value)| {
                if value < acc.1 {
                    (index, value)
                } else {
                    acc
                }
            });

            
            if mse < state.best_mse {
                state.best_mse = mse;
                state.best_feature = feat_idx;
                state.best_split= state.current_splits[feat_idx];
            }
            
            if state.current_splits[0] < mean_features[0]{
                state.current_splits = state.current_splits.iter().zip(lower_split_interval.iter()).map(|(a ,b)| a+b).collect();
            }
            else{
                state.current_splits = state.current_splits.iter().zip(upper_split_interval.iter()).map(|(a ,b)| a+b).collect();
            }
        }
        
        let (mut best_feature_index, best_split_value) = (state.best_feature, state.best_split);

        best_feature_index = feature_indices[best_feature_index];

        //to track the samples in the stream based on the split made
        //filter left_treshold_lower < x < left_treshold_higher
        left_treshold_lower[best_feature_index] = 
        left_treshold_higher[best_feature_index] = 
        //filter right_treshold_lower < x < right_treshold_higher
        right_treshold_lower[best_feature_index] = 
        right_treshold_higher[best_feature_index] = 

        *depth+=1;
        let left = Box::new(build_tree_regression(max_features, max_depth, depth, min_samples, config.clone(), path.clone()));
        let right = Box::new(build_tree_regression(max_features, max_depth, depth, min_samples, config.clone(), path.clone()));

        Node::Split {
            feature_index: best_feature_index,
            split_value: best_split_value,
            left,
            right,
        }

        }
    }


fn predict_sample(sample: &[f64], node: Node) -> f64 {
    match node {
        Node::Split { feature_index, split_value, left, right } => {
            let sample_value = sample[feature_index];

            if sample_value <= split_value {
                predict_sample(sample, *left)
            } else {
                predict_sample(sample, *right)
            }
        }
        Node::Leaf { prediction } => prediction,
    }
}



#[derive(Clone, Serialize, Debug, Deserialize, Default)]
struct DecisionTree {
    tree: DTree,
    fitted: bool,
}

impl DecisionTree {fn new() -> DecisionTree{ 
    DecisionTree{ 
                tree: DTree { root: None },
                fitted: false,
                }
    }}

//train the model with sgd or adam
impl DecisionTree {
    fn fit(&mut self, path_to_data: &String, num_features: usize, max_features: usize, max_depth: usize, min_samples: usize, config: EnvironmentConfig)
        {

        self.fitted = true;

        self.tree = train_decision_tree(num_features, max_features, max_depth, min_samples, path_to_data, config);                          
    }


    fn mse_score(&self, path_to_data: &String, config: &EnvironmentConfig) -> f64{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}
        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let tree = self.tree.clone();

        let predictions = env.stream(source)
    
            .map(move |mut x| {
                let y = x.pop().unwrap();
                let pred = predict_sample(&x,tree.clone().root.unwrap());
                let squared_err = (y - pred).powi(2);
                squared_err
            })
            .group_by_avg(|&_k| true, |&v| v).drop_key()   
            .collect_vec();
        
        env.execute();
            
        let result = predictions.get().unwrap()[0];
        result
    }


    fn predict(&self, path_to_data: &String, config: &EnvironmentConfig) -> Vec<f64>{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}
        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let tree = self.tree.clone();

        let predictions = env.stream(source)
    
            .map(move |x| {
                let pred = predict_sample(&x,tree.clone().root.unwrap());
                pred
            })    
            .collect_vec();
        
        env.execute();
            
        let result = predictions.get().unwrap();
        result
    }

    
}
   



       





fn main() { 
    let (config, _args) = EnvironmentConfig::from_args();

    //let training_set = "wine_quality.csv".to_string();
    let training_set = "diabetes.csv".to_string();
    let data_to_predict = "diabetes.csv".to_string();

    let mut model = DecisionTree::new();
    
    let max_features = 10;
    let max_depth = 20;
    let min_samples = 100;

    let start = Instant::now();
    
    model.fit(&training_set, max_features, max_depth, min_samples,  config.clone());

    let elapsed = start.elapsed();

    //compute the score over the training set
    let start = Instant::now();
    let score = model.mse_score(&training_set, &config);
    let elapsed_score = start.elapsed();

    let start = Instant::now();
    let predictions = model.predict(&data_to_predict, &config);
    let elapse_pred = start.elapsed();

    

    print!("\nMSE: {:?}\n", score);
    print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<f64>>());
    eprintln!("\nElapsed fit: {elapsed:?}");
    eprintln!("\nElapsed score: {elapsed_score:?}");
    eprintln!("\nElapsed pred: {elapse_pred:?}");
    eprintln!("{:#?}",model.tree);

}