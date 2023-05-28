use noir::prelude::*;
use noir_ml::sample::Sample;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use rand::Rng;
use rand::seq::SliceRandom;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Serialize, Debug, Deserialize, Default)]
struct DecisionTree {
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
    Forward {
        id: usize,
        feature: Vec<f64>,
        target: f64,
    },
    Void {},
}

fn train_decision_tree(feature_count: usize, min_features: usize, max_features: usize, path: &String, config: EnvironmentConfig) -> DecisionTree {

        let mut rng = rand::thread_rng();
        let range = (0..feature_count).collect::<Vec<_>>();

        let mut feature_indices: Vec<usize> = range
            .choose_multiple(&mut rng, rand::thread_rng().gen_range(min_features..=max_features))
            .cloned()
            .collect();

        let root = build_tree_regression(&mut feature_indices, config, path.clone());
        DecisionTree { root: Some(root) }
    }


#[derive(Clone, Serialize, Deserialize, Default)]
struct StateSplit {
    best_feature: usize,
    best_mse: f64,
    best_split: f64,
    current_splits: Vec<f64>,
    iter: usize
}

impl StateSplit {
    fn new(starting_split: Vec<f64>) -> StateSplit {
        StateSplit {
            best_feature: 0,
            best_mse: f64::MAX,
            best_split: 0.,
            current_splits: starting_split,
            iter: 0
        }}}


fn build_tree_regression(feature_indices: &mut Vec<usize>, config: EnvironmentConfig, path: String) -> Node {

    let source = CsvSource::<Sample>::new(path.clone()).has_headers(true).delimiter(b',');
    let mut env = StreamEnvironment::new(config.clone());
    env.spawn_remote_workers();

    //compute mean of each features and target
    let result = &env.stream(source)
    .group_by_avg(|_| true, |x| x.clone()).drop_key().collect_vec().get().unwrap()[0].0;
    let dim_features = result.len() - 1;
    let mean_target = result[dim_features];
    let mean_features = result.iter().take(dim_features).cloned().collect::<Vec::<f64>>();
    
    env.execute();

    if feature_indices.is_empty() {
        Node::Leaf {
            prediction: mean_target,
        }
    }
    else {
        //find_best_feature and best split value
        let source = CsvSource::<Vec<f64>>::new(path.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let (max_features, min_features) = &env.stream(source.clone())
        .fold_assoc((vec![f64::MIN;dim_features],vec![f64::MAX;dim_features]), 
    |(max, min), x| {
                for (i, &elem) in x.iter().enumerate(){
                    if elem>max[i]{
                        max[i] = elem;
                    }
                    if elem<min[i]{
                        min[i] = elem;
                    }
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
        .collect_vec().get().unwrap()[0];

        env.execute();
        
        let num_splits = 10;
        let upper_split_interval = mean_features.iter().zip(max_features.iter()).map(|(mean, max)| (max - mean) / num_splits as f64).collect::<Vec<f64>>();
        let lower_split_interval = mean_features.iter().zip(min_features.iter()).map(|(mean, min)| (mean - min) / num_splits as f64).collect::<Vec<f64>>();
        let starting_splits: Vec<f64> = min_features.iter().zip(lower_split_interval.iter()).map(|(a ,b)| a+b).collect();
        

        //Find best split
        let mut state = StateSplit::new(starting_splits);

        for _ in 0..2*num_splits-1{
        
        let current_splits = state.clone().current_splits;
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        //compute the mean of the features based on their split left or right
        let result  = &env.stream(source.clone()).fold_assoc(
            (vec![0.;dim_features],vec![0.;dim_features], vec![0;dim_features], vec![0;dim_features]),
            move |(left,right, count_left, count_right), x|
            for (i,&elem) in x.iter().enumerate(){
                if elem<current_splits[i]/*state.get().current_splits[i]*/{
                    left[i] += elem;
                    count_left[i]+=1;
                }
                else{
                    right[i] += elem;
                    count_right[i]+=1;
                }
            },
            |(left,right, count_left, count_right), tuple|{
                *left = left.iter().zip(tuple.0.iter().zip(count_left.iter())).map(|(a,(b,c))| a+ (b/ *c as f64)).collect::<Vec<f64>>();
                *right = right.iter().zip(tuple.1.iter().zip(count_right.iter())).map(|(a,(b,c))| a + (b/ *c as f64)).collect::<Vec<f64>>();
                //counter for the number of replica
                count_left[0]+=1;
            }
        ).collect_vec().get().unwrap()[0];

        env.execute();

        let (left_means, right_means): (Vec<f64>, Vec<f64>) = (result.0.iter().map(|a|a/result.3[0] as f64).collect(), result.1.iter().map(|a|a/result.3[0] as f64).collect());

        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let current_splits = state.clone().current_splits;

        let tuple = &env.stream(source.clone()).fold_assoc(
            (vec![0.;dim_features],vec![0.;dim_features], vec![0;dim_features], vec![0;dim_features]),
            move |(left_mse,right_mse, count_left, count_right), x|
            for (i,&elem) in x.iter().enumerate(){
                if elem<current_splits[i]{
                    left_mse[i] += (elem-left_means[i]).powi(2);
                    count_left[i]+=1;
                }
                else{
                    right_mse[i] += (elem-right_means[i]).powi(2);
                    count_right[i]+=1;
                }
            },
            |(left,right, count_left, count_right), tuple|{
                *left = left.iter().zip(tuple.0.iter().zip(count_left.iter())).map(|(a,(b,c))| a+ (b/ *c as f64)).collect::<Vec<f64>>();
                *right = right.iter().zip(tuple.1.iter().zip(count_right.iter())).map(|(a,(b,c))| a + (b/ *c as f64)).collect::<Vec<f64>>();
            }).collect_vec().get().unwrap()[0];

            env.execute();
            
            let mse_vec: Vec<f64> = tuple.0.iter().zip(tuple.1.iter()).map(|(a,b)| (a+b)/2.).collect();

            let (feat_idx, &mse) = mse_vec.iter().enumerate().min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
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

        let (best_feature_index, best_split_value) = (state.best_feature, state.best_mse);
    
        feature_indices.retain(|&x| x != best_feature_index);

        let left = Box::new(build_tree_regression(feature_indices, config.clone(), path.clone()));
        let right = Box::new(build_tree_regression(feature_indices, config.clone(), path.clone()));

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
        Node::Forward { id: _, feature: _, target: _ } => 0.,
        Node::Void {  } => 0.,
    }
}



#[derive(Clone, Serialize, Debug, Deserialize, Default)]
struct RandomForestRegressor {
    forest: Vec<DecisionTree>,
    fitted: bool,
}

impl RandomForestRegressor {fn new() -> RandomForestRegressor{ 
    RandomForestRegressor{ 
                forest:  Vec::<DecisionTree>::new(), 
                fitted: false,
                }
    }}

//train the model with sgd or adam
impl RandomForestRegressor {
    fn fit(&mut self, path_to_data: &String, num_features:usize, min_features: usize, max_features: usize, config: EnvironmentConfig)
        {

        self.fitted = true;

        let tree = train_decision_tree(num_features, min_features, max_features, path_to_data, config);                          
    }


    fn mse_score(&self, path_to_data: &String, config: &EnvironmentConfig) -> f64{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}
        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let forest = self.forest.clone();

        let predictions = env.stream(source)
    
            .map(move |mut x| {
                let y = x.pop().unwrap();
                let mut sum = 0.;
                let mut count = 0.;
                for tree in forest.clone(){
                    let pred = predict_sample(&x,tree.root.unwrap());
                    sum+= pred;
                    count+=1.;
                }
                let squared_err = (y - sum/count).powi(2);
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
        
        let forest = self.forest.clone();

        let predictions = env.stream(source)
    
            .map(move |x| {
                let mut sum = 0.;
                let mut count = 0.;
                for tree in forest.clone(){
                    let pred = predict_sample(&x,tree.root.unwrap());
                    sum+=pred;
                    count+=1.;
                }
                sum/count
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
    let training_set = "wine_color.csv".to_string();
    let data_to_predict = "wine_color.csv".to_string();

    let start = Instant::now();

    let mut model = RandomForestRegressor::new();
    
    let num_tree = 10;
    let min_features = 3;
    let max_features = 5;

    
    model.fit(&training_set, num_tree, min_features, max_features,  config.clone());

    //compute the score over the training set
    let score = model.mse_score(&training_set, &config);

    let predictions = model.predict(&data_to_predict, &config);

    let elapsed = start.elapsed();

    // //print!("\nCoefficients: {:?}\n", model.features_coef);
    // //print!("Intercept: {:?}\n", model.intercept);  

    print!("\nMSE: {:?}\n", score);
    print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<f64>>());
    eprintln!("\nElapsed: {elapsed:?}");
    //eprintln!("{:#?}",model.forest);

}