#![allow(dead_code)]

use noir::prelude::*;
use std::{time::Instant, collections::HashMap};
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

fn train_decision_tree(data: &[Vec<f64>], targets: &[f64], min_samples_split: usize, max_features: usize, max_depth: usize, split_point: String, dynamic_flag: usize, n_split: usize) -> DecisionTree {
    if targets.is_empty(){
        DecisionTree { root: Some(Node::Void {}) } 
    }
    else{

        let depth = 0;

        let root = build_tree_regression(data, targets, depth, max_depth, min_samples_split, max_features, &split_point.to_lowercase(), dynamic_flag, n_split);
        DecisionTree { root: Some(root) }
    }
}

fn build_tree_regression(data: &[Vec<f64>], targets: &[f64], depth: usize, max_depth: usize, min_samples_split: usize, max_features: usize, split_point: &String, dynamic_flag: usize, n_split: usize) -> Node {
    if targets.is_empty() {
        Node::Void {}
    } else {
        let mean_target = targets.iter().sum::<f64>() / targets.len() as f64;

        if targets.len()<=2 {
            Node::Leaf {
                prediction: mean_target,
            }
        } else if depth == max_depth || targets.len() <= min_samples_split{
            //Create a leaf node if there are no remaining features to split on
            Node::Leaf {
                prediction: mean_target,
            }
        } else {

            let feature_count = data[0].len();
            let mut rng = rand::thread_rng();
            let range = (0..=feature_count-1).collect::<Vec<_>>(); 

            let feature_indices: Vec<usize> = range
                .choose_multiple(&mut rng, max_features)
                .cloned()
                .collect();

            let (best_feature_index, best_split_value);

            if dynamic_flag == 0{
                (best_feature_index, best_split_value) =
                match split_point.as_str()
                {

                    "complete" | "expensive" | "all"  => find_best_split_expensive(data, targets, &feature_indices),

                    "mean"  => split_mean(data, targets, &feature_indices),

                    "median" | _ => split_median(data, targets, &feature_indices),
                };
            }
            else{
                (best_feature_index, best_split_value) =
                match split_point.as_str()
                {
                    "uniform" => find_best_split_uniform(data, targets, &feature_indices, n_split),
                    "k-tile" | _ => find_best_split_ktile(data, targets, &feature_indices, n_split),
                };    
            }

                let mut left_data: Vec<Vec<f64>> = Vec::new();
                let mut left_targets: Vec<f64> = Vec::new();
                let mut right_data: Vec<Vec<f64>> = Vec::new();
                let mut right_targets: Vec<f64> = Vec::new();

                for i in 0..data.len() {
                    let feature_value = data[i][best_feature_index];
                    if feature_value <= best_split_value {
                        left_data.push(data[i].clone());
                        left_targets.push(targets[i]);
                    } else {
                        right_data.push(data[i].clone());
                        right_targets.push(targets[i]);
                    }
                }

            // Recursively build the left and right subtrees
            let left = Box::new(build_tree_regression(&left_data, &left_targets, depth+1, max_depth, min_samples_split, max_features, &split_point.clone(), dynamic_flag, n_split));
            let right = Box::new(build_tree_regression(&right_data, &right_targets, depth+1, max_depth, min_samples_split, max_features, &split_point, dynamic_flag, n_split));


                Node::Split {
                    feature_index: best_feature_index,
                    split_value: best_split_value,
                    left,
                    right,
                }
        }
    }
}



fn find_best_split_uniform(
    data: &[Vec<f64>],
    targets: &[f64],
    feature_indices: &[usize],
    n_split: usize
) -> (usize, f64) {
    let mut best_feature_index = 0;
    let mut best_split_value = 0.0;
    let mut best_mse = f64::MAX;

    for &feature_index in feature_indices {
        let feature_values: Vec<f64> = data.iter().map(|sample| sample[feature_index]).collect();
        let max = feature_values.iter().fold(f64::NEG_INFINITY, |max, &x| max.max(x));
        let min = feature_values.iter().copied().fold(f64::INFINITY, f64::min);

        for i in 1..n_split+1 {
            let split_value = min + i as f64 * (max-min)/n_split as f64;

            let (left_counts, right_counts) = split_targets_regression(targets, &data, feature_index, split_value);

            let mse = calculate_mean_squared_error(&left_counts, &right_counts);
            if mse < best_mse {
                best_mse = mse;
                best_feature_index = feature_index;
                best_split_value = split_value;
            }
        }
    }

    (best_feature_index, best_split_value)
}


fn find_best_split_ktile(
    data: &[Vec<f64>],
    targets: &[f64],
    feature_indices: &[usize],
    n_split: usize
) -> (usize, f64) {
    let mut best_feature_index = 0;
    let mut best_split_value = 0.0;
    let mut best_mse = f64::MAX;

    for &feature_index in feature_indices {
        let mut feature_values: Vec<f64> = data.iter().map(|sample| sample[feature_index]).collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let tile_size = feature_values.len() / (n_split+1);

        for i in 1..n_split+1 {
            let split_value = feature_values[tile_size*i];

            let (left_counts, right_counts) = split_targets_regression(targets, &data, feature_index, split_value);

            let mse = calculate_mean_squared_error(&left_counts, &right_counts);
            if mse < best_mse {
                best_mse = mse;
                best_feature_index = feature_index;
                best_split_value = split_value;
            }
        }
    }

    (best_feature_index, best_split_value)
}

fn find_best_split_expensive(
    data: &[Vec<f64>],
    targets: &[f64],
    feature_indices: &[usize],
) -> (usize, f64) {
    let mut best_feature_index = 0;
    let mut best_split_value = 0.0;
    let mut best_mse = f64::MAX;

    for &feature_index in feature_indices {
        let mut feature_values: Vec<f64> = data.iter().map(|sample| sample[feature_index]).collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for i in 0..feature_values.len() - 1 {
            let split_value = (feature_values[i] + feature_values[i + 1]) / 2.0;

            let (left_targets, right_targets) = split_targets_regression(targets, data, feature_index, split_value);

            let mse = calculate_mean_squared_error(&left_targets, &right_targets);
            if mse < best_mse {
                best_mse = mse;
                best_feature_index = feature_index;
                best_split_value = split_value;
            }
        }
    }

    (best_feature_index, best_split_value)
}



fn split_median(data: &[Vec<f64>],targets: &[f64],feature_indices: &[usize],) 
    -> (usize, f64) 
{
    let mut best_feature_index = 0;
    let mut best_split_value = 0.0;
    let mut best_mse = f64::MAX;

    for &feature_index in feature_indices {
        let mut feature_values: Vec<f64> = data.iter().map(|sample| sample[feature_index]).collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_index = feature_values.len() / 2;
        let split_value = feature_values[median_index];

        let (left_targets, right_targets) = split_targets_regression(targets, data, feature_index, split_value);

        let mse = calculate_mean_squared_error(&left_targets, &right_targets);
        if mse < best_mse {
            best_mse = mse;
            best_feature_index = feature_index;
            best_split_value = split_value;
        }
    }

    (best_feature_index, best_split_value)
}


fn split_mean(data: &[Vec<f64>],targets: &[f64],feature_indices: &[usize],) 
    -> (usize, f64) 
{
    let mut best_feature_index = 0;
    let mut best_split_value = 0.0;
    let mut best_mse = f64::MAX;

    for &feature_index in feature_indices {
        let feature_values: Vec<f64> = data.iter().map(|sample| sample[feature_index]).collect();
        let mean: f64 = feature_values.iter().sum::<f64>()/feature_values.len() as f64;

        let (left_targets, right_targets) = split_targets_regression(targets, data, feature_index, mean);

        let mse = calculate_mean_squared_error(&left_targets, &right_targets);
        if mse < best_mse {
            best_mse = mse;
            best_feature_index = feature_index;
            best_split_value = mean;
        }
    }

    (best_feature_index, best_split_value)
}



fn split_targets_regression(
    targets: &[f64],
    data: &[Vec<f64>],
    feature_index: usize,
    split_value: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut left_targets: Vec<f64> = Vec::new();
    let mut right_targets: Vec<f64> = Vec::new();

    for i in 0..data.len() {
        let feature_value = data[i][feature_index];
        let target = targets[i];

        if feature_value <= split_value {
            left_targets.push(target);
        } else {
            right_targets.push(target);
        }
    }

    (left_targets, right_targets)
}

fn calculate_mean_squared_error(left_targets: &[f64], right_targets: &[f64]) -> f64 {
    let left_mean = left_targets.iter().sum::<f64>() / left_targets.len() as f64;
    let right_mean = right_targets.iter().sum::<f64>() / right_targets.len() as f64;

    let left_mse = left_targets.iter().map(|&y| (y - left_mean).powi(2)).sum::<f64>() / left_targets.len() as f64;
    let right_mse = right_targets.iter().map(|&y| (y - right_mean).powi(2)).sum::<f64>() / right_targets.len() as f64;

    (left_mse + right_mse) / 2.0
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

#[derive(Clone, Serialize, Deserialize, Default)]
struct StateRF {
    //regression coefficients
    forest: Vec<DecisionTree>,
    iter: usize,
}

impl StateRF {
    fn new() -> StateRF {
        StateRF {
            forest:  Vec::<DecisionTree>::new(),
            iter : 0,
        }}}


#[derive(Clone, Serialize, Debug, Deserialize, Default)]
struct RandomForestRegressor {
    forest: Vec<DecisionTree>,
    fitted: bool,
    num_tree: usize,
    max_features: usize,
    max_depth: usize,
    min_samples_split: usize,
}

impl RandomForestRegressor {fn new(num_tree:usize, max_features: usize, max_depth: usize, min_samples_split: usize) -> RandomForestRegressor{ 
    RandomForestRegressor{ 
                forest:  Vec::<DecisionTree>::new(), 
                fitted: false,
                num_tree,
                max_features,
                max_depth,
                min_samples_split,
                }
    }}


impl RandomForestRegressor {
    fn fit(&mut self, path_to_data: &String, data_fraction: f64, split_point: String, config: &EnvironmentConfig)
        {
        
        let num_tree: usize= self.num_tree;
        let max_features= self.max_features;
        let min_samples_split= self.min_samples_split;
        let max_depth = self.max_depth;
        self.fitted = true;
        
        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let fit = env.stream(source)
        .map(move |mut x| {
            //id of the tree which will get for sure the sample
            let tree_id = rand::thread_rng().gen_range(0..=num_tree-1);
            //target class
            let y = x.pop().unwrap();
            //the structure forwarded is a DecisionTree because we need it to be streamed in the replay
            DecisionTree{root: Some(Node::Forward { id: tree_id, feature: x, target: y})}
        })
        .shuffle()
        .replay(
            2,
            StateRF::new(),
            move |s, state| 
            {
                s.rich_filter_map({
                    //for each tree a "matrix" with data and vec with targets
                    let mut local_trees_data: HashMap<usize, (Vec<Vec<f64>>, Vec<f64>)> = HashMap::new();
                    let mut flag_result = 0;
                    move | x|{
                        if state.get().iter == 1 && flag_result<num_tree{
                         
                            let tree = train_decision_tree(&local_trees_data.get(&flag_result).unwrap().0,
                                        &local_trees_data.get(&flag_result).unwrap().1, min_samples_split, max_features, max_depth, split_point.clone(), 0, 1);
                            flag_result+=1;
                            Some(tree)
                        }
                        else if state.get().iter==0{
                            let mut features = Vec::new();
                            let mut id_tree = 0;
                            let mut y = 0.;
                            match x.root.unwrap() {
                                //get the sample information
                                Node::Forward { id, feature, target } => {
                                    features = feature;
                                    id_tree = id;
                                    y = target;
                                }
                                //no node will be one of the following
                                Node::Split { feature_index: _, split_value: _, left: _, right: _ } => {}
                                Node::Leaf { prediction: _ } => {}
                                Node::Void {} =>{}
                            };
                            
                            //add to the corresponding tree id the features and the class of the sample
                            local_trees_data.entry(id_tree).or_insert((Vec::new(),Vec::new())).0.push(features.clone());
                            local_trees_data.entry(id_tree).or_insert((Vec::new(),Vec::new())).1.push(y);
                            
                            //for each tree probability of data_fraction% to use the sample for training
                            for i in 0..num_tree{
                                if i!=id_tree && rand::thread_rng().gen::<f64>() > (1.-data_fraction){
                                    local_trees_data.entry(i).or_insert((Vec::new(),Vec::new())).0.push(features.clone());
                                    local_trees_data.entry(i).or_insert((Vec::new(),Vec::new())).1.push(y);
                                }
                            }
                            None
                }
                else{
                    None
                }
            }})
            },

             move |local_trees: &mut Vec<DecisionTree>, tree: DecisionTree| 
            {   
                local_trees.push(tree);
            },

            move |state, local_trees| 
            {   
                if state.iter ==1{
                    for i in 0..local_trees.len(){
                        state.forest.push(local_trees[i].clone());
                    }
                }
            },

            move|state| 
            {   
                state.iter +=1;       
                true
            },

        )
        .collect_vec();


    env.execute();

    let state = fit.get().unwrap()[0].clone();
    self.forest = state.forest;   
    }


    fn dynamic_fit(&mut self, path_to_data: &String, data_fraction: f64, dynamic_split_method: String, n_split: usize, config: &EnvironmentConfig)
        {
        
        let num_tree: usize= self.num_tree;
        let max_features= self.max_features;
        let min_samples_split= self.min_samples_split;
        let max_depth = self.max_depth;
        self.fitted = true;
        
        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let fit = env.stream(source)
        .map(move |mut x| {
            //id of the tree which will get for sure the sample
            let tree_id = rand::thread_rng().gen_range(0..=num_tree-1);
            //target class
            let y = x.pop().unwrap();
            //the structure forwarded is a DecisionTree because we need it to be streamed in the replay
            DecisionTree{root: Some(Node::Forward { id: tree_id, feature: x, target: y})}
        })
        .shuffle()
        .replay(
            2,
            StateRF::new(),
            move |s, state| 
            {
                s.rich_filter_map({
                    //for each tree a "matrix" with data and vec with targets
                    let mut local_trees_data: HashMap<usize, (Vec<Vec<f64>>, Vec<f64>)> = HashMap::new();
                    let mut flag_result = 0;
                    move | x|{
                        if state.get().iter == 1 && flag_result<num_tree{
                         
                            let tree = train_decision_tree(&local_trees_data.get(&flag_result).unwrap().0,
                                                &local_trees_data.get(&flag_result).unwrap().1, min_samples_split, max_features, max_depth, dynamic_split_method.clone(), 1, n_split);
                            flag_result+=1;
                            Some(tree)
                        }
                        else if state.get().iter==0{
                            let mut features = Vec::new();
                            let mut id_tree = 0;
                            let mut y = 0.;
                            match x.root.unwrap() {
                                //get the sample information
                                Node::Forward { id, feature, target } => {
                                    features = feature;
                                    id_tree = id;
                                    y = target;
                                }
                                //no node will be one of the following
                                Node::Split { feature_index: _, split_value: _, left: _, right: _ } => {}
                                Node::Leaf { prediction: _ } => {}
                                Node::Void {} =>{}
                            };
                            
                            //add to the corresponding tree id the features and the class of the sample
                            local_trees_data.entry(id_tree).or_insert((Vec::new(),Vec::new())).0.push(features.clone());
                            local_trees_data.entry(id_tree).or_insert((Vec::new(),Vec::new())).1.push(y);
                            
                            //for each tree probability of data_fraction% to use the sample for training
                            for i in 0..num_tree{
                                if i!=id_tree && rand::thread_rng().gen::<f64>() > (1.-data_fraction){
                                    local_trees_data.entry(i).or_insert((Vec::new(),Vec::new())).0.push(features.clone());
                                    local_trees_data.entry(i).or_insert((Vec::new(),Vec::new())).1.push(y);
                                }
                            }
                            None
                }
                else{
                    None
                }
            }})
            },

             move |local_trees: &mut Vec<DecisionTree>, tree: DecisionTree| 
            {   
                local_trees.push(tree);
            },

            move |state, local_trees| 
            {   
                if state.iter ==1{
                    for i in 0..local_trees.len(){
                        state.forest.push(local_trees[i].clone());
                    }
                }
            },

            move|state| 
            {   
                state.iter +=1;       
                true
            },

        )
        .collect_vec();


    env.execute();

    let state = fit.get().unwrap()[0].clone();
    self.forest = state.forest;   
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


    fn r2_score(&self, path_to_data: &String, config: &EnvironmentConfig) -> f64{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}

        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let res = env.stream(source.clone())
        .group_by_avg(|_x| true, move|x| x[x.len()-1]).drop_key().collect_vec();
        env.execute();
        let avg_y = res.get().unwrap()[0];

        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let forest = self.forest.clone();

        let score = env.stream(source)
    
            .map(move |mut x| {
                let y = x.pop().unwrap();
                let mut sum = 0.;
                let mut count = 0.;
                for tree in forest.clone(){
                    let pred = predict_sample(&x,tree.root.unwrap());
                    sum+= pred;
                    count+=1.;
                }
                let pred = sum/count;
                [(y-pred).powi(2),(y-avg_y).powi(2)]           
            })
    
            .fold_assoc([0.,0.],
                |acc,value| {acc[0]+=value[0];acc[1]+=value[1];}, 
                |acc,value| {acc[0]+=value[0];acc[1]+=value[1];})
            .collect_vec();
        
        env.execute();
            
        let mut r2 = -999.;
        if let Some(res3) = score.get() {
            r2 = 1.-(res3[0][0]/res3[0][1]);}
        
        r2
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
    let training_set = "housing_numeric.csv".to_string();
    let data_to_predict = "housing_numeric.csv".to_string();

    
    let num_tree = 100;
    let min_samples_split = 20;
    let max_features = 5;
    let max_depth = 5;

    let mut model = RandomForestRegressor::new(num_tree, max_features, max_depth, min_samples_split);

    
    let data_fraction = 0.1;
    //let split_method = "expensive".to_string(); //"median" //"mean"
    let dynamic_split_method = "uniform".to_string(); //"k-tile"
    
    let start = Instant::now();
    //return the trained model
    //model.fit(&training_set, data_fraction, split_method, &config);

    model.dynamic_fit(&training_set, data_fraction, dynamic_split_method, 100, &config);

    let elapsed = start.elapsed();

    //compute the score over the training set
    //let score = model.mse_score(&training_set, &config);
    let score = model.r2_score(&training_set, &config);

    let predictions = model.predict(&data_to_predict, &config);

    // //print!("\nCoefficients: {:?}\n", model.features_coef);
    // //print!("Intercept: {:?}\n", model.intercept);  

    print!("\nR2: {:?}\n", score);
    print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<f64>>());
    eprintln!("\nElapsed: {elapsed:?}");
    //eprintln!("{:#?}",model.forest);

}