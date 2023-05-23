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
    //a node where a feature is splitted
    Split {
        feature_index: usize,
        split_value: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
    //leaf node with the label of the majority class
    Leaf {
        class_label: usize,
    },
    //a node just to forward the information of each sample,
    //to be then used in the replay
    Forward{
        id: usize,
        feature: Vec<f64>,
        target: usize,
    },
    Void{}
}


fn train_decision_tree(data: &[Vec<f64>], targets: &[usize], mut min_features: usize, mut max_features: usize) -> DecisionTree {
    if targets.is_empty(){
        DecisionTree { root: Some(Node::Void {}) } 
    }
    else{
        let feature_count = data[0].len();
        let mut rng = rand::thread_rng();
        let range = (0..=feature_count-1).collect::<Vec<_>>(); 
        if max_features<feature_count{
            max_features = feature_count;
        }
        if min_features > max_features{
            min_features = max_features;
        }  
        let mut feature_indices: Vec<usize> = range
        .choose_multiple(&mut rng, rand::thread_rng().gen_range(min_features..max_features+1))
        .cloned()
        .collect();
        let root = build_tree(data, targets, &mut feature_indices);
        DecisionTree { root: Some(root) }}
}

fn build_tree(data: &[Vec<f64>], targets: &[usize], feature_indices: &mut Vec<usize>) -> Node {
    if targets.is_empty(){
        Node::Void {}
    }
    else{
    let class_counts = count_class_occurrences(targets);
    let majority_class = get_majority_class(&class_counts);

    if class_counts.len() == 1 {
        // Create a leaf node if all samples belong to the same class
        Node::Leaf {
            class_label: targets[0],
        }
    } else if feature_indices.is_empty() {
        // Create a leaf node if there are no remaining features to split on
        Node::Leaf {
            class_label: majority_class,
        }
    } else {
        let (best_feature_index, best_split_value) =
        split_median(data, targets, feature_indices);
        //find_best_split_expensive(data, targets, feature_indices);

        let mut left_data: Vec<Vec<f64>> = Vec::new();
        let mut left_targets: Vec<usize> = Vec::new();
        let mut right_data: Vec<Vec<f64>> = Vec::new();
        let mut right_targets: Vec<usize> = Vec::new();

        // Split the data and targets based on the best split point
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

        // Remove the used feature index
        feature_indices.retain(|&x| x != best_feature_index);

        // Recursively build the left and right subtrees
        let left = Box::new(build_tree(&left_data, &left_targets, feature_indices));
        let right = Box::new(build_tree(&right_data, &right_targets, feature_indices));

        // Create a split node
        Node::Split {
            feature_index: best_feature_index,
            split_value: best_split_value,
            left,
            right,
        }
    }}
}

fn count_class_occurrences(targets: &[usize]) -> HashMap<usize, usize> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &target in targets {
        *counts.entry(target).or_insert(0) += 1;
    }
    counts
}

fn get_majority_class(class_counts: &HashMap<usize, usize>) -> usize {
    let (majority_class, _) = class_counts
        .iter()
        .max_by_key(|(_, &count)| count)
        .unwrap();
    *majority_class
}


// fn find_best_split_expensive(
//     data: &[Vec<f64>],
//     targets: &[usize],
//     feature_indices: &[usize],
// ) -> (usize, f64) {
//     let mut best_feature_index = 0;
//     let mut best_split_value = 0.0;
//     let mut best_gini_index = f64::MAX;

//     for &feature_index in feature_indices {
//         let mut feature_values: Vec<f64> = data.iter().map(|sample| sample[feature_index]).collect();
//         feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

//         for i in 0..feature_values.len() - 1 {
//             let split_value = (feature_values[i] + feature_values[i + 1]) / 2.0;

//             let (left_counts, right_counts) = count_class_occurrences_split(targets, &data, feature_index, split_value);

//             let gini_index = calculate_gini_index(&left_counts, &right_counts);
//             if gini_index < best_gini_index {
//                 best_gini_index = gini_index;
//                 best_feature_index = feature_index;
//                 best_split_value = split_value;
//             }
//         }
//     }

//     (best_feature_index, best_split_value)
// }


fn split_median(data: &[Vec<f64>], targets: &[usize], feature_indices: &[usize]) -> (usize, f64) {
    let mut best_feature_index = 0;
    let mut best_split_value = 0.0;
    let mut best_gini_index = f64::MAX;

    for &feature_index in feature_indices {
        let mut feature_values: Vec<f64> = data.iter().map(|sample| sample[feature_index]).collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_index = feature_values.len() / 2;
        let split_value = feature_values[median_index];

        let (left_counts, right_counts) = count_class_occurrences_split(targets, &data, feature_index, split_value);

        let gini_index = calculate_gini_index(&left_counts, &right_counts);
        if gini_index < best_gini_index {
            best_gini_index = gini_index;
            best_feature_index = feature_index;
            best_split_value = split_value;
        }
    }

    (best_feature_index, best_split_value)
}


fn count_class_occurrences_split(
    targets: &[usize],
    data: &[Vec<f64>],
    feature_index: usize,
    split_value: f64,
) -> (HashMap<usize, usize>, HashMap<usize, usize>) {
    let mut left_counts: HashMap<usize, usize> = HashMap::new();
    let mut right_counts: HashMap<usize, usize> = HashMap::new();

    for i in 0..data.len() {
        let feature_value = data[i][feature_index];
        let class_label = targets[i];

        if feature_value <= split_value {
            *left_counts.entry(class_label).or_insert(0) += 1;
        } else {
            *right_counts.entry(class_label).or_insert(0) += 1;
        }
    }

    (left_counts, right_counts)
}

fn calculate_gini_index(
    left_counts: &HashMap<usize, usize>,
    right_counts: &HashMap<usize, usize>,
) -> f64 {
    let left_total: usize = left_counts.values().sum();
    let right_total: usize = right_counts.values().sum();

    let left_gini = calculate_gini_impurity(left_counts.values().cloned().collect());
    let right_gini = calculate_gini_impurity(right_counts.values().cloned().collect());

    let total = left_total + right_total;
    let gini_index = (left_total as f64 / total as f64) * left_gini
        + (right_total as f64 / total as f64) * right_gini;

    gini_index
}

fn calculate_gini_impurity(class_counts: Vec<usize>) -> f64 {
    let total: usize = class_counts.iter().sum();
    let impurity: f64 = class_counts
        .iter()
        .map(|&count| {
            let p = count as f64 / total as f64;
            p * (1.0 - p)
        })
        .sum();

    impurity
}


fn predict_sample(sample: &[f64], node: Node) -> usize {
    match node {
        Node::Split { feature_index, split_value, left, right } => {
            let sample_value = sample[feature_index];

            if sample_value <= split_value {
                predict_sample(sample, *left)
            } else {
                predict_sample(sample, *right)
            }
        }
        Node::Leaf { class_label } => class_label,
        Node::Forward { id: _, feature: _, target: _ } => 0,
        Node::Void {  } => 0,
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
struct RandomForest {
    forest: Vec<DecisionTree>,
    fitted: bool,
}

impl RandomForest {fn new() -> RandomForest{ 
    RandomForest{ 
                forest:  Vec::<DecisionTree>::new(), 
                fitted: false,
                }
    }}

//train the model with sgd or adam
impl RandomForest {
    fn fit(&mut self, path_to_data: &String, num_tree:usize, min_features: usize, max_features: usize, data_fraction: f64, config: &EnvironmentConfig)
        {

        self.fitted = true;
        
        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let fit = env.stream(source)
        .map(move |mut x| {
            //id of the tree which will get for sure the sample
            let tree_id = rand::thread_rng().gen_range(0..=num_tree-1);
            //target class
            let y = x.pop().unwrap() as usize;
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
                    let mut local_trees_data: HashMap<usize, (Vec<Vec<f64>>, Vec<usize>)> = HashMap::new();
                    let mut flag_result = 0;
                    move | x|{
                        if state.get().iter == 1 && flag_result<num_tree{
                         
                            let tree = train_decision_tree(&local_trees_data.get(&flag_result).unwrap().0,
                                                                 &local_trees_data.get(&flag_result).unwrap().1, min_features, max_features);
                            flag_result+=1;
                            Some(tree)
                        }
                        else if state.get().iter==0{
                            let mut features = Vec::new();
                            let mut id_tree = 0;
                            let mut class = 0;
                            match x.root.unwrap() {
                                //get the sample information
                                Node::Forward { id, feature, target } => {
                                    features = feature;
                                    id_tree = id;
                                    class = target;
                                }
                                //no node will be one of the following
                                Node::Split { feature_index: _, split_value: _, left: _, right: _ } => {}
                                Node::Leaf { class_label: _ } => {}
                                Node::Void {} =>{}
                            };
                            
                            //add to the corresponding tree id the features and the class of the sample
                            local_trees_data.entry(id_tree).or_insert((Vec::new(),Vec::new())).0.push(features.clone());
                            local_trees_data.entry(id_tree).or_insert((Vec::new(),Vec::new())).1.push(class);
                            
                            //for each tree probability of data_fraction% to use the sample for training
                            for i in 0..num_tree{
                                if i!=id_tree && rand::thread_rng().gen::<f64>() > (1.-data_fraction){
                                    local_trees_data.entry(i).or_insert((Vec::new(),Vec::new())).0.push(features.clone());
                                    local_trees_data.entry(i).or_insert((Vec::new(),Vec::new())).1.push(class);
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


    fn score(&self, path_to_data: &String, config: &EnvironmentConfig) -> f64{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}
        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let forest = self.forest.clone();

        let predictions = env.stream(source)
    
            .map(move |mut x| {
                let y = x.pop().unwrap();
                let mut class_pred_count: HashMap<usize, usize> = HashMap::new();
                for tree in forest.clone(){
                    let pred = predict_sample(&x,tree.root.unwrap());
                    *class_pred_count.entry(pred).or_insert(0) += 1;
                }
                let pred = get_majority_class(&class_pred_count) as f64;
                if y == pred{
                    return 1.;
                }
                else{
                    return 0.;
                }
            })
            .group_by_avg(|&_k| true, |&v| v).drop_key()   
            .collect_vec();
        
        env.execute();
            
        let result = predictions.get().unwrap()[0];
        result
    }


    fn predict(&self, path_to_data: &String, config: &EnvironmentConfig) -> Vec<usize>{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}
        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let forest = self.forest.clone();

        let predictions = env.stream(source)
    
            .map(move |x| {
                let mut class_pred_count: HashMap<usize, usize> = HashMap::new();
                for tree in forest.clone(){
                    let pred = predict_sample(&x,tree.root.unwrap());
                    *class_pred_count.entry(pred).or_insert(0) += 1;
                }
                get_majority_class(&class_pred_count)
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

    let mut model = RandomForest::new();
    
    let num_tree = 10;
    let min_features = 4;
    let max_features = 8;
    let data_fraction = 0.5;

    
    model.fit(&training_set, num_tree, min_features, max_features, data_fraction, &config);

    //compute the score over the training set
    let score = model.score(&training_set, &config);

    let predictions = model.predict(&data_to_predict, &config);

    let elapsed = start.elapsed();

    // //print!("\nCoefficients: {:?}\n", model.features_coef);
    // //print!("Intercept: {:?}\n", model.intercept);  

    print!("\nScore: {:?}\n", score);
    print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<usize>>());
    eprintln!("\nElapsed: {elapsed:?}");
    eprintln!("{:#?}",model.forest);

}