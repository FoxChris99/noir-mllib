//SISTEMARE: le prediction non sono ordinate
use noir::prelude::*;
use std::time::Instant;
use noir_ml::{sample::Sample,basic_stat::get_moments,basic_stat::sigmoid, sgd_regressor::logistic_sgd, adam_regressor::logistic_adam};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

//Linear regression model
#[derive(Clone, Debug)]
struct LogisticRegression {
    //all n+1 coeff
    coefficients: Vec<Vec<f64>>,
    //if the features have been normalized during training
    normalization: bool,
    //mean of the training set (includes target)
    train_mean: Vec<f64>,
    //standard deviation of the training set (includes target)
    train_std: Vec<f64>,
    //if the model has been trained at least one time
    fitted: bool,
    num_classes: usize
}


impl LogisticRegression {
    fn new(num_classes: usize) -> LogisticRegression {
        LogisticRegression {
            coefficients: Vec::new(),
            normalization: false,
            train_mean: Vec::<f64>::new(),
            train_std: Vec::<f64>::new(),
            fitted: false,
            num_classes: num_classes
        }}
}

//train the model with sgd or adam
impl LogisticRegression {
    fn fit(&mut self, path_to_data: &String, method: String, num_iters:usize, learn_rate: f64, data_fraction: f64, normalize: bool, weight_decay: bool, config: &EnvironmentConfig)
        {
            
        self.fitted = true;

        //to normalize the samples we need their mean and std
        if normalize==true{
            self.normalization = true;
            (self.train_mean, self.train_std) = get_moments(&config, &path_to_data);
        }
        
        // let source2 = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');
        // let mut env2 = StreamEnvironment::new(config.clone());
        // env2.spawn_remote_workers();

        // let samples_per_class = env2.stream(source)
        // .group_by_count(|x| x.0[x.0.len()-1] as usize).collect_vec().get().unwrap();

        // env2.execute();
        
        // print!("\nNumber of samples per class:\n ");
        // for (class, count) in samples_per_class.iter(){
        //     print!("class {:?} : {:?}\n", class, count);
        // }

        // let num_classes = samples_per_class.len();
        let weights;
        let num_classes = self.num_classes;
        //choose the iterative algorithm
        match  method.as_str(){

            "ADAM" | "adam"  =>
            {    
            let state = logistic_adam(num_classes, weight_decay, learn_rate, data_fraction, num_iters, path_to_data, normalize, self.train_mean.clone(), self.train_std.clone(), config);
            weights = state.weights;
            },

            "SGD" | "sgd" | _ => 
            {
            let state = logistic_sgd(num_classes, weight_decay, learn_rate, data_fraction, num_iters, path_to_data, normalize, self.train_mean.clone(), self.train_std.clone(), config);
            weights = state.weights;
            }
        }    

        self.coefficients = weights.clone();
    }
}     



       
//score takes as input also the target, a dataset with row of length num_features+1
//can be used for evaluating both training and test set
impl LogisticRegression {
    fn score(&self, path_to_data: &String, config: &EnvironmentConfig) -> f64{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}
        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let normalization = self.normalization;
        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        let coefficients = self.coefficients.clone();

        let score = env.stream(source)
    
            .map(move |mut x| {
                let dim = x.0.len();   
                let class = x.0[dim-1] as usize; //class number  
                let num_classes = coefficients.len();
                //scale the features
                if normalization==true{
                    x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                    }               
                x.0[dim-1] = 1.;   
                let mut prediction = vec![0.;num_classes];
                for i in 0..num_classes{
                    let y_hat:f64 = x.0.iter().zip(coefficients[i].iter()).map(|(xi, wi)| xi * wi).sum();
                    prediction[i] = sigmoid(y_hat);
                }
                let mut argmax = 0;
                let mut max = 0.;
                for (i, &x) in prediction.iter().enumerate() {
                    if x> max{
                        max = x;
                        argmax = i;
                    }
                }
                //let argmax = prediction.iter().enumerate().max_by_key(|(_, &x)| x.partial_cmp(&f64::NAN).unwrap()).unwrap().0;
                if class==argmax{
                    1.
                }        
                else {
                    0.
                }
            })    
            .group_by_avg(|&_k| true, |&v| v).drop_key()
            .collect_vec();
       
        env.execute();
            
        let mut evaluation_score: f64 = -999.;
        if let Some(res3) = score.get() {
            evaluation_score = res3[0];}
        
        evaluation_score
    }
}

  
//predict doesn't take as input the target, so a dataset with row of length num_features
impl LogisticRegression {
    fn predict(&self, path_to_data: &String, config: &EnvironmentConfig) -> Vec<usize>{

        if self.fitted != true {panic!("Can't compute predictions before fitting the model!");}
        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();       

        let normalization = self.normalization;
        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        let coefficients = self.coefficients.clone();

        let prediction = env.stream(source)
    
            .map(move |mut x| {
                let mut highest_prob = 0.;
                let mut predicted_class:usize = 0;
                let num_classes = coefficients.len();
                x.0.push(1.); //push the intercept
                if normalization==true{
                        x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                        }  
                for i in 0..num_classes {
                    let pred = x.0.iter().zip(coefficients[i].iter()).map(|(xi,wi)| xi*wi).sum();
                    if highest_prob > pred{
                        predicted_class = i;
                        highest_prob = pred;
                    }            
                }
                predicted_class           
            })
            .collect_vec();
        
        env.execute();

        let mut pred = vec![0;1];
        if let Some(res3) = prediction.get() {
            pred = res3;}
        
        pred
    }
}




fn main() { 
    let (config, _args) = EnvironmentConfig::from_args();

    let training_set = "data/class_1milion_4features_multiclass.csv".to_string();
    let data_to_predict = "data/class_1milion_4features_multiclass.csv".to_string();

    let num_classes = 4;
    let mut model = LogisticRegression::new(num_classes);
    
    let method = "SGD".to_string(); //"ADAM".to_string()
    let num_iters = 100;
    let learn_rate = 1e-1;
    let data_fraction = 1.;
    let normalize = false;
    let weight_decay = false;

    let start = Instant::now();
    //return the trained model
    model.fit(&training_set, method, num_iters, learn_rate, data_fraction, normalize, weight_decay, &config);

    let elapsed = start.elapsed();

    let start_score = Instant::now();
    //compute the score over the training set
    let score = model.score(&training_set, &config);
    let elapsed_score = start_score.elapsed();
    
    let start_pred = Instant::now();
    let predictions = model.predict(&data_to_predict, &config);
    let elapsed_pred = start_pred.elapsed();
    
 
    print!("\nScore: {:?}\n", score);
    print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<usize>>());
    eprintln!("\nElapsed fit: {elapsed:?}");
    eprintln!("\nElapsed score: {elapsed_score:?}"); 
    eprintln!("\nElapsed pred: {elapsed_pred:?}");     

}