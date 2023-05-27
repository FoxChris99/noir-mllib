use noir::prelude::*;

use std::{time::Instant};

use noir_ml::{sgd_regressor::linear_sgd, adam_regressor::linear_adam, basic_stat::get_moments, sample::Sample};


#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;


//Linear regression model
#[derive(Clone, Debug)]
struct RegularizedRegression {
    reg_type: String,
    //all n+1 coeff
    coefficients: Vec<f64>,
    //n coeff without intercept
    features_coef: Vec<f64>,
    intercept: f64,
    //if the features have been normalized during training
    normalization: bool,
    //mean of the training set (includes target)
    train_mean: Vec<f64>,
    //standard deviation of the training set (includes target)
    train_std: Vec<f64>,
    //if the model has been trained at least one time
    fitted: bool,
}


impl RegularizedRegression {
    fn new(regularization_type: &str) -> RegularizedRegression {

        let reg_type = match  regularization_type.to_lowercase().as_str(){
            "lasso" | "ridge" | "elastic-net" =>regularization_type.to_lowercase(),
           "none" | _ =>"none".to_string()
        };
        
        RegularizedRegression {
            reg_type: reg_type,
            coefficients: Vec::<f64>::new(),
            features_coef: Vec::<f64>::new(),
            intercept: 0.,
            normalization: false,
            train_mean: Vec::<f64>::new(),
            train_std: Vec::<f64>::new(),
            fitted: false,
        }}
}


//train the model with sgd or adam
impl RegularizedRegression {
    fn fit(&mut self, path_to_data: &String, method: String, num_iters:usize, learn_rate: f64, 
        data_fraction: f64, normalize: bool, lambda: f64, weight_decay: bool, config: &EnvironmentConfig)
        {
            
        self.fitted = true;
        //to normalize the samples we need their mean and std
        if normalize==true{
            self.normalization = true;
            (self.train_mean, self.train_std) = get_moments(&config, &path_to_data);
        }
                
        let weights;
        //choose the iterative algorithm
        match  method.as_str(){

            "ADAM" | "adam"  =>
            {    
            let state = linear_adam(weight_decay, learn_rate, data_fraction, num_iters, path_to_data, normalize, self.train_mean.clone(), self.train_std.clone(), config, self.reg_type.as_str(), lambda);
            weights = state.weights;
            },

            "SGD" | "sgd" | _ => 
            {
            let state = linear_sgd(weight_decay, learn_rate, data_fraction, num_iters, path_to_data, normalize, self.train_mean.clone(), self.train_std.clone(), config, self.reg_type.as_str(), lambda);
            weights = state.weights;
            }
        }    
        self.coefficients = weights.clone();
        self.features_coef = weights.iter().take(weights.len()-1).cloned().collect::<Vec::<f64>>();
        self.intercept = weights[weights.len()-1];

    }
    

        
//score takes as input also the target, a dataset with row of length num_features+1
//can be used for evaluating both training and test set
    fn score(&self, path_to_data: &String, config: &EnvironmentConfig) -> f64{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}
        let mut avg_y = 0.;
        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');
        //if the data weren't normalized then we need to compute the mean of the target
        if self.normalization == false
        {
            let mut env = StreamEnvironment::new(config.clone());
            env.spawn_remote_workers();
            let res = env.stream(source.clone())
            .group_by_avg(|_x| true, move|x| x.0[x.0.len()-1]).drop_key().collect_vec();
            env.execute();
            avg_y = res.get().unwrap()[0];
        }

        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let normalization = self.normalization;
        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        let coefficients = self.coefficients.clone();
        //compute the residuals sums for the R2
        let score = env.stream(source)
                
                .map(move |mut x| {
                let mut mean_y = avg_y;
                //scale the features and the target   
                if normalization==true{
                    x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                    mean_y = 0.; 
                }
                let dim = x.0.len();                     
                let y = x.0[dim-1];
                x.0[dim-1] = 1.;   
                let pred: f64 = x.0.iter().zip(coefficients.iter()).map(|(xi,wi)| xi*wi).sum();
                [(y-pred).powi(2),(y-mean_y).powi(2)]           
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


//predict doesn't take as input the target, but a new dataset with rows of length num_features
    fn predict(&self, path_to_data: &String, config: &EnvironmentConfig) -> Vec<f64>{

        if self.fitted != true {panic!("Can't compute predictions before fitting the model!");}
        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let normalization = self.normalization;
        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        let coefficients = self.coefficients.clone();

        let predictions = env.stream(source)    
            .map(move |mut x| {
                let pred:f64;
                let dim = x.0.len();
                if dim==coefficients.len()-1{
                    x.0.push(1.); //push the intercept
                }
                else{
                    x.0[dim-1] = 1.;//when the dataset has the target as last column like in the test set
                }
                
                if normalization==true{
                    x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                    let y_mean = train_mean[train_mean.len()-1];
                    let y_std = train_std[train_mean.len()-1];
                    pred = x.0.iter().zip(coefficients.iter()).map(|(xi,wi)| xi*wi).sum::<f64>() * y_std + y_mean;
                }   
                else{                  
                pred = x.0.iter().zip(coefficients.iter()).map(|(xi,wi)| xi*wi).sum();
                }
                pred           
            })
            .collect_vec();
        
        env.execute();
        
        predictions.get().unwrap()
    }
}




fn main() {
    
    let (config, _args) = EnvironmentConfig::from_args();

    let training_set = "housing_numeric.csv".to_string();
    //let test_set = "housing_numeric.csv".to_string();
    let data_to_predict = "housing_numeric.csv".to_string();

    let start = Instant::now();

    let mut model = RegularizedRegression::new("Lasso"); //Ridge //elastic-net //none
    
    let method = "ADAM".to_string(); //"ADAM".to_string()
    let num_iters = 1000;
    let learn_rate = 1e-2;
    let data_fraction = 0.001;
    let normalize = true;
    let lambda = 0.004;
    let weight_decay = false;

    //return the trained model
    model.fit(&training_set, method, num_iters, learn_rate, data_fraction, normalize, lambda, weight_decay, &config);

    //compute the score over the training set
    let r2 = model.clone().score(&training_set, &config);

    //let r2_test = model.clone().score(&test_set, &config);

    let predictions = model.clone().predict(&data_to_predict, &config);

    let elapsed = start.elapsed();

    print!("\nCoefficients: {:?}\n", model.features_coef);
    print!("Intercept: {:?}\n", model.intercept);  

    print!("\nR2 score: {:?}\n", r2);
    //print!("\nTest score: {:?}\n", r2_test);

    print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<f64>>());


    eprintln!("\nElapsed: {elapsed:?}");

    
}