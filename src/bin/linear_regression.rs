use noir::prelude::*;

use serde::{Deserialize, Serialize};

use std::{time::Instant};

use noir_ml::sample::Sample;
use noir_ml::matrix_utils::*;


//State for SGD
#[derive(Clone, Serialize, Deserialize, Default)]
struct StateSGD {
    //regression coefficients
    weights: Vec<f64>,
    //total gradient of the batch
    global_grad: Vec<f64>,
    //iterations over the dataset
    epoch: usize,
}

impl StateSGD {
    fn new() -> StateSGD {
        StateSGD {
            weights:  Vec::<f64>::new(),
            global_grad: Vec::<f64>::new(),
            epoch : 0,
        }}}


//State for ADAM
#[derive(Clone, Serialize, Deserialize, Default)]
struct StateAdam {
    //regression coefficients
    weights: Vec<f64>,
    //total gradient of the batch
    global_grad: Vec<f64>,
    //first gradient moment
    m: Vec<f64>,
    //second gradient moment
    v: Vec<f64>, 
    //iterations over the dataset
    epoch: usize,}

impl StateAdam {
    fn new() -> StateAdam {
        StateAdam {
            weights:  Vec::<f64>::new(),
            global_grad: Vec::<f64>::new(),
            m: Vec::<f64>::new(),
            v: Vec::<f64>::new(),
            epoch : 0,
        }}}



//Linear regression model
#[derive(Clone, Debug)]
struct LinearRegression {
    //all n+1 coeff
    coefficients: Vec<f64>,
    //n coeff without intercept
    features_coef: Vec<f64>,
    intercept: f64,
    //R2
    score: f64,
    //if the features have been normalized during training
    normalization: bool,
    //mean of the training set (includes target)
    train_mean: Vec<f64>,
    //standard deviation of the training set (includes target)
    train_std: Vec<f64>,
    //if the model has been trained at least one time
    fitted: bool,
    method: String,
}


impl LinearRegression {
    fn new() -> LinearRegression {

        LinearRegression {
            coefficients: Vec::<f64>::new(),
            features_coef: Vec::<f64>::new(),
            intercept: 0.,
            score: 0.,
            normalization: false,
            train_mean: Vec::<f64>::new(),
            train_std: Vec::<f64>::new(),
            fitted: false,
            method: "".to_string(),
        }}
}


impl LinearRegression {
    fn fit_ols(mut self, path_to_data: &String, normalize: bool, config: &EnvironmentConfig)->LinearRegression{
            
            self.fitted = true;

            //to normalize the samples we need their mean and std
            if normalize==true{

                self.normalization = true;
    
                let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
                let mut env0 = StreamEnvironment::new(config.clone());
                env0.spawn_remote_workers();
                
                //get the mean of all the features + target and their second moments E[x^2]
                let features_mean = env0.stream(source)
                .map(move |mut x| 
                    {   
                        //push the square of the features to get both E[x] and E[x^2] 
                        x.0.extend(x.0.iter().map(|xi| xi.powi(2)).collect::<Vec<f64>>());
                        x
                    })
                .group_by_avg(|_x| true, |x| x.clone()).drop_key().collect_vec();
    
                env0.execute();
                
                let mut moments:Vec<f64> = vec![0.;1];
    
                if let Some(means_vector) = features_mean.get() {
                    moments = means_vector[0].0.clone();
                }
                
                self.train_mean= moments.iter().take(moments.len()/2).cloned().collect::<Vec<f64>>();
                
                self.train_std = moments.iter().skip(moments.len()/2).zip(self.train_mean.iter()).map(|(e2,avg)| (e2-avg.powi(2)).sqrt()).collect();
             }


            let train_mean = self.train_mean.clone();
            let train_std = self.train_std.clone();

            let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
            let mut env = StreamEnvironment::new(config.clone());
            env.spawn_remote_workers();
            
            self.method = "OLS".to_string();
            //return the weights computed with OLS thanks to the model.fit method
             let fit = env.stream(source.clone())
                .replay(
                    2,
                    StateSGD::new(),

                    move |s, state| 
                    {
                        //shuffle the samples
                        s.shuffle()
                        //each replica filter a number of samples equal to batch size and
                        //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                        .rich_filter_map({
                            let mut local_matrix: Vec<Vec<f64>> = Vec::new();
                            let mut target = Vec::<f64>::new();
                            let mut flag_result = 0;
                            move |mut x|{
                                //first iteration: populate the matrix
                                if state.get().epoch==0{
                                    if self.normalization==true{
                                        //scale the features and the target
                                        x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                    }
                                    let last = x.0.len()-1;
                                    target.push(x.0[last]);
                                    x.0[last] = 1.;
                                    local_matrix.push(x.0);                               
     
                                    None
                                }
                                //second iteration: compute local weights
                                else{
                                    if flag_result==0{
                                        flag_result = 1;
                                        let (mut q,mut r) = qr_decomposition(&local_matrix);
                                        r = invert_r(&r);
                                        q = transpose(&q);
                                        let weights_ols = matrix_vector_product(&matrix_product(&r, &q), &target);
                                        Some(Sample(weights_ols))
                                        //Some(Sample(ols(&local_matrix, &target)))
                                    } 
                                    else {
                                        None
                                    } }
                            }})
                        //the average of the gradients is computed and forwarded as a single value
                        .group_by_avg(|_x| true, |x| x.clone()).drop_key().max_parallelism(1)
                    },

                    move |local_grad: &mut Sample, avg_grad| 
                    {   
                        *local_grad = avg_grad;
                    },

                    move |state, local_grad| 
                    {   
                        //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                        if local_grad.0.len()!=0{
                        state.global_grad = local_grad.0.clone();}
                    },

                    move|state| 
                    {   
                        state.epoch+=1;
                        state.weights = state.global_grad.clone();
                        state.epoch<2
                    },

                )
                .collect_vec();

        env.execute();

        if let Some(res) = fit.get() {
            let state = &res[0];
            self.coefficients = state.weights.clone();
            self.intercept = state.weights[self.coefficients.len()-1];}

            LinearRegression {
                coefficients: self.coefficients.clone(),
                features_coef: self.coefficients.iter().take(self.coefficients.len()-1).cloned().collect::<Vec::<f64>>(),
                intercept: self.intercept,
                score: 0.,
                normalization: self.normalization,
                train_mean: train_mean,
                train_std: train_std,
                fitted: self.fitted,
                method: "OLS".to_string()
            }
        }
}


//train the model with sgd or adam
impl LinearRegression {
    fn fit(mut self, path_to_data: &String, method: String, num_iters:usize, learn_rate: f64, batch_size: usize, normalize: bool, weight_decay: bool, config: &EnvironmentConfig)
    -> LinearRegression{
            
            self.fitted = true;

            //to normalize the samples we need their mean and std
            if normalize==true{

            self.normalization = true;

            let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
            let mut env0 = StreamEnvironment::new(config.clone());
            env0.spawn_remote_workers();
            
            //get the mean of all the features + target and their second moments E[x^2]
            let features_mean = env0.stream(source)
            .map(move |mut x| 
                {   
                    //push the square of the features to get both E[x] and E[x^2] 
                    x.0.extend(x.0.iter().map(|xi| xi.powi(2)).collect::<Vec<f64>>());
                    x
                })
            .group_by_avg(|_x| true, |x| x.clone()).drop_key().collect_vec();

            env0.execute();
            
            let mut moments:Vec<f64> = vec![0.;1];

            if let Some(means_vector) = features_mean.get() {
                moments = means_vector[0].0.clone();
            }
            
            self.train_mean= moments.iter().take(moments.len()/2).cloned().collect::<Vec<f64>>();
            
            self.train_std = moments.iter().skip(moments.len()/2).zip(self.train_mean.iter()).map(|(e2,avg)| (e2-avg.powi(2)).sqrt()).collect();
         }


        let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        

        //choose the algorithm
        match  method.as_str(){

            "ADAM" | "adam"  =>

            {    
                self.method = "ADAM".to_string();
                let beta1 = 0.9;
                let beta2 = 0.999;

                let fit = env.stream(source.clone())
                    .replay(
                    num_iters,
                    StateAdam::new(),

                    move |s, state| 
                    {
                        //shuffle the samples
                        s.shuffle()
                        //each replica filter a number of samples equal to batch size and
                        //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                        .rich_filter_map({
                            let mut count = 0;
                            move |mut x|{
                                let dim = x.0.len();
                                //at first iter (epoch=0) count goes from 0 to batch_size; at epoch=1 from batchsize to 2*batch_size...
                                if count < batch_size * (state.get().epoch+1) {
                                    count+=1;
                                    if self.normalization==true{
                                        //scale the features and the target
                                        x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                        }
                                    //the target is in the last element of each sample
                                    let y: f64 = x.0[dim-1]; 
                                    //switch the target with a 1 for the intercept
                                    x.0[dim-1] = 1.;
                                    
                                    let mut current_weights = &state.get().weights;
                                    let vec = vec![0.;dim];
                                    if state.get().epoch == 0{
                                        current_weights = &vec;
                                    }

                                    let prediction: f64 = x.0.iter().zip(current_weights.iter()).map(|(xi, wi)| xi * wi).sum();
                                    let error = prediction - y;
                                    let sample_grad: Vec<f64> = x.0.iter().map(|xi| xi * error).collect();                            
                                    Some(Sample(sample_grad))
                                    }
                                else {None}}})
                        //the average of the gradients is computed and forwarded as a single value
                        .group_by_avg(|_x| true, |x| x.clone()).drop_key().max_parallelism(1)
                    },

                    move |local_grad: &mut Sample, avg_grad| 
                    {   
                        *local_grad = avg_grad;
                    },

                    move |state, local_grad| 
                    {   
                        //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                        if local_grad.0.len()!=0{
                        state.global_grad = local_grad.0.clone();}
                    },

                    move|state| 
                    {   
                        //initialize
                        if state.epoch==0{
                            let dim = state.global_grad.len();
                            state.m = vec![0.;dim];
                            state.v = vec![0.;dim];
                            state.weights = vec![0.;dim];
                        }
                        //update iterations
                        state.epoch +=1;
                        //update the moments
                        state.m = state.m.iter().zip(state.global_grad.iter()).map(|(mi, gi)|beta1 * mi + ((1. - beta1) * gi)).collect();
                        state.v =  state.v.iter().zip(state.global_grad.iter()).map(|(vi, gi)|beta2 * vi + ((1. - beta2) * gi.powi(2))).collect();
                        let m_hat: Vec<f64> = state.m.iter().map(|mi|mi/(1. - beta1.powi(state.epoch as i32))).collect();
                        let v_hat: Vec<f64> = state.v.iter().map(|vi|vi/(1. - beta2.powi(state.epoch as i32))).collect();
                        let adam: Vec<f64> =  m_hat.iter().zip(v_hat.iter()).map(|(mi,vi)| mi/(vi.sqrt()+1e-6)).collect();
                        //update the weights (optional with weight decay)
                        state.weights = state.weights.iter().zip(adam.iter()).map(|(wi,a)| wi - learn_rate*a).collect();
                        if weight_decay==true{
                            state.weights = state.weights.iter().map(|wi| wi -  learn_rate * 0.002 * wi).collect();
                        }
                        //tolerance=gradient's L2 norm for the stop condition
                        let tol: f64 = adam.iter().map(|v| v*v).sum();
                        //reset the global gradient for the next iteration
                        state.global_grad = vec![0.;adam.len()];
                        //loop condition
                        state.epoch < num_iters && tol.sqrt() > 1e-4
                    },

                )
                .collect_vec();

        env.execute();

        if let Some(res) = fit.get() {
            let state = &res[0];
            self.coefficients = state.weights.clone();
            self.intercept = state.weights[state.weights.len()-1];}

            LinearRegression {
                coefficients: self.coefficients.clone(),
                features_coef: self.coefficients.iter().take(self.coefficients.len()-1).cloned().collect::<Vec::<f64>>(),
                intercept: self.intercept,
                score: 0.,
                normalization: self.normalization,
                train_mean: train_mean,
                train_std: train_std,
                fitted: self.fitted,
                method: "ADAM".to_string()
            }
            }

            "OLS" => 

            {
            self.method = "OLS".to_string();
            //return the weights computed with OLS thanks to the model.fit method
             let fit = env.stream(source.clone())
                .replay(
                    2,
                    StateSGD::new(),

                    move |s, state| 
                    {
                        //shuffle the samples
                        s.shuffle()
                        //each replica filter a number of samples equal to batch size and
                        //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                        .rich_filter_map({
                            let mut local_matrix: Vec<Vec<f64>> = Vec::new();
                            let mut target = Vec::<f64>::new();
                            let mut flag_result = 0;
                            move |mut x|{
                                //first iteration: populate the matrix
                                if state.get().epoch==0{
                                    if self.normalization==true{
                                        //scale the features and the target
                                        x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                    }
                                    let last = x.0.len()-1;
                                    target.push(x.0[last]);
                                    x.0[last] = 1.;
                                    local_matrix.push(x.0);                               
     
                                    None
                                }
                                //second iteration: compute local weights
                                else{
                                    if flag_result==0{
                                        flag_result = 1;
                                        let local_transpose = transpose(&local_matrix);
                                        local_matrix = matrix_product(&local_transpose, &local_matrix);
                                        local_matrix = invert_matrix(&local_matrix);
                                        local_matrix = matrix_product(&local_matrix, &local_transpose);
                                        let weights_ols = matrix_vector_product(&local_matrix,&target);
                                       Some(Sample(weights_ols))
                                    } 
                                    else {
                                        None
                                    } }
                            }})
                        //the average of the gradients is computed and forwarded as a single value
                        .group_by_avg(|_x| true, |x| x.clone()).drop_key().max_parallelism(1)
                    },

                    move |local_grad: &mut Sample, avg_grad| 
                    {   
                        *local_grad = avg_grad;
                    },

                    move |state, local_grad| 
                    {   
                        //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                        if local_grad.0.len()!=0{
                        state.global_grad = local_grad.0.clone();}
                    },

                    move|state| 
                    {   
                        state.epoch+=1;
                        state.weights = state.global_grad.clone();
                        state.epoch<2
                    },

                )
                .collect_vec();

        env.execute();

        if let Some(res) = fit.get() {
            let state = &res[0];
            self.coefficients = state.weights.clone();
            self.intercept = state.weights[self.coefficients.len()-1];}

            LinearRegression {
                coefficients: self.coefficients.clone(),
                features_coef: self.coefficients.iter().take(self.coefficients.len()-1).cloned().collect::<Vec::<f64>>(),
                intercept: self.intercept,
                score: 0.,
                normalization: self.normalization,
                train_mean: train_mean,
                train_std: train_std,
                fitted: self.fitted,
                method: "OLS".to_string()
            }

            }



            "SGD" | _ => 

            {
            self.method = "SGD".to_string();
            //return the weights computed with SGD thanks to the model.fit method
             let fit = env.stream(source.clone())
                .replay(
                    num_iters,
                    StateSGD::new(),

                    move |s, state| 
                    {
                        //shuffle the samples
                        s.shuffle()
                        //each replica filter a number of samples equal to batch size and
                        //for each sample computes the gradient of the mse loss (a vector of length: n_features+1)
                        .rich_filter_map({
                            let mut count = 0;
                            move |mut x|{
                                let dim = x.0.len();
                                //at first iter (epoch=0) count goes from 0 to batch_size; at epoch=1 from batchsize to 2*batch_size...
                                if count < batch_size * (state.get().epoch+1) {
                                    count+=1;
                                    if self.normalization==true{
                                        //scale the features and the target
                                        x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                        }
                                    //the target is in the last element of each sample
                                    let y: f64 = x.0[dim-1]; 
                                    //switch the target with a 1 for the intercept
                                    x.0[dim-1] = 1.;

                                    let mut current_weights = &state.get().weights;
                                    let vec = vec![0.;dim];
                                    if state.get().epoch == 0{
                                        current_weights = &vec;
                                    }

                                    let prediction: f64 = x.0.iter().zip(current_weights.iter()).map(|(xi, wi)| xi * wi).sum();
                                    let error = prediction - y;
                                    let sample_grad: Vec<f64> = x.0.iter().map(|xi| xi * error).collect();
                                    Some(Sample(sample_grad))
                                    }
                                else {None}}})
                        //the average of the gradients is computed and forwarded as a single value
                        .group_by_avg(|_x| true, |x| x.clone()).drop_key().max_parallelism(1)
                    },

                    move |local_grad: &mut Sample, avg_grad| 
                    {   
                        *local_grad = avg_grad;
                    },

                    move |state, local_grad| 
                    {   
                        //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                        if local_grad.0.len()!=0{
                        state.global_grad = local_grad.0.clone();}
                    },

                    move|state| 
                    {   
                        //initialize
                        if state.epoch==0{
                            state.weights = vec![0.;state.global_grad.len()]
                        }
                        //update iterations
                        state.epoch +=1;
                        //update the weights (optional with weight decay)
                        state.weights = state.weights.iter().zip(state.global_grad.iter()).map(|(wi,g)| wi - learn_rate*g).collect();
                        if weight_decay==true{
                            state.weights = state.weights.iter().map(|wi| wi -  learn_rate * 0.002 * wi).collect();
                        }
                        //tolerance=gradient's L2 norm for the stop condition
                        let tol: f64 = state.global_grad.iter().map(|v| v*v).sum();
                        //reset the global gradient for the next iteration
                        state.global_grad = vec![0.;state.weights.len()];
                        //loop condition
                        state.epoch < num_iters && tol.sqrt() > 1e-4
                    },

                )
                .collect_vec();

        env.execute();

        if let Some(res) = fit.get() {
            let state = &res[0];
            self.coefficients = state.weights.clone();
            self.intercept = state.weights[self.coefficients.len()-1];}

            LinearRegression {
                coefficients: self.coefficients.clone(),
                features_coef: self.coefficients.iter().take(self.coefficients.len()-1).cloned().collect::<Vec::<f64>>(),
                intercept: self.intercept,
                score: 0.,
                normalization: self.normalization,
                train_mean: train_mean,
                train_std: train_std,
                fitted: self.fitted,
                method: "SGD".to_string()
            }

            }

           
        }     

    }
    }

        
//score takes as input also the target, a dataset with row of length num_features+1
//can be used for evaluating both training and test set
impl LinearRegression {
    fn score(mut self, path_to_data: &String, config: &EnvironmentConfig) -> f64{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}

        let mut avg_y = 0.;
        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');

        if self.normalization == false
        {

            let mut env = StreamEnvironment::new(config.clone());
            env.spawn_remote_workers();
        
            let res2 = env.stream(source.clone())
            .group_by_avg(|_x| true, move|x| x.0[x.0.len()-1]).drop_key().collect_vec();
        
            env.execute();
        
            if let Some(res2) = res2.get() {
            avg_y = res2[0];}
        }

        let mut env = StreamEnvironment::new(config.clone());

        env.spawn_remote_workers();
    
        //compute the residuals sums for the R2
        let score = env.stream(source)
    
            .map(move |mut x| {
                let mut mean_y = avg_y;
                //scale the features and the target   
                if self.normalization==true{
                    x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                    mean_y = 0.; 
                    }
                let dim = x.0.len();                     
                let y = x.0[dim-1];
                x.0[dim-1] = 1.;   
                let pred: f64 = x.0.iter().zip(self.coefficients.iter()).map(|(xi,wi)| xi*wi).sum();
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
        self.score = r2;
        
        r2

    }
        }


//predict doesn't take as input the target, so a dataset with row of length num_features
impl LinearRegression {
    fn predict(self, path_to_data: &String, config: &EnvironmentConfig) -> Vec<f64>{

        if self.fitted != true {panic!("Can't compute predictions before fitting the model!");}

        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');

        let mut y_mean:f64 = 0.;
        let mut y_std:f64 = 1.;

        if self.normalization == false
        {

            let mut env = StreamEnvironment::new(config.clone());
            env.spawn_remote_workers();
        
            let res2 = env.stream(source.clone())
            .group_by_avg(|_x| true, move|x| x.0[x.0.len()-1]).drop_key().collect_vec();
        
            env.execute();
        
            if let Some(res2) = res2.get() {
                y_mean = res2[0];}
        }

        else{
            y_mean = self.train_mean[self.train_mean.len()-1];
            y_std = self.train_std[self.train_mean.len()-1];
        }

        let mut env = StreamEnvironment::new(config.clone());

        env.spawn_remote_workers();
        

        let prediction = env.stream(source)
    
            .map(move |mut x| {
                let pred:f64;
                x.0.push(1.); //push the intercept
                if self.normalization==true{
                    x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                    pred = x.0.iter().zip(self.coefficients.iter()).map(|(xi,wi)| xi*wi).sum::<f64>() * y_std + y_mean;
                }   
                else{                  
                pred = x.0.iter().zip(self.coefficients.iter()).map(|(xi,wi)| xi*wi).sum();
                }
                pred           
            })
            .collect_vec();
        
        env.execute();

        let mut pred = vec![0.;1];
        if let Some(res3) = prediction.get() {
            pred = res3;}
        
        pred

    }
        }




fn main() {
    
    let (config, _args) = EnvironmentConfig::from_args();

    let training_set = "forest_fire.csv".to_string();
    //let test_set = "housing_numeric.csv".to_string();
    let data_to_predict = "forest_fire.csv".to_string();

    let start = Instant::now();

    let mut model = LinearRegression::new();
    
    let method = "OLS".to_string(); //"ADAM".to_string() //"SGD".to_string()
    let num_iters = 50;
    let learn_rate = 1e-1;
    let batch_size = 100;
    let normalize = false;
    let weight_decay = false;

    //return the trained model
    model = model.fit(&training_set, method, num_iters, learn_rate, batch_size, normalize, weight_decay, &config);

    //model = model.fit_ols(&training_set, normalize, &config);
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