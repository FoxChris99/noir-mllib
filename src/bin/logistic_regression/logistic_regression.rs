//SISTEMARE: dato che c'è lo shuffle nel training, nel predict poi si sfalsano i label delle classi
//bisogna ritornare il label delle classi dopo il training per permettere interpretazione

use noir::prelude::*;

use serde::{Deserialize, Serialize};

use std::{time::Instant};

mod sample;
use sample::Sample;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;


//State for SGD
#[derive(Clone, Serialize, Deserialize, Default)]
struct StateSGD {
    //regression coefficients
    weights: Vec<Vec<f64>>,
    //total gradient of the batch
    global_grad: Vec<Vec<f64>>,
    //iterations over the dataset
    epoch: usize,
}

impl StateSGD {
    fn new() -> StateSGD {
        StateSGD {
            weights:  Vec::new(),
            global_grad: Vec::new(),
            epoch : 0,
        }}}


//State for ADAM
#[derive(Clone, Serialize, Deserialize, Default)]
struct StateAdam {
    //regression coefficients
    weights: Vec<Vec<f64>>,
    //total gradient of the batch
    global_grad: Vec<Vec<f64>>,
    //first gradient moment
    m: Vec<Vec<f64>>,
    //second gradient moment
    v: Vec<Vec<f64>>, 
    //iterations over the dataset
    epoch: usize,}

impl StateAdam {
    fn new() -> StateAdam {
        StateAdam {
            weights:  Vec::new(),
            global_grad: Vec::new(),
            m: Vec::new(),
            v: Vec::new(),
            epoch : 0,
        }}}


fn sigmoid(v: f64) -> f64{
    if v >= 0.{
        1./(1. + (-v).exp())
    } 
    else{ 
        v.exp()/(1. + v.exp())
    }
}


//Linear regression model
#[derive(Clone, Debug)]
struct LogisticRegression {
    num_classes: usize,
    //all n+1 coeff
    coefficients: Vec<Vec<f64>>,
    //n coeff without intercept
    features_coef: Vec<Vec<f64>>,
    intercept: Vec<f64>,
    //number of samples correctly classified
    score: f64,
    //if the features have been normalized during training
    normalization: bool,
    //mean of the training set (includes target)
    train_mean: Vec<f64>,
    //standard deviation of the training set (includes target)
    train_std: Vec<f64>,
    //if the model has been trained at least one time
    fitted: bool,
}


impl LogisticRegression {
    fn new(num_classes: usize) -> LogisticRegression {

        LogisticRegression {
            num_classes: num_classes,
            coefficients: Vec::new(),
            features_coef: Vec::new(),
            intercept: Vec::<f64>::new(),
            score: 0.,
            normalization: false,
            train_mean: Vec::<f64>::new(),
            train_std: Vec::<f64>::new(),
            fitted: false,
        }}
}

//train the model with sgd or adam
impl LogisticRegression {
    fn fit(mut self, path_to_data: &String, method: String, num_iters:usize, learn_rate: f64, batch_size: usize, normalize: bool, weight_decay: bool, config: &EnvironmentConfig)
    -> LogisticRegression{
            
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
                                    //the target is in the last element of each sample, y one hot encoding
                                    let mut y = vec![0;self.num_classes];
                                    y[x.0[dim-1] as usize] = 1; //assigned before normalization because it's a classification task
                                    if self.normalization==true{
                                        //scale the features and the target
                                        x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                        };
                                    //switch the target with a 1 for the intercept
                                    x.0[dim-1] = 1.;

                                    let mut current_weights = &state.get().weights;
                                    let vec = vec![vec![0.;dim]; self.num_classes];
                                    if state.get().epoch == 0{
                                        current_weights = &vec;
                                    }

                                    let mut prediction = vec![0.;self.num_classes];
                                    let mut sample_grad = Vec::new();
                                    for i in 0..self.num_classes{
                                        let y_hat:f64 = x.0.iter().zip(current_weights[i].iter()).map(|(xi, wi)| xi * wi).sum();
                                        prediction[i] = sigmoid(y_hat);
                                        sample_grad.extend(x.0.iter().map(|xi| xi * (prediction[i] - y[i] as f64)).collect::<Vec::<f64>>());
                                    }
                                     
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
                            let dim = local_grad.0.len()/self.num_classes;
                            //we have to push every dim elements, since Sample type is a Vec and not Vec<vec<f64>>
                            for i in 0..self.num_classes{
                                state.global_grad.push(local_grad.0.iter().skip(i*dim).take(dim).cloned().collect());
                            }
                            // state.global_grad.extend(local_grad.0)??
                        }
                    },

                    move|state| 
                    {   
                        //initialize
                        if state.epoch==0{
                            let dim = state.global_grad[0].len();
                            state.weights = vec![vec![0.;dim];self.num_classes];
                            state.m = vec![vec![0.;dim];self.num_classes];
                            state.v = vec![vec![0.;dim];self.num_classes];
                        }
                        //update iterations
                        state.epoch +=1;
                        //update the weights (optional with weight decay)
                        for i in 0..self.num_classes{
                            state.m[i] = state.m[i].iter().zip(state.global_grad[i].iter()).map(|(mi, gi)|beta1 * mi + ((1. - beta1) * gi)).collect();
                            state.v[i] =  state.v[i].iter().zip(state.global_grad[i].iter()).map(|(vi, gi)|beta2 * vi + ((1. - beta2) * gi.powi(2))).collect();
                            let m_hat: Vec<f64> = state.m[i].iter().map(|mi|mi/(1. - beta1.powi(state.epoch as i32))).collect();
                            let v_hat: Vec<f64> = state.v[i].iter().map(|vi|vi/(1. - beta2.powi(state.epoch as i32))).collect();
                            let adam: Vec<f64> =  m_hat.iter().zip(v_hat.iter()).map(|(mi,vi)| mi/(vi.sqrt()+1e-6)).collect();
                            //update the weights (optional with weight decay)
                            state.weights[i] = state.weights[i].iter().zip(adam.iter()).map(|(wi,a)| wi - learn_rate*a).collect();
                            if weight_decay==true{
                                state.weights[i] = state.weights[i].iter().map(|wi| wi -  learn_rate * 0.002 * wi).collect();
                            }
                        }
                        //tolerance=gradient's L2 norm for the stop condition
                        //let tol: f64 = adam.iter().map(|v| v*v).sum();
                        //reset the global gradient for the next iteration
                        state.global_grad = Vec::new();
                        //loop condition
                        state.epoch < num_iters //&& tol.sqrt() > 1e-4
                    },

                )
                .collect_vec();

        env.execute();

        if let Some(res) = fit.get() {
            let state = &res[0];
            for i in 0..self.num_classes{
                self.coefficients.push(state.weights[i].clone());
                self.intercept.push(state.weights[i][self.coefficients.len()-1]);
                self.features_coef.push(state.weights[i].iter().take(state.weights[i].len()-1).cloned().collect())
            }
            }


            LogisticRegression {
                coefficients: self.coefficients.clone(),
                features_coef: self.features_coef,
                intercept: self.intercept,
                score: 0.,
                normalization: self.normalization,
                train_mean: train_mean,
                train_std: train_std,
                fitted: self.fitted,
                num_classes: self.num_classes
            }

            }



            "SGD" | _ => 

            {
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
                                    //the target is in the last element of each sample, y one hot encoding
                                    let mut y = vec![0;self.num_classes];
                                    y[x.0[dim-1] as usize] = 1; //assigned before normalization because it's a classification task
                                    if self.normalization==true{
                                        //scale the features and the target
                                        x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                                        };
                                    //switch the target with a 1 for the intercept
                                    x.0[dim-1] = 1.;

                                    let mut current_weights = &state.get().weights;
                                    let vec = vec![vec![0.;dim]; self.num_classes];
                                    if state.get().epoch == 0{
                                        current_weights = &vec;
                                    }

                                    let mut prediction = vec![0.;self.num_classes];
                                    let mut sample_grad = Vec::new();
                                    for i in 0..self.num_classes{
                                        let y_hat:f64 = x.0.iter().zip(current_weights[i].iter()).map(|(xi, wi)| xi * wi).sum();
                                        prediction[i] = sigmoid(y_hat);
                                        sample_grad.extend(x.0.iter().map(|xi| xi * (prediction[i] - y[i] as f64)).collect::<Vec::<f64>>());
                                    }
                                     
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
                        //print!("\nGRAD {:?}\n", local_grad);
                        //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                        if local_grad.0.len()!=0{
                            let dim = local_grad.0.len()/self.num_classes;
                            //we have to push every dim elements, since Sample type is a Vec and not Vec<vec<f64>>
                            for i in 0..self.num_classes{
                                state.global_grad.push(local_grad.0.iter().skip(i*dim).take(dim).cloned().collect());
                            }
                            // state.global_grad.extend(local_grad.0)??
                        }
                        //print!("\nGRAD {:?}\n", state.global_grad);
                    },

                    move|state| 
                    {   
                        //initialize
                        if state.epoch==0{
                            state.weights = vec![vec![0.;state.global_grad[0].len()];self.num_classes];
                        }
                        //update iterations
                        state.epoch +=1;
                        //update the weights (optional with weight decay)
                        for i in 0..self.num_classes{
                            state.weights[i] = state.weights[i].iter().zip(state.global_grad[i].iter()).map(|(wi,g)| wi - learn_rate*g).collect();
                            if weight_decay==true{
                                state.weights[i] = state.weights[i].iter().map(|wi| wi -  learn_rate * 0.002 * wi).collect();
                            }
                        }
                        //tolerance=gradient's L2 norm for the stop condition
                        //let tol: f64 = state.global_grad.iter().map(|v| v*v).sum();
                        //reset the global gradient for the next iteration
                        state.global_grad = Vec::new();
                        //loop condition
                        state.epoch < num_iters //&& tol.sqrt() > 1e-4
                    },

                )
                .collect_vec();

        env.execute();

        if let Some(res) = fit.get() {
            let state = &res[0];
            for i in 0..self.num_classes{
                self.coefficients.push(state.weights[i].clone());
                self.intercept.push(state.weights[i][self.coefficients.len()-1]);
                self.features_coef.push(state.weights[i].iter().take(state.weights[i].len()-1).cloned().collect())
            }
            }


            LogisticRegression {
                coefficients: self.coefficients.clone(),
                features_coef: self.features_coef,
                intercept: self.intercept,
                score: 0.,
                normalization: self.normalization,
                train_mean: train_mean,
                train_std: train_std,
                fitted: self.fitted,
                num_classes: self.num_classes
            }

            }

           
        }     

    }
    }

       
//score takes as input also the target, a dataset with row of length num_features+1
//can be used for evaluating both training and test set
impl LogisticRegression {
    fn score(mut self, path_to_data: &String, config: &EnvironmentConfig) -> f64{

        if self.fitted != true {panic!("Can't compute score before fitting the model!");}

        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');

        let mut env = StreamEnvironment::new(config.clone());

        env.spawn_remote_workers();
    
        let score = env.stream(source)
    
            .map(move |mut x| {
                let dim = x.0.len();   
                let mut y = vec![0;self.num_classes];
                let class = x.0[dim-1] as usize; //class number  
                y[class] = 1; //one hot encoding before normalization because it's a classification task
                //scale the features
                if self.normalization==true{
                    x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                    }               
                x.0[dim-1] = 1.;   
                let mut prediction = vec![0.;self.num_classes];
                for i in 0..self.num_classes{
                    let y_hat:f64 = x.0.iter().zip(self.coefficients[i].iter()).map(|(xi, wi)| xi * wi).sum();
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
        self.score = evaluation_score;
        
        evaluation_score

    }
        }

  
//predict doesn't take as input the target, so a dataset with row of length num_features
impl LogisticRegression {
    fn predict(self, path_to_data: &String, config: &EnvironmentConfig) -> Vec<usize>{

        if self.fitted != true {panic!("Can't compute predictions before fitting the model!");}

        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');

        let mut env = StreamEnvironment::new(config.clone());

        env.spawn_remote_workers();
        

        let prediction = env.stream(source)
    
            .map(move |mut x| {
                let mut highest_prob = 0.;
                let mut predicted_class:usize = 0;
                x.0.push(1.); //push the intercept
                if self.normalization==true{
                        x.0 = x.0.iter().zip(self.train_mean.iter().zip(self.train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                        }  
                for i in 0..self.num_classes {
                    let pred = x.0.iter().zip(self.coefficients[i].iter()).map(|(xi,wi)| xi*wi).sum();
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

    let training_set = "wine_quality.csv".to_string();
    //let training_set = "housing_numeric.csv".to_string();
    //let test_set = "housing_numeric.csv".to_string();
    let data_to_predict = "wine_quality.csv".to_string();

    let start = Instant::now();

    let num_classes = 11;
    let mut model = LogisticRegression::new(num_classes);
    
    let method = "SGD".to_string(); //"ADAM".to_string()
    let num_iters = 100;
    let learn_rate = 1e-2;
    let batch_size = 100;
    let normalize = true;
    let weight_decay = false;

    //return the trained model
    model = model.fit(&training_set, method, num_iters, learn_rate, batch_size, normalize, weight_decay, &config);

    //compute the score over the training set
    let score = model.clone().score(&training_set, &config);

    //let score_test = model.clone().score(&test_set, &config);

    let predictions = model.clone().predict(&data_to_predict, &config);

    let elapsed = start.elapsed();

    //print!("\nCoefficients: {:?}\n", model.features_coef);
    //print!("Intercept: {:?}\n", model.intercept);  

    print!("\nScore: {:?}\n", score);
    //print!("\nTest score: {:?}\n", score_test);

    print!("\nPredictions: {:?}\n", predictions.iter().take(25).cloned().collect::<Vec<usize>>());


    eprintln!("\nElapsed: {elapsed:?}");

    
}