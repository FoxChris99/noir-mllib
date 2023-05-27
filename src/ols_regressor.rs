use noir::prelude::*;
use serde::{Deserialize, Serialize};
use crate::{sample::Sample, matrix_utils::*};


#[derive(Clone, Serialize, Deserialize, Default)]
struct StateOLS {
    epoch: usize,
    weights: Vec<f64>
}


pub fn ols_training(path_to_data: &String, normalization: bool, train_mean: Vec<f64>, train_std: Vec<f64>, config: &EnvironmentConfig) 
    -> Vec<f64> {

        let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let fit = env.stream(source.clone())
        .replay(
            2,
            StateOLS {epoch: 0, weights: Vec::new()},

            move |s, state| 
            {
                s
                .rich_filter_map({
                    let mut local_matrix: Vec<Vec<f64>> = Vec::new();
                    //let mut local_matrix: Vec<f64> = Vec::new();
                    let mut target = Vec::<f64>::new();
                    let mut flag_result = 0;
                    move |mut x|{
                        //first iteration: populate the matrix
                        if state.get().epoch==0{
                            if normalization==true{
                                //scale the features and the target
                                x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                            }
                            let last = x.0.len()-1;
                            target.push(x.0[last]);
                            //switch the target with the intercept
                            x.0[last] = 1.;
                            local_matrix.push(x.0);                               
                            //local_matrix.extend(x.0);  
                            None
                        }
                        //second iteration: compute local weights
                        else{
                            if flag_result==0{
                                flag_result = 1;
                                let local_transpose = transpose(&local_matrix);
                                local_matrix = matrix_product(&local_transpose, &local_matrix);
                                local_matrix = invert_lu(&mut local_matrix);
                                //local_matrix = invert_matrix(&local_matrix);
                                local_matrix = matrix_product(&local_matrix, &local_transpose);
                                let weights_ols = matrix_vector_product(&local_matrix,&target);
                                // let cols = x.0.len();
                                // let rows = target.len();
                                // let mut local_transpose = transpose_mat(&local_matrix, rows, cols);
                                // print!("TRANSPOSE");
                                // local_matrix = vec_product(&local_transpose, &local_matrix);
                                // print!("FIRST PRODUCT");
                                // invert_lu_inplace(&mut local_matrix, cols);
                                // print!("INVERSE");
                                // local_transpose = vec_product(&local_matrix, &local_transpose);
                                // let weights_ols = mat_vec_product(&local_transpose,&target, cols, rows);//cols rows switched
                                Some(Sample(weights_ols))
                            } 
                            else {
                                None
                            } }
                    }})

            },

             |local_weights: &mut Sample, avg_weights| 
            {   
                *local_weights = avg_weights;
            },

             |state, local_grad| 
            {   
                state.weights = local_grad.0;
            },

            |state| 
            {   
                state.epoch+=1;
                state.epoch<2
            },

        )
        .collect_vec();

        env.execute();
        let state = fit.get().unwrap()[0].clone();
        
        state.weights
}


use ndarray::{Array2, Array1};
use linfa::Dataset;
use linfa_linear::LinearRegression;
use linfa::traits::Fit;

pub fn ols_training_array(path_to_data: &String, normalization: bool, train_mean: Vec<f64>, train_std: Vec<f64>, config: &EnvironmentConfig) 
    -> Vec<f64> {

        let source = CsvSource::<Sample>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let fit = env.stream(source.clone())
        .replay(
            2,
            StateOLS {epoch: 0, weights: Vec::new()},

            move |s, state| 
            {
                s
                .rich_filter_map({
                    let mut local_matrix: Array2<f64> = Array2::zeros((1,1));
                    let mut count = 0;
                    let mut count2 = 0;
                    let mut target = Vec::<f64>::new();
                    let mut flag_create_matrix = 0;
                    move |mut x|{
                        //first iteration: count the samples
                        if state.get().epoch==0{
                            count +=1;
                            None
                        }
                        //second iteration: populate matrix and compute local weights
                        else{
                            target.push(x.0.pop().unwrap());                            
                            if flag_create_matrix == 0{
                                flag_create_matrix = 1;
                                local_matrix = Array2::zeros((count, x.0.len()));
                            }
                            if normalization==true{
                                //scale the features and the target
                                x.0 = x.0.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect();
                            }
                            local_matrix.row_mut(count2).assign(&Array1::from(x.0));
                            count2+=1;

                            if count2 == count{
                                let model = LinearRegression::default().fit(&Dataset::new(local_matrix.clone(), Array1::from(target.clone()))).unwrap();
                                let mut weights = model.params().to_vec();
                                weights.push(model.intercept());

                                Some(Sample(weights))
                            } 
                            else {
                                None
                            } }
                    }})

            },

             |local_weights: &mut Sample, avg_weights| 
            {   
                *local_weights = avg_weights;
            },

             |state, local_grad| 
            {   
                state.weights = local_grad.0;
            },

            |state| 
            {   
                state.epoch+=1;
                state.epoch<2
            },

        )
        .collect_vec();

        env.execute();
        let state = fit.get().unwrap()[0].clone();
        
        state.weights
}