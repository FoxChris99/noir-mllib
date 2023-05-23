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