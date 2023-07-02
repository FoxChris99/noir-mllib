#![allow(unused)]
use noir::{prelude::*, config};
use std::time::Instant;
use serde::{Serialize,Deserialize};

use noir_ml::{nn_prelude::*, sample::NNvector, basic_stat::get_moments};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;


#[derive(Clone, Serialize, Deserialize, Default)]
pub struct StateNN<T: LayerTrait> {
    layers: Vec<T>,
    epoch: usize,
    loss: f64,
}


impl StateNN<Dense>{
    pub fn new(layers: &[Dense]) -> Self {
        Self {
            layers: layers.try_into().unwrap(),
            epoch: 0,
            loss: 0.,
        }
    }}

    

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sequential<T: LayerTrait> {
    pub layers: Vec<T>,
    pub optimizer: Optimizer,
    pub loss: Loss,
    pub train_mean: Vec<f64>,
    pub train_std: Vec<f64>
}


impl Sequential<Dense> {
    pub fn new(layers: &[Dense]) -> Self {
        Self {
            layers: layers.try_into().unwrap(),
            optimizer: Optimizer::None,
            loss: Loss::None,
            train_mean: Vec::new(),
            train_std: Vec::new(),
        }
    }

    pub fn summary(&self) {
        let mut total_param = 0;
        let mut res = "\nModel Sequential\n".to_string();
        res.push_str("-------------------------------------------------------------\n");
        res.push_str("Layer (Type)\t\t Output shape\t\t No.of params\n");
        for layer in self.layers.iter() {
            let a = layer.w.len();
            let b = layer.b.len();
            total_param += a + b;
            res.push_str(&format!("{}\t\t\t  (None, {})\t\t  {}\n", layer.typ(), b, a + b));
        }
        res.push_str("-------------------------------------------------------------\n");
        res.push_str(&format!("Total params: {}\n", total_param));
        println!("{}", res);
    }

    pub fn compile(&mut self, optimizer: Optimizer, loss: Loss) {
        self.optimizer = optimizer;
        self.loss = loss;
    }

    pub fn fit(&mut self, x: Array2<f64>, y: Array2<f64>, epochs: usize, verbose: bool) {
        for i in 0..epochs {
            // cache (required for back propagation)
            let mut z_cache = vec![];
            let mut a_cache = vec![];
            let mut z: Array2<f64>;
            let mut a = x.clone();
            a_cache.push(a.clone());

            // forward propagate and cache the results
            for layer in self.layers.iter() {
                (z, a) = layer.forward(a.clone());
                z_cache.push(z.clone());
                a_cache.push(a.clone());
            }
            // cost computation
            let y_hat = a_cache.pop().unwrap();  
            let (loss, mut da) = criteria(y_hat, y.clone(), self.loss.clone());
            
            if verbose {
                println!("Epoch: {}/{} cost computation: {:?}", i, epochs, loss);
            }

            // back propagation
            let mut dw_cache = vec![];
            let mut db_cache = vec![];
            let mut dw: Array2<f64>;
            let mut db: Array2<f64>;

            // loss = da
            for ((layer, z), a) in (self.layers.iter()).rev().zip((z_cache.clone().iter()).rev()).zip((a_cache.clone().iter()).rev()) {
                (dw, db, da) = layer.backward(z.clone(), a.clone(), da);
                dw_cache.insert(0, dw);
                db_cache.insert(0, db);
            }
            

            for ((layer, dw), db) in (self.layers.iter_mut()).zip(dw_cache.clone().iter()).zip(db_cache.clone().iter()) {
                layer.optimize(dw.clone(), db.clone(), self.optimizer.clone());
            }
        }
    }
    

    pub fn parallel_train(&mut self, num_iters: usize, 
        path_to_data: &String, tol: f64, n_iter_no_change:usize, normalization: bool, config: &EnvironmentConfig) 
         {
    
            let loss = self.loss.clone();
            let optimizer = self.optimizer.clone();
            let lr = match optimizer {
                Optimizer::SGD { lr } => {lr},
                _ => {0.} //wip
            };

            if normalization==true{
                (self.train_mean, self.train_std) = get_moments(&config, &path_to_data);
            }
    
            let train_mean = self.train_mean.clone();
            let train_std = self.train_std.clone();
    
            let source = CsvSource::<Vec<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
            let mut env = StreamEnvironment::new(config.clone());
            env.spawn_remote_workers();
            let fit = env.stream(source.clone())
            .map(move |mut x| {
                if normalization==true{
                    //scale the features and the target
                    x = x.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
                }
                let y = Array2::from_elem((1,1), x.pop().unwrap());
                NNvector(vec![(Array2::from_shape_vec((1,x.len()),x).unwrap(),y)])
            })
            .replay(
                num_iters,
                StateNN::new(&self.layers),
    
                move |s, state| 
                {
                    s
                    .rich_filter_map({
                        let mut flag = 0;
                        let mut new_layers = state.get().layers.clone();
                        let mut count = 0;
                        let mut count2 = 0;
    
                        move |mut v|{
                            if state.get().epoch == 0{
                                count+=1;
                                
                                None 
                            }
    
                            else {
                            
                            let mut forward_weights: Vec<(Array2<f64>,Array2<f64>)> = Vec::new();
                            count2+=1;
                            let x = v.0[0].0.clone();
                            let y = v.0[0].1.clone();
    
                                //in the first sample "iteration" of the stream we set the final weights of the last global iteration
                                if flag == 0{
                                    new_layers = state.get().layers.clone();
                                    forward_weights = Vec::new();
                                    flag = 1;
                                }
    
                                let mut z_cache = vec![];
                                let mut a_cache = vec![];
                                let mut z: Array2<f64>;
                                let mut a = x;
    
                                a_cache.push(a.clone());
    
                                for layer in new_layers.iter() {
                                    (z, a) = layer.forward(a.clone());
                                    z_cache.push(z.clone());
                                    a_cache.push(a.clone());
                                }
                                // cost computation
                                let y_hat = a_cache.pop().unwrap();  
                                let (loss, mut da) = criteria(y_hat.clone(), y.clone(), loss.clone());
                                
                                // if state.get().epoch == 999{
                                // print!("y_hat: {:}, Y: {:}\n", y_hat, y);
                                // }

                                // back propagation
                                let mut dw_cache = vec![];
                                let mut db_cache = vec![];
                                let mut dw: Array2<f64>;
                                let mut db: Array2<f64>;
                    
                                // loss = da
                                for ((layer, z), a) in (new_layers.iter()).rev().zip((z_cache.clone().iter()).rev()).zip((a_cache.clone().iter()).rev()) {
                                    (dw, db, da) = layer.backward(z.clone(), a.clone(), da);
                                    dw_cache.insert(0, dw);
                                    db_cache.insert(0, db);
                                }
                                
                    
                                for ((layer, dw), db) in (new_layers.iter_mut()).zip(dw_cache.clone().iter()).zip(db_cache.clone().iter())
                                {
                                        // match optimizer {
                                           //"sgd"
                                            forward_weights.push((layer.w.clone() - lr * dw, layer.b.clone() - lr * db));
                                            // Adam { lr, beta1, beta2, epsilon } =>         
                                }
    
                                if count2==count{
                                    count2 = 0;
                                    flag = 0;
                                }
                                //push loss to compute the global loss each epoch
                                forward_weights.push((Array2::from_elem((1,1), loss),Array2::from_elem((1,1), 0.)));
                                Some(NNvector(forward_weights.clone()))
                            }
                }})
                    //the average of the gradients is computed and forwarded as a single value
                    .group_by_avg(|_x| true, |x| x.clone()).drop_key()
                },
    
                move |local_weights: &mut Vec<(Array2<f64>,Array2<f64>)>, weights| 
                {   
                    if weights.0.len()!=0{
                    *local_weights = weights.0;}
                },
    
                move |state, mut local_weights| 
                {   
                    //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                    if local_weights.len()!=0{
                        state.loss = local_weights.pop().unwrap().0.into_raw_vec()[0];
                        for (i,layer) in local_weights.iter().enumerate(){
                            state.layers[i].w = layer.0.clone();
                            state.layers[i].b = layer.1.clone();
                        }
                        
                    }
                },
    
                move|state| 
                {   
                    //update iterations
                    //print!("Epoch: {:?}, Loss: {:?}\n", state.epoch, state.loss);
                    state.epoch +=1;
                    state.epoch < num_iters
                },
    
            )
            .collect_vec();
    
        env.execute();
    
        let state = fit.get().unwrap()[0].clone();
        self.layers = state.layers;
    }



pub fn parallel_train_sgd(&mut self, method: String, learn_rate: f64, num_iters: usize, 
    path_to_data: &String, tol: f64, n_iter_no_change:usize, normalization: bool, config: &EnvironmentConfig) 
    {

        let loss = self.loss.clone();
        let optimizer = self.optimizer.clone();
        if normalization==true{
            (self.train_mean, self.train_std) = get_moments(&config, &path_to_data);
        }

        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();

        let source = CsvSource::<Vec<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let fit = env.stream(source.clone())
        .map(move |mut x| {
            if normalization==true{
                //scale the features and the target
                x = x.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
            }
            let y = Array2::from_elem((1,1), x.pop().unwrap());
            NNvector(vec![(Array2::from_shape_vec((1,x.len()),x).unwrap(),y)])
        })
        .replay(
            num_iters,
            StateNN::new(&self.layers),

            move |s, state| 
            {
                s
                .rich_filter_map({
                    let mut flag = 0;
                    let mut new_layers = state.get().layers.clone();
                    let mut count = 0;
                    let mut count2 = 0;
                    let mut forward_weights: Vec<(Array2<f64>,Array2<f64>)> = Vec::new();

                    move |mut v|{
                        if state.get().epoch == 0{
                            count+=1;
                            None
                        }

                        else {

                        count2+=1;
                        let x = v.0[0].0.clone();
                        let mut y = v.0[0].1.clone();


                            //in the first sample "iteration" of the stream we set the final weights of the last global iteration
                            if flag == 0{
                                new_layers = state.get().layers.clone();
                                forward_weights = Vec::new();
                                flag = 1;
                            }

                            let mut z_cache = vec![];
                            let mut a_cache = vec![];
                            let mut z: Array2<f64>;
                            let mut a = x;

                            a_cache.push(a.clone());

                            for layer in new_layers.iter() {
                                (z, a) = layer.forward(a.clone());
                                z_cache.push(z.clone());
                                a_cache.push(a.clone());
                            }
                            // cost computation
                            let y_hat = a_cache.pop().unwrap();  
                            let (loss, mut da) = criteria(y_hat.clone(), y.clone(), loss.clone());

                
                            // back propagation
                            let mut dw_cache = vec![];
                            let mut db_cache = vec![];
                            let mut dw: Array2<f64>;
                            let mut db: Array2<f64>;
                
                            // loss = da
                            for ((layer, z), a) in (new_layers.iter()).rev().zip((z_cache.clone().iter()).rev()).zip((a_cache.clone().iter()).rev()) {
                                (dw, db, da) = layer.backward(z.clone(), a.clone(), da);
                                dw_cache.insert(0, dw);
                                db_cache.insert(0, db);
                            }
                            
                
                            for ((layer, dw), db) in (new_layers.iter_mut()).zip(dw_cache.clone().iter()).zip(db_cache.clone().iter()) {
                                layer.optimize(dw.clone(), db.clone(), optimizer.clone());
                                if count2==count{
                                    forward_weights.push((layer.w.clone(),layer.b.clone()));
                                }
                            }
                        

                            if count2==count{
                                count2 = 0;
                                flag = 0;
                                Some(NNvector(forward_weights.clone()))
                            }
                            else{
                                None
                            }
                        }
            }})
                //the average of the gradients is computed and forwarded as a single value
                .group_by_avg(|_x| true, |x| x.clone()).drop_key()//.max_parallelism(1)
            },

            move |local_weights: &mut Vec<(Array2<f64>,Array2<f64>)>, weights| 
            {   
                if weights.0.len()!=0{
                *local_weights = weights.0;}
            },

            move |state, local_weights| 
            {   
                //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                if local_weights.len()!=0{
                    for (i,layer) in local_weights.iter().enumerate(){
                        state.layers[i].w = layer.0.clone();
                        state.layers[i].b = layer.1.clone();
                    }
                    
                }
            },

            move|state| 
            {   
                //update iterations
                state.epoch +=1;
                state.epoch < num_iters
            },

        )
        .collect_vec();

    env.execute();

    let state = fit.get().unwrap()[0].clone();
    self.layers = state.layers;
}



    pub fn predict(&self, path_to_data: &String, normalization: bool, config: &EnvironmentConfig) -> Vec<f64> {

        let source = CsvSource::<Vec<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let layers = self.layers.clone();

        let predictions = env.stream(source.clone())
        .map(move |mut x| {
            // if normalization == true{
            //     x =
            // }
            let mut x = Array2::from_shape_vec((1,x.len()), x).unwrap();
            for layer in layers.iter() {
                (_, x) = layer.forward(x);
            }
            x.into_raw_vec()
        }).collect_vec();
        
        env.execute();
        
        predictions.get().unwrap()[0].clone()
    }


    pub fn compute_loss(&self, path_to_data: &String, normalization: bool, config: &EnvironmentConfig) -> f64 {

        let source = CsvSource::<Vec<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        //let source = CsvSource::<Array2<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let layers = self.layers.clone();
        let loss_type = self.loss.clone();

        let score = env.stream(source.clone())
        .map(move |mut x| {
            // if normalization == true{
            //     x =
            // }
            let y = Array2::from_elem((1,1), x.pop().unwrap());
            let mut x = Array2::from_shape_vec((1,x.len()), x).unwrap();
            for layer in layers.iter() {

                (_, x) = layer.forward(x);
            }
            let (loss, _) = criteria(x.clone(), y.clone(), loss_type.clone());
            //print!("y_hat: {:}, Y: {:}\n", x, y);
            loss
        })
        .group_by_avg(|_|true, |&sample_loss| sample_loss).drop_key().collect_vec();
        
        env.execute();
        
        score.get().unwrap()[0]
}


pub fn score(&self, path_to_data: &String, normalization: bool, config: &EnvironmentConfig) -> f64 {

    let source = CsvSource::<Vec<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
    //let source = CsvSource::<Array2<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
    let mut env = StreamEnvironment::new(config.clone());
    env.spawn_remote_workers();

    let layers = self.layers.clone();
    let loss_type = self.loss.clone();

    let score = env.stream(source.clone())
    .map(move |mut x| {
        // if normalization == true{
        //     x =
        // }
        let class = x[x.len()-1] as usize;
        let y = Array2::from_elem((1,1), x.pop().unwrap());
        let mut x = Array2::from_shape_vec((1,x.len()), x).unwrap();
        for layer in layers.iter() {

            (_, x) = layer.forward(x);
        }
        let (max_index, _) = x.iter().enumerate().max_by_key(|(_, &value)| value.to_bits()).unwrap();
        if max_index == class{
            1.
        }
        else{
            0.
        }
    })
    .group_by_avg(|_|true, |&v| v).drop_key().collect_vec();
    
    env.execute();
    
    score.get().unwrap()[0]
}


}
    // pub fn evaluate(&self, x: Array2<f64>, y: Array2<f64>) -> f64 {
    //         let (loss, _) = criteria(self.predict(x), y, self.loss.clone());
    //         loss
    // }




    // pub fn save(&self, path: &str) {
    //     let encoded: Vec<u8> = bincode::serialize(&self.layers).unwrap();
    //     let mut file = File::create(path).unwrap();
    //     file.write(&encoded).unwrap();
    // }

    // pub fn load(&self, path: &str) -> Sequential<Dense>{
    //     let mut file = File::open(path).unwrap();
    //     let mut decoded = Vec::new();
    //     file.read_to_end(&mut decoded).unwrap();
    //     let model: Sequential<_> = bincode::deserialize(&decoded[..]).unwrap();
    //     println!("model: {:?}", model);
    //     model
    // }










fn main() {
    let (config, _args) = EnvironmentConfig::from_args();
    //let training_set = "data/class_10milion_50features_multiclass.csv".to_string();
    let training_set = "wine_color.csv".to_string();
    let mut model = Sequential::new(&[
        Dense::new(8, 11, Activation::Relu),
        Dense::new(8, 8, Activation::Relu),
        Dense::new(2, 8, Activation::Softmax),
    ]);
    model.summary();
    model.compile(Optimizer::SGD{lr: 0.01}, Loss::NLL);

    let start = Instant::now();

    model.parallel_train(10, &training_set, 1e-4, 5, false, &config);
    
    let elapsed = start.elapsed();

    //let predict = model.predict(&new_set, false, &config);
    
    let loss = model.compute_loss(&training_set, false, &config);
    let score = model.score(&training_set, false, &config);    

    print!("Loss: {:?}\n", loss);
    print!("Score: {:?}\n", score);
    
    print!("\nElapsed fit: {elapsed:?}");
    // print!("\nCoefficients: {:?}\n", model.features_coef);
    // print!("Intercept: {:?}\n", model.intercept);  
    // print!("\nR2 score: {:?}\n", r2);
    // print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<f64>>());
    // eprintln!("\nElapsed fit: {elapsed:?}");
    // eprintln!("\nElapsed score: {elapsed_score:?}"); 
    // eprintln!("\nElapsed pred: {elapsed_pred:?}");     

}