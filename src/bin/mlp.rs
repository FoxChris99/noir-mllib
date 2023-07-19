#![allow(unused)]
use ndarray::ShapeBuilder;
use noir::{prelude::*, config};
use rand::Rng;
use std::time::Instant;
use serde::{Serialize,Deserialize};

use noir_ml::{nn_prelude::*, sample::{NNvector, Sample}, basic_stat::get_moments};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;


#[derive(Clone, Serialize, Deserialize, Default)]
pub struct StateNN<T: LayerTrait> {
    layers: Vec<T>,
    epoch: usize,
    loss: f64,
    best_loss: f64,
    n_iter_early_stopping: usize,
    best_network: Vec<T>
}


impl StateNN<Dense>{
    pub fn new(layers: &[Dense]) -> Self {
        Self {
            layers: layers.try_into().unwrap(),
            epoch: 0,
            loss: 0.,
            best_loss: f64::MAX,
            n_iter_early_stopping: 0,
            best_network: layers.try_into().unwrap()
        }
    }}

    

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sequential<T: LayerTrait> {
    pub layers: Vec<T>,
    pub optimizer: Optimizer,
    pub loss: Loss,
    pub train_mean: Vec<f64>,
    pub train_std: Vec<f64>,
    task: String,
}


impl Sequential<Dense> {
    pub fn new(layers: &[Dense]) -> Self {
        Self {
            layers: layers.try_into().unwrap(),
            optimizer: Optimizer::None,
            loss: Loss::None,
            train_mean: Vec::new(),
            train_std: Vec::new(),
            task: "".to_string(),
        }
    }

    pub fn summary(&self) {
        let mut total_param = 0;
        let mut res = "\nModel Sequential\n".to_string();
        res.push_str("_____________________________________________________________\n\n");
        res.push_str("Layer (Type)\t\t Output Shape\t\t Param #\n");
        res.push_str("=============================================================\n");
        res.push_str(&format!("Input\t\t\t  (None, {})\t\t  {}\n", self.layers[0].w.dim().0, 0));
        for layer in self.layers.iter() {
            let a = layer.w.len();
            let b = layer.b.len();
            total_param += a + b;
            res.push_str(&format!("{}\t\t\t  (None, {})\t\t  {}\n", layer.typ(), b, a + b));
        }
        res.push_str("=============================================================\n");
        res.push_str(&format!("Total params: {}\n", total_param));
        res.push_str("_____________________________________________________________\n");
        println!("{}", res);
    }

    pub fn compile(&mut self, optimizer: Optimizer, loss: Loss) {
        self.optimizer = optimizer;
        self.loss = loss;
        match self.loss {
            Loss::CCE => {
                self.task = "classification".to_string();
            },
            _ =>{
                self.task = "regression".to_string();
            }
        
    }}


    pub fn fit(&mut self, num_iters: usize, 
        path_to_data: &String, tol: f64, n_iter_no_change:usize, normalization: bool, verbose: bool, config: &EnvironmentConfig) 
         {
    
            let loss = self.loss.clone();
            let optimizer = self.optimizer.clone();

            if normalization==true{
                (self.train_mean, self.train_std) = get_moments(&config, &path_to_data);
                match loss {
                    Loss::CCE => {
                        self.train_mean = self.train_mean.iter().cloned().take(self.train_mean.len()-1).collect();
                        self.train_std = self.train_std.iter().cloned().take(self.train_std.len()-1).collect();
                        self.task = "classification".to_string();
                    },
                    _ =>{
                        self.task = "regression".to_string();
                    }
            }}

            else{
                match loss {
                    Loss::CCE => {
                        self.task = "classification".to_string();
                    },
                    _ =>{
                        self.task = "regression".to_string();
                    }
                }
            }
            
            let task = self.task.clone();
            let train_mean = self.train_mean.clone();
            let train_std = self.train_std.clone();
    
            let source = CsvSource::<Vec<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
            let mut env = StreamEnvironment::new(config.clone());
            env.spawn_remote_workers();
            let fit = env.stream(source.clone())
            .map(move |mut x| {
                let mut y = Array2::from_elem((1,1), 0.);
                if normalization==true{
                    if task == "classification".to_string(){
                        //first pop the target class
                        y = Array2::from_elem((1,1), x.pop().unwrap());
                        x = x.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
                    }
                    else {
                        x = x.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
                        y = Array2::from_elem((1,1), x.pop().unwrap());}

                    }
                else{
                    y = Array2::from_elem((1,1), x.pop().unwrap());
                }
                NNvector(vec![(Array2::from_shape_vec((1,x.len()),x).unwrap(),y)])
            })
            .replay(
                num_iters + 1,
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
                            
                            count2+=1;
                            let mut forward_weights: Vec<(Array2<f64>,Array2<f64>)> = Vec::new();
                            let mut x = &v.0[0].0;
                            let mut y = &v.0[0].1;

    
                                //in the first sample "iteration" of the stream we set the final weights of the last global iteration
                                if flag == 0{
                                    new_layers = state.get().layers.clone();
                                    forward_weights = Vec::new();
                                    flag = 1;
                                }
    
                                let mut z_cache = vec![];
                                let mut a_cache = vec![];
                                let mut z: Array2<f64>;
                                let mut a = x.clone();
    
                                a_cache.push(a.clone());
    
                                for layer in new_layers.iter() {
                                    (z, a) = layer.forward(a);
                                    z_cache.push(z.clone());
                                    a_cache.push(a.clone());
                                }
                                // loss computation
                                let y_hat = a_cache.pop().unwrap();  

                                let (loss, mut da) = criteria(&y_hat, &y, &loss);
                                
                                // back propagation
                                let mut dw: Array2<f64>;
                                let mut db: Array2<f64>;
                    
                                for ((layer, z), a) in (new_layers.iter()).rev().zip((z_cache.iter()).rev()).zip((a_cache.iter()).rev()) {
                                    (dw, db, da) = layer.backward(z, a, da);
                                    forward_weights.insert(0,(dw.clone(), db.clone()));
                                }                               
    
                                if count2==count{
                                    count2 = 0;
                                    flag = 0;
                                }

                                //push loss to compute the global loss each epoch and a 0 to make the tuple
                                forward_weights.push((Array2::from_elem((1,1), loss),Array2::from_elem((1,1), 0.)));
                                Some(NNvector(forward_weights.clone()))
                            }
                            }
                })
                    //the average of the gradients is computed and forwarded as a single value
                    .group_by_avg(|_x| true, |x| x.clone()).drop_key()
                },
    
                move |local_dw: &mut Vec<(Array2<f64>,Array2<f64>)>, dweights| 
                {   
                    if dweights.0.len()!=0{
                    *local_dw = dweights.0;}
                },
    
                move |state, mut local_dw| 
                {   
                    //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                    if local_dw.len()!=0{
                        state.loss = local_dw.pop().unwrap().0.into_raw_vec()[0];
                        
                        if state.best_loss>state.loss{
                            state.best_network = state.layers.clone();
                        }

                        match optimizer {
                            Optimizer::SGD { lr } => {
                                for (i,layer) in local_dw.iter().enumerate(){
                                    Zip::from(&mut state.layers[i].w).and(&layer.0).for_each(|w, &dw| *w -= lr * dw);
                                    Zip::from(&mut state.layers[i].b).and(&layer.1).for_each(|b, &db| *b -= lr * db);
                                }
                            }
                            Optimizer::Adam { lr, beta1, beta2 } => {
                                for (i,layer) in local_dw.iter().enumerate(){
                                    Zip::from(&mut state.layers[i].m).and(&layer.0).for_each(|m, &dw| *m = *m * beta1 + (1. - beta1) * dw);
                                    Zip::from(&mut state.layers[i].v).and(&layer.0).for_each(|v, &dw| *v = *v * beta2 + (1. - beta2) * dw.powi(2));
                                    Zip::from(&mut state.layers[i].m_b).and(&layer.1).for_each(|m, &db| *m = *m * beta1 + (1. - beta1) * db);
                                    Zip::from(&mut state.layers[i].v_b).and(&layer.1).for_each(|v, &db| *v = *v * beta2 + (1. - beta2) * db.powi(2));
                                    let m_hat = &state.layers[i].m/(1. - beta1.powi(state.epoch as i32));
                                    let v_hat = &state.layers[i].v/(1. - beta2.powi(state.epoch as i32));
                                    let mb_hat = &state.layers[i].m_b/(1. - beta1.powi(state.epoch as i32));
                                    let vb_hat = &state.layers[i].v_b/(1. - beta2.powi(state.epoch as i32));
                                    let gradw: Array2<f64> = m_hat / (v_hat.mapv(|v| v.sqrt()+ 1e-8));
                                    let gradb: Array2<f64> = mb_hat / (vb_hat.mapv(|v| v.sqrt()+ 1e-8));
                                    Zip::from(&mut state.layers[i].w).and(&gradw).for_each(|w: &mut f64, &dw| *w -= lr * dw);
                                    Zip::from(&mut state.layers[i].b).and(&gradb).for_each(|b: &mut f64, &db| *b -= lr * db);
                                }
                            }
                            Optimizer::None => (),

                        }   
                    }
                },
    
                move|state| 
                {   
                    //update iterations
                    if verbose && state.epoch>0{
                        print!("Iter: {:?}/{:?} --- Loss: {:?}\n", state.epoch, num_iters, state.loss);}


                    if state.epoch != 0
                    {    
                    //early stopping if for n iters the loss doesn't improve
                    if state.loss > state.best_loss - tol && tol!=0.{
                        state.n_iter_early_stopping+=1;
                        
                    }
                    else{
                        state.n_iter_early_stopping=0;
                    }

                    if state.best_loss>state.loss{
                        state.best_loss = state.loss;
                    }

                    if state.n_iter_early_stopping >= n_iter_no_change {
                            print!("\nEarly Stopping at iter: {:?}\n", state.epoch);
                    }
                    }
                    state.epoch +=1;
                    state.epoch < num_iters + 1 && state.n_iter_early_stopping < n_iter_no_change
                },
    
            )
            .collect_vec();
    
        env.execute();
    
        let state = fit.get().unwrap()[0].clone();

        self.layers = state.best_network;

    }



    pub fn fit2(&mut self, num_iters: usize, 
        path_to_data: &String, tol: f64, n_iter_no_change:usize, normalization: bool, verbose: bool, config: &EnvironmentConfig) 
         {
    
            let loss = self.loss.clone();
            let optimizer = self.optimizer.clone();

            if normalization==true{
                (self.train_mean, self.train_std) = get_moments(&config, &path_to_data);
                match loss {
                    Loss::CCE => {
                        self.train_mean = self.train_mean.iter().cloned().take(self.train_mean.len()-1).collect();
                        self.train_std = self.train_std.iter().cloned().take(self.train_std.len()-1).collect();
                        self.task = "classification".to_string();
                    },
                    _ =>{
                        self.task = "regression".to_string();
                    }
            }}

            else{
                match loss {
                    Loss::CCE => {
                        self.task = "classification".to_string();
                    },
                    _ =>{
                        self.task = "regression".to_string();
                    }
                }
            }
            
            let task = self.task.clone();
            let train_mean = self.train_mean.clone();
            let train_std = self.train_std.clone();
    
            let source = CsvSource::<Vec<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
            let mut env = StreamEnvironment::new(config.clone());
            env.spawn_remote_workers();
            let fit = env.stream(source.clone())
            .map(move |mut x| {
                let mut y = Array2::from_elem((1,1), 0.);
                if normalization==true{
                    if task == "classification".to_string(){
                        //first pop the target class
                        y = Array2::from_elem((1,1), x.pop().unwrap());
                        x = x.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
                    }
                    else {
                        x = x.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
                        y = Array2::from_elem((1,1), x.pop().unwrap());}

                    }
                else{
                    y = Array2::from_elem((1,1), x.pop().unwrap());
                }
                NNvector(vec![(Array2::from_shape_vec((1,x.len()),x).unwrap(),y)])
            })
            .replay(
                num_iters + 1,
                StateNN::new(&self.layers),
    
                move |s, state| 
                {
                    s
                    .shuffle().rich_filter_map({
                        let mut curr_loss = 0.;
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
                            
                            count2+=1;
                            let mut forward_weights: Vec<(Array2<f64>,Array2<f64>)> = Vec::new();
                            let mut x = &v.0[0].0;
                            let mut y = &v.0[0].1;

    
                                //in the first sample "iteration" of the stream we set the final weights of the last global iteration
                                if flag == 0{
                                    new_layers = state.get().layers.clone();
                                    forward_weights = Vec::new();
                                    flag = 1;
                                    curr_loss = 0.;
                                }
    
                                let mut z_cache = vec![];
                                let mut a_cache = vec![];
                                let mut z: Array2<f64>;
                                let mut a = x.clone();
    
                                a_cache.push(a.clone());
    
                                for layer in new_layers.iter() {
                                    (z, a) = layer.forward(a);
                                    z_cache.push(z.clone());
                                    a_cache.push(a.clone());
                                }
                                // loss computation
                                let y_hat = a_cache.pop().unwrap();  

                                let (_, mut da) = criteria(&y_hat, &y, &loss);

                                //compute loss of the previous epoch
                                if state.get().epoch > 0{
                                    let mut output: Array2<f64> = x.clone();
                                    for layer in state.get().layers.iter() {
                                        (_, output) = layer.forward(output);
                                    }

                                    let (sample_loss, _) = criteria(&output, &y, &loss);
                                    curr_loss+=sample_loss;

                                }
                                
                                // back propagation
                                let mut dw: Array2<f64>;
                                let mut db: Array2<f64>;      
                                    

                                for ((layer, z), a) in 
                                (new_layers.iter_mut()).rev().zip((z_cache.iter()).rev()).zip((a_cache.iter()).rev()) 
                                {
                                    (dw, db, da) = layer.backward(z, a, da);

                                    layer.optimize(dw, db, optimizer.clone(), state.get().epoch as i32);

                                    if count2==count{
                                        forward_weights.insert(0,(layer.w.clone(), layer.b.clone()));
                                    } 
                                }   
                                     
     
                                if count2==count{
                                    //push loss to compute the global loss each epoch and a 0 to make the tuple
                                    forward_weights.push((Array2::from_elem((1,1), curr_loss/count2 as f64),Array2::from_elem((1,1), 0.)));

                                    count2 = 0;
                                    flag = 0;

                                    Some(NNvector(forward_weights.clone()))
                                }

                                else{
                                    None
                                }

                            }
                            }
                })
                    //the average of the gradients is computed and forwarded as a single value
                    .group_by_avg(|_x| true, |x| x.clone()).drop_key()
                },
    
                move |local_w: &mut Vec<(Array2<f64>,Array2<f64>)>, weights| 
                {   
                    if weights.0.len()!=0{
                    *local_w = weights.0;}
                },
    
                move |state, mut local_w| 
                {   
                    //we don't want to read empty replica gradient (this should be solved by using the max_parallelism(1) above)
                    if local_w.len()!=0{
                        state.loss = local_w.pop().unwrap().0.into_raw_vec()[0];

                        for (i,layer) in local_w.iter().enumerate(){
                            state.layers[i].w = layer.0.clone();
                            state.layers[i].b = layer.1.clone();
                        }
                        
                        if state.best_loss>state.loss{
                            state.best_network = state.layers.clone();
                        }

                    }
                },
    
                move|state| 
                {   
                    //update iterations
                    if verbose && state.epoch>0{
                        print!("Iter: {:?}/{:?} --- Loss: {:?}\n", state.epoch, num_iters, state.loss);}


                    if state.epoch != 0
                    {    
                    //early stopping if for n iters the loss doesn't improve
                    if state.loss > state.best_loss - tol && tol!=0.{
                        state.n_iter_early_stopping+=1;
                        
                    }
                    else{
                        state.n_iter_early_stopping=0;
                    }

                    if state.best_loss>state.loss{
                        state.best_loss = state.loss;
                    }

                    if state.n_iter_early_stopping >= n_iter_no_change {
                            print!("\nEarly Stopping at iter {:?} with loss: {:}\n", state.epoch, state.best_loss);
                    }
                    }
                    state.epoch +=1;
                    state.epoch < num_iters + 1 && state.n_iter_early_stopping < n_iter_no_change
                },
    
            )
            .collect_vec();
    
        env.execute();
    
        let state = fit.get().unwrap()[0].clone();

        self.layers = state.best_network;

    }



    pub fn predict(&self, path_to_data: &String, normalization: bool, config: &EnvironmentConfig) -> Vec<f64> {
        let source = CsvSource::<Vec<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let layers = self.layers.clone();
        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        let task = self.task.clone();

        let predictions = env.stream(source.clone())
        .map(move |mut x| {
            //////////////////////////////////////
            //SOLO PER TESTARE
            if layers[0].w.dim().1 == x.len()-1{
                x.pop();
            }
            else{
                x.pop();
            }

            //////////////////////////////////////
            /// 
            if normalization==true{
                //scale the features and the target
                x = x.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
            }

            let mut x = Array2::from_shape_vec((1,x.len()), x).unwrap();
            

            for layer in layers.iter() {
                (_, x) = layer.forward(x);
            }

            if task == "classification".to_string(){
                //find argmax to get the class index
                let (max_index, _) = x.iter().enumerate()
                .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap()).unwrap();

                max_index as f64
            }

            else{
                x.into_raw_vec()[0]
            }
            
        }).collect_vec();
        
        env.execute();
        
        predictions.get().unwrap()
    }


    pub fn compute_loss(&self, path_to_data: &String, normalization: bool, config: &EnvironmentConfig) -> f64 {

        let source = CsvSource::<Vec<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        //let source = CsvSource::<Array2<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let layers = self.layers.clone();
        let loss_type = self.loss.clone();
        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        let task = self.task.clone();
        let score = env.stream(source.clone())
        .map(move |mut x| {
            let mut y = Array2::zeros((1,1));
            if task == "classification".to_string(){
                y = Array2::from_elem((1,1), x.pop().unwrap());
            }
            if normalization==true{
                //scale the features and the target
                x = x.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
            }
            if task == "regression".to_string(){
                y = Array2::from_elem((1,1), x.pop().unwrap());
            }
            let mut x = Array2::from_shape_vec((1,x.len()), x).unwrap();
            

            for layer in layers.iter() {

                (_, x) = layer.forward(x);
            }

            let (loss, _) = criteria(&x, &y, &loss_type);

            loss
        })
        .group_by_avg(|_|true, |&sample_loss| sample_loss).drop_key().collect_vec();
        
        env.execute();
        
        score.get().unwrap()[0]
}


pub fn score(&self, path_to_data: &String, normalization: bool, config: &EnvironmentConfig) -> f64 {


    if self.task == "classification".to_string(){

        let source = CsvSource::<Vec<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        //let source = CsvSource::<Array2<f64>>::new(path_to_data.clone()).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let layers = self.layers.clone();
        let loss_type = self.loss.clone();
        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        let score = env.stream(source.clone())
        .map(move |mut x| {
            let class = x[x.len()-1] as usize;
            let y = Array2::from_elem((1,1), x.pop().unwrap());  
            if normalization==true{
                //scale the features and the target
                x = x.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
            }
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

    //Compute R2 for regression task
    else{
        let mut avg_y = 0.;
        let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        let res = env.stream(source.clone())
        .group_by_avg(|_x| true, move|x| x.0[x.0.len()-1]).drop_key().collect_vec();
        env.execute();
        avg_y = res.get().unwrap()[0];
    
        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();
        
        let train_mean = self.train_mean.clone();
        let train_std = self.train_std.clone();
        let layers = self.layers.clone();
        let task = self.task.clone();
        //compute the residuals sums for the R2
        let score = env.stream(source)
                
                .map(move |mut x| {
                let mut mean_y = 0.;
                if normalization==true{
                    //scale the features and the target
                    x = x.iter().zip(train_mean.iter().zip(train_std.iter())).map(|(xi,(m,s))| (xi-m)/s).collect::<Vec::<f64>>();
                }
                else{
                    mean_y = avg_y;
                }
                    
                let y = x.pop().unwrap();

                let mut x = Array2::from_shape_vec((1,x.len()), x).unwrap();
                
                // print!("\n{:},\n", x);
                for layer in layers.iter() {
    
                    (_, x) = layer.forward(x);
                }
              
                // print!("\n{:},\n", x);
                // print!("\n{:?},\n", avg_y);
                // print!("\n{:?},\n", (y-avg_y).powi(2));
                // print!("\n{:?},\n", (y-x.clone().into_raw_vec()[0]).powi(2));
                [(y-x.into_raw_vec()[0]).powi(2),(y-mean_y).powi(2)]           
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

}


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
}









fn main() {
    let (config, _args) = EnvironmentConfig::from_args();
    //let training_set = "data/class_10milion_50features_multiclass.csv".to_string();
    //let training_set = "data/class_1milion_4features_multiclass.csv".to_string();
    //let training_set = "diabetes.csv".to_string();
    //let training_set = "housing_numeric.csv".to_string();
    //let training_set = "forest_fire.csv".to_string();
    let training_set = "class_100k_10features_classification.csv".to_string();
    //let training_set: String = "data/class_1milion_4features_multiclass.csv".to_string();
    let mut model = Sequential::new(&[
        Dense::new(32, 10, Activation::Relu),
        Dense::new(32, 32, Activation::Relu),
        Dense::new(32, 32, Activation::Relu),
        Dense::new(8, 32, Activation::Softmax),
    ]);
    model.summary();
    //model.compile(Optimizer::SGD{lr: 0.01}, Loss::MSE);
    model.compile(Optimizer::Adam { lr: 0.01, beta1: 0.9, beta2: 0.999}, Loss::CCE);

    let start = Instant::now();

    model.fit(1000, &training_set, 0.,50, false, true, &config);
    //model.fit2(1000, &training_set, 1e-4 ,50, true, true, &config);
   
    let elapsed = start.elapsed();

    let predict = model.predict(&training_set, false, &config);

    let loss = model.compute_loss(&training_set, false, &config);

    let score = model.score(&training_set, false, &config);  

    print!("Loss: {:?}\n", loss);
    print!("Score: {:?}\n", score);
    
    print!("\nElapsed fit: {elapsed:?}");
    // print!("\nCoefficients: {:?}\n", model.features_coef);
    // print!("Intercept: {:?}\n", model.intercept);  
    // print!("\nR2 score: {:?}\n", r2);
    //print!("\nPredictions: {:?}\n", predict.iter().take(5).cloned().collect::<Vec<f64>>());
    // eprintln!("\nElapsed fit: {elapsed:?}");
    // eprintln!("\nElapsed score: {elapsed_score:?}"); 
    // eprintln!("\nElapsed pred: {elapsed_pred:?}");     

}