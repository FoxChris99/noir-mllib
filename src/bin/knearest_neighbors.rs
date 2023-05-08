#![allow(unused)]
use noir::prelude::*;
use std::time::Instant;
use noir_ml::sample::Sample;
use std::fs::File;
use std::num::ParseFloatError;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn euclidean_distance(a: &Vec<f64>, b: &Vec<f64>) -> f64{
    a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powi(2)).sum()    
}

#[derive(Clone, Debug)]
struct KNeighborsClassifier {
}

impl KNeighborsClassifier {
    fn new() -> KNeighborsClassifier {KNeighborsClassifier {}}}


impl KNeighborsClassifier {
    fn predict(self, path_to_data: &String, data_to_predict: &String, k: usize, config: &EnvironmentConfig) -> Vec<f64>{

        let file = File::open(path_to_data).unwrap();
        let mut reader = csv::ReaderBuilder::new().has_headers(true).from_reader(file);

        let source = CsvSource::<Vec<f64>>::new(path_to_data).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        for result in reader.records() {

            let row: Vec<f64> = result.unwrap().iter()
            .map(|value| value.parse::<f64>().unwrap())
            .collect();
            
            let predictions = env.stream(source)
            .rich_filter_map({
                let mut kcount = 0;
                let mut kclasses = Vec::<f64>::new();
                let mut kdistances = Vec::<f64>::new();
                let mut max_of_kdistances=-1.;
                let mut max_idx: usize;
                    move |x|{
                        let d = euclidean_distance(&x, &row);
                        
                        //initialization
                        if kclasses.len()<k { 
                            if max_of_kdistances < d{
                                max_of_kdistances = d;
                                max_idx = kclasses.len()-1;
                            }
                            kclasses.push(x[x.len()-1]);
                            kdistances.push(d);                            
                        }
                        else{
                            if d < max_of_kdistances
                        }

                    
                    Some(x)
                }});
            
    
        }
        

        env.execute();
    

        vec![0.;1]
}
}



fn main() {
    let (config, _args) = EnvironmentConfig::from_args();
    let training_set = "forest_fire.csv".to_string();
    let data_to_predict = "forest_fire.csv".to_string();
 
    let model = KNeighborsClassifier::new();

    let k = 5;

    let predictions = model.predict(&training_set, &data_to_predict, k, &config);


    let start_score = Instant::now();
    //compute the score over the training set
    let score = model.score(&training_set, &config);
    let elapsed_score = start_score.elapsed();
    


    print!("\n score: {:?}\n", r2);
    print!("\nPredictions: {:?}\n", predictions.iter().take(5).cloned().collect::<Vec<f64>>());
    eprintln!("\nElapsed pred: {elapsed_pred:?}");
    eprintln!("\nElapsed score: {elapsed_score:?}");

}
