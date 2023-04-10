use noir::prelude::*;

use serde::{Deserialize, Serialize};

use std::time::Instant;
use std::ops::{AddAssign,Div};


#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Serialize, Deserialize, Default, Debug)]
struct Sample(Vec<f64>);

impl AddAssign for Sample {
    fn add_assign(&mut self, other: Self) {
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");

        for (i, element) in other.0.into_iter().enumerate() {
            self.0[i] += element;
        }
    }
}

impl Div<f64> for Sample {
    type Output = Self;

    fn div(self, other: f64) -> Self::Output {
        
        let mut result = Sample(vec![0.0; self.0.len()]);
        for (i, element) in self.0.into_iter().enumerate() {
            result.0[i] = element / other;
        }

        result
    }
}

fn main() {
    let (config, args) = EnvironmentConfig::from_args();


    let path_to_data: String;
    

    match args.len() {

        1 => {path_to_data = args[0].parse().expect("Invalid file path");}

        _ => panic!("Wrong number of arguments!"),
    }


    //read from csv source
    let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');


    let mut env0 = StreamEnvironment::new(config.clone());
    env0.spawn_remote_workers();
    //get the mean of all the features + target and the second moment E[x^2]
    let  res = env0.stream(source.clone())
    .map( |mut x| 
        {   
            //add the features^2 to get the second moment
            x.0.extend(x.0.iter().map(|xi| xi.powi(2)).collect::<Vec<f64>>());
            x
        })
    .group_by_avg(|_x| true, |x| x.clone()).drop_key().collect_vec();
    
    let start = Instant::now();
    env0.execute();
    let elapsed = start.elapsed();
    
    let mut moments = Vec::<f64>::new();

    if let Some(res) = res.get() {
        moments = res[0].0.clone();}

    let dim = moments.len()/2;
    
    let mean: Vec<f64> = moments.iter().take(dim).cloned().collect::<Vec<f64>>();
    let mut std: Vec<f64> = moments.iter().skip(dim).cloned().collect::<Vec<f64>>();

    std = std.iter().zip(mean.iter()).map(|(e2,avg)| (e2-avg.powi(2)).sqrt()).collect();
    
    print!("\nMean: {:?}\n\nStd: {:?}\n\n", mean, std);
    eprintln!("\nElapsed: {elapsed:?}");

}