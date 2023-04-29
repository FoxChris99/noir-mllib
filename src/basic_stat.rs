use noir::prelude::*;
use crate::sample::Sample;

//get mean and std of the columns of a dataset
pub fn get_moments(config: &EnvironmentConfig, path_to_data: &String)->(Vec<f64>, Vec<f64>) {
    
    //read from csv source
    let source = CsvSource::<Sample>::new(path_to_data).has_headers(true).delimiter(b',');
    let mut env = StreamEnvironment::new(config.clone());
    env.spawn_remote_workers();
    //get the mean of all the features + target and the second moment E[x^2]
    let  res = env.stream(source)
    .map( |mut x| 
        {   
            //add the features^2 to get the second moment
            x.0.extend(x.0.iter().map(|xi| xi.powi(2)).collect::<Vec<f64>>());
            x
        })
    .group_by_avg(|_x| true, |x| x.clone()).drop_key().collect_vec();

    env.execute();
    
    let moments = res.get().unwrap()[0].0.clone();

    let dim = moments.len()/2;
    
    let mean: Vec<f64> = moments.iter().take(dim).cloned().collect::<Vec<f64>>();
    let mut std: Vec<f64> = moments.iter().skip(dim).cloned().collect::<Vec<f64>>();

    std = std.iter().zip(mean.iter()).map(|(e2,avg)| (e2-avg.powi(2)).sqrt()).collect();
    
    (mean, std)

}




pub fn sigmoid(v: f64) -> f64{
    if v >= 0.{
        1./(1. + (-v).exp())
    } 
    else{ 
        v.exp()/(1. + v.exp())
    }
}