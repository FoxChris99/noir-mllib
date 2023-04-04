use noir::prelude::*;
use std::collections::HashSet;
use std::collections::HashMap;
use regex::Regex;

use serde::{Deserialize, Serialize};

use std::time::Instant;


#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Message{
    pub text: String,
    pub class: String,
}

/*#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenizedMessage<'a>{
    pub tokens: HashSet<&'a str>,
    pub class: String,
}*/

pub fn tokenize(text: String) -> HashSet<String> {
    let mut set = HashSet::new();
    let tokens: Vec<String> = Regex::new(r"[A-Za-z0-9']+")
        .unwrap()
        .find_iter(&text)
        .map(|mat| mat.as_str().to_lowercase())
        .collect();

    for token in tokens.iter(){
        set.insert(token.to_string());
    }
    set
}

pub struct NaiveBayesClassifier {
    pub alpha: f64,
    pub tokens: HashSet<String>,
    pub token_counts: Vec<HashMap<String, i32>>,
    pub messages_count: HashMap<String, i32>
}

impl NaiveBayesClassifier{
    fn new(alpha: f64, num_classes: usize)-> NaiveBayesClassifier{
        NaiveBayesClassifier {
            alpha,
            tokens: Default::default(),
            token_counts: vec![HashMap::new(); num_classes],
            messages_count: Default::default()
        }
    }
}

fn main(){
    let (config, args) = EnvironmentConfig::from_args();
    let path: String;
    let num_classes: usize;
    let alpha = 1.;

    path = args[0].parse().expect("Invalid file path");
    num_classes = args[1].parse().expect("Invalid number of features");

    let words = String::from("Ciao Cacca Ciao Blu Bello Ciao");
    println!("{:?}",tokenize(words));

    let mut classifier = NaiveBayesClassifier::new(alpha, num_classes);

    //read from csv source
    let source = CsvSource::<Message>::new(path).has_headers(true).delimiter(b',');

    //create the environment
    let mut env = StreamEnvironment::new(config);
    env.spawn_remote_workers();

    let res = env.stream(source.clone())
        .group_by_count(|n| n.class.clone())
        .collect_vec(); 

    let res = env.stream(source)
        .group_by_fold(
            |n| n.class.clone(),
            HashSet::new(),
            |update, m| {for token in tokenize(m.text).iter() {update.insert(token);}},
            //|update, m| update = update.union(&HashSet::new()/*&tokenize(m.text)*/).collect(),
            |update, m| update.union(&m),
        ).collect_vec(); 
  

    let start = Instant::now();
    env.execute();
    let elapsed = start.elapsed();

    

    if let Some(res) = res.get() {
        let state = &res;
        eprintln!("{:?}",state); 
    }
    eprintln!("Elapsed: {elapsed:?}");     

}