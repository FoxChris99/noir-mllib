use noir::prelude::*;
use std::collections::HashSet;
use std::collections::HashMap;
use regex::Regex;

use serde::{Deserialize, Serialize};

use std::time::Instant;

//Struct for messages coming from the source
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Message{
    pub text: String,
    pub class: String,
}

//Struct for messages, after being tokenized into single words
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenizedMessage{
    pub tokens: HashSet<String>,
    pub class: String,
}

//Split a message text into its single words, removing duplicates
pub fn tokenize(text: String) -> HashSet<String> {
    let mut set = HashSet::new();
    let tokens: Vec<String> = Regex::new(r"[A-Za-z0-9']+")
        .unwrap()
        .find_iter(&text)
        .map(|word| word.as_str().to_lowercase())
        .collect();

    for token in tokens.iter(){
        set.insert(token.to_string());
    }
    set
}

//Struct for classifier model, used for training and prediction
#[derive(Default, Debug, Clone)] 
pub struct NaiveBayesClassifier {
    pub alpha: f64,
    pub tokens: HashSet<String>,
    pub token_counts: HashMap<String,HashMap<String, i32>>,
    pub messages_count: HashMap<String, usize>
}

impl NaiveBayesClassifier{
    fn new(alpha: f64)-> NaiveBayesClassifier{
        NaiveBayesClassifier {
            alpha,
            ..Default::default()
        }
    }

    fn token_probs(self, token: String) -> HashMap<String,f64> {
        let mut map: HashMap<String,f64> = HashMap::new();
        for class in self.messages_count.keys() {
            let count: i32;
            match self.token_counts[class].get(&token) {
                Some(val) => {count = *val;}
                None => {count = 0;}
            }
            let prob = (count as f64 + self.alpha) / (self.messages_count[class] as f64 + self.messages_count.keys().len() as f64 * self.alpha);
            map.insert(class.clone(), prob); 
        }
        map
    }
}


fn main(){
    let (config, args) = EnvironmentConfig::from_args();
    let train_path: String;
    let test_path: String;
    let alpha = 1.;

    train_path = args[0].parse().expect("Invalid file path");
    test_path = args[0].parse().expect("Invalid file path");

    let mut classifier = NaiveBayesClassifier::new(alpha);

    //read from csv source
    let source = CsvSource::<Message>::new(train_path).has_headers(true).delimiter(b',');

    //create the environment
    let mut env = StreamEnvironment::new(config.clone());
    env.spawn_remote_workers();

    //count number of messages per class
    let res_msg_counts = env.stream(source.clone())
        .group_by_count(|n| n.class.clone())
        .collect_vec();

    //count amount of tokens per class
    let res_token_counts = env.stream(source.clone())
        .rich_map({
            move |x|{
                TokenizedMessage {tokens: tokenize(x.text), class: x.class}
            }
            }
        )
        .group_by_fold(
            |n| n.class.to_string(),
            HashMap::new(),
            move |set, msg| {
                for token in msg.tokens{
                    match set.get(&token.to_string()){
                        Some(count) => {set.insert(token.to_string(), count+1);}
                        None => {set.insert(token.to_string(), 1);} 
                    }
                }
             },
             move |set, msg| { 
                for (token,val) in msg.iter() {
                    match set.get(&token.to_string()){
                        Some(count) => {set.insert(token.to_string(), count+val);}
                        None => {set.insert(token.to_string(), *val);} 
                    }
                }
             },
        )
        .collect_vec(); 

    let start = Instant::now();
    env.execute();
    let elapsed = start.elapsed();

    //Configure classifier model with resulting data from stream    
    if let Some(res_msg_counts) = res_msg_counts.get(){
        let counts =  &res_msg_counts;
        for (key, val) in counts{
            classifier.messages_count.insert(key.to_string(),*val);
        }
    }

    if let Some(res_token_counts) = res_token_counts.get() {
        let tokens = &res_token_counts;
        for (key, val) in tokens{
            classifier.token_counts.insert(key.to_string(),val.clone());
            for token in val.keys(){
                classifier.tokens.insert(token.to_string());
            }
        }
    }

    eprintln!("{:#?}",classifier);
    eprintln!("Elapsed: {elapsed:?}");     


    let source = CsvSource::<Message>::new(test_path).has_headers(true).delimiter(b',');
    let mut env = StreamEnvironment::new(config);
    env.spawn_remote_workers();

    let res = env.stream(source.clone())
    .rich_map({
        move |x|{
            let model = classifier.clone();
            let mut probs_tot: HashMap<String,f64> = HashMap::new();
            for class in classifier.messages_count.keys(){
               probs_tot.insert(class.to_string(),0.);     
            } 
            let tokenized = TokenizedMessage {tokens: tokenize(x.text), class: x.class};

            for token in tokenized.tokens{
                let probs = classifier.token_probs(token);
                for (key,val) in probs_tot{
                    if let Some(a) = probs.get(&key){
                        probs_tot.insert(key, val+a);
                    }
                }    
            }
            x
             
        }    
        })    
    .collect_vec();


    
    
}