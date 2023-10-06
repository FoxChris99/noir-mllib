use noir::prelude::*;
use std::collections::HashSet;
use std::collections::HashMap;
use std::sync::Arc;
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
pub fn tokenize(text: &String) -> HashSet<String> {
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
    pub total_samples: usize,
    pub tokens: HashSet<String>,
    pub message_counts: HashMap<String, usize>,
    pub token_counts: HashMap<String,HashMap<String, i32>>    
}

impl NaiveBayesClassifier{
    fn new(alpha: f64)-> NaiveBayesClassifier{
        NaiveBayesClassifier {
            alpha,
            total_samples: 0,
            ..Default::default()
        }
    }

    fn fit(&mut self, path: String, config: &EnvironmentConfig){
        //read from csv source
        let source = CsvSource::<Message>::new(path).has_headers(true).delimiter(b',');

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
                    TokenizedMessage {tokens: tokenize(&x.text), class: x.class}
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


        //Configure classifier model with resulting data from stream    
        if let Some(res_msg_counts) = res_msg_counts.get(){
            let counts =  &res_msg_counts;
            for (key, val) in counts{
                self.message_counts.insert(key.to_string(),*val);
                self.total_samples += *val;
            }
        }

        if let Some(res_token_counts) = res_token_counts.get() {
            let tokens = &res_token_counts;
            for (key, val) in tokens{
                self.token_counts.insert(key.to_string(),val.clone());
                for token in val.keys(){
                    self.tokens.insert(token.to_string());
                }
            }
        }

        let elapsed = start.elapsed();
        eprintln!("Training time: {elapsed:?}");               
    }


    fn predict(&mut self, path: String, config: &EnvironmentConfig){
        let source = CsvSource::<Message>::new(path).has_headers(true).delimiter(b',');
        let mut env = StreamEnvironment::new(config.clone());
        env.spawn_remote_workers();

        let classifier = Arc::new(self.clone());
        let res = env.stream(source.clone())
        .rich_map({
            move |x|{
                let mut best_class: String="".to_string();
                let mut max: f64=0.0;

                //Initialize class probabilities
                let mut probs_tot: HashMap<String,f64> = HashMap::new();
                for class in classifier.message_counts.keys(){
                    probs_tot.insert(class.to_string(),0.);     
                } 
                let tokenized = TokenizedMessage {tokens: tokenize(&x.text), class: x.class};

                //Compute probabilities of all tokens in the message
                for token in tokenized.tokens{
                    let mut token_probs: HashMap<String,f64> = HashMap::new();
                    for class in classifier.message_counts.keys() {
                        let count: i32;
                        match classifier.token_counts[class].get(&token) {
                            Some(val) => {count = *val;}
                            None => {count = 0;}
                        }
                        let numerator = count as f64 + classifier.alpha;
                        let denominator = (classifier.message_counts[class] as f64/classifier.total_samples as f64) + classifier.message_counts.keys().len() as f64 * classifier.alpha;
                        token_probs.insert(class.clone(), numerator / denominator); 
                    }

                    //Update class probability
                    for (key,val) in probs_tot.clone(){
                        if let Some(a) = token_probs.get(&key){
                            probs_tot.insert(key.to_string(), val+a.ln());
                        }
                    }    
                }

                //Find most probable class
                for(key,val) in probs_tot.clone(){
                    if val.exp()>max {
                        best_class = key;
                        max = val.exp();
                    }
                }

                let mut res = 0.;
                if best_class == tokenized.class{
                    res = 1.;
                }
                res
            }    
            })
        .group_by_avg(|&_k| true, |&n| n as f64)
        .drop_key()      
        .collect_vec();

        let start = Instant::now();
        env.execute();
        let elapsed = start.elapsed();

        if let Some(res) = res.get(){
            eprintln!("{:#?}",res);  
        }
        eprintln!("Prediction time: {elapsed:?}");  

    }
}


fn main(){
    let (config, args) = EnvironmentConfig::from_args();
    let train_path: String;
    let test_path: String;
    let alpha = 1.;

    train_path = args[0].parse().expect("Invalid file path");
    test_path = args[1].parse().expect("Invalid file path");

    let mut classifier = NaiveBayesClassifier::new(alpha);

   classifier.fit(train_path, &config);
   classifier.predict(test_path, &config);
}