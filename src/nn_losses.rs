use crate::nn_prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Loss {
    MSE,
    CCE,
    None,
}

pub fn criteria(y_hat: &Array2<f64>, y: &Array2<f64>, loss_type: &Loss) -> (f64, Array2<f64>) {
    
    use Loss::*;
    match loss_type {

        MSE => {
            let da = y_hat - y ;
            let loss = 0.5* (da.map(|el| el.powi(2)).mean()).unwrap();
            (loss, da)
        },


        CCE => {
            let y = y.clone().into_raw_vec()[0] as usize;
            let mut one_hot_y = Array2::<f64>::zeros((1, y_hat.dim().1));
            one_hot_y[(0,y)] = 1.;
            
            let da =  y_hat.clone() - one_hot_y.clone();
            //let loss = -(one_hot_y * y_hat.mapv(|v| (v+1e-8).log(e))).sum_axis(Axis(1)).mean().unwrap();
            // in case of more samples to modify
            let loss = -y_hat[(0,y)].ln(); 
            (loss, da)
        },

        None => {
            let da = y_hat.clone() - y.clone();
            let loss = (y_hat - y).mean().unwrap();
            (loss, da)
        },
        
    }
}