use crate::nn_prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Loss {
    MSE,
    CCE,
    None,
}

pub fn criteria(y_hat: Array2<f64>, y: Array2<f64>, loss_ty: Loss) -> (f64, Array2<f64>) {
    
    use Loss::*;
    match loss_ty {

        MSE => {
            let da = y_hat - y ;
            //let loss = (0.5 * (y_hat - y).mapv(|v| v.powf(2.0))).mean().unwrap();
            let loss = 0.5* (da.map(|el| el.powi(2)).mean()).unwrap();
            (loss, da)
        },

        // NLL => {
        //     let da = -((y.clone() / y_hat.clone())-((1.0-y.clone())/(1.0-y_hat.clone())));
        //     let loss = -(y.clone() * y_hat.mapv(|y| y.log(e)).reversed_axes() + (1.0 - y)*(1.0 - y_hat.mapv(|y| y.log(e)).reversed_axes())).mean().unwrap();
        //     (loss, da)
        // },

        CCE => {
            let y = y.into_raw_vec()[0] as usize;
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