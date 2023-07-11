use crate::nn_prelude::*;


pub trait LayerTrait {
    fn new(units: usize, prev: usize, activation: Activation) -> Self;
    
    fn typ(&self) -> String;
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Dense {
    pub w: Array2<f64>,
    pub b: Array2<f64>,
    pub activation: Activation,
    pub m: Array2<f64>,
    pub m_b: Array2<f64>,
    pub v: Array2<f64>,
    pub v_b: Array2<f64>,

    
}

impl LayerTrait for Dense {
    fn new(units: usize, prev: usize, activation: Activation) -> Self {
        Self {
            w: rand_array!(prev, units),
            b: rand_array!(1, units),
            activation,
            //adam parameters
            m: Array2::zeros((prev, units)),
            m_b: Array2::zeros((1, units)),
            v: Array2::zeros((prev, units)),
            v_b: Array2::zeros((1, units)),
        }
    }

    fn typ(&self) -> String {
        "Dense".into()
    }
}

impl Dense {
    pub fn forward(&self, a: Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        // z = a * w + b
        let z = a.dot(&self.w) + self.b.clone();

        // a = g(z)
        use Activation::*;
        let a = match self.activation {
            Linear => z.clone(),
            Relu => relu(&z),
            Sigmoid => sigmoid(&z),
            Tanh => tanh(&z),
            Softmax => softmax(z.clone()),
        };
        
        (z, a)
    }

    pub fn backward(&self, z: &Array2<f64>, a: &Array2<f64>, da: Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // dz = da * g(z)
        use Activation::*;
        let dz = match self.activation {
            Linear => da,
            Relu => da * relu_backward(z),
            Sigmoid => da * sigmoid_backward(z),
            Tanh => da * z.mapv(|z| 1.0 - z.tanh().powf(2.0)),
            Softmax => da,
        };
        
        // dw = dz.a , db = dz , da = w.dz
        let dw = (a.clone().reversed_axes().dot(&dz))/(dz.len() as f64);
        let db = dz.sum_axis(Axis(0)).insert_axis(Axis(0))/(dz.len() as f64);
        let da = dz.dot(&self.w.t())+ 1.0*1e-8;

        (dw, db, da)
    }

}

impl Optimization for Dense {
    fn optimize(&mut self, dw: Array2<f64>, db: Array2<f64>, optimizer: Optimizer, epoch: i32) {
        use Optimizer::*;
        match optimizer {
            SGD { lr } => {
                Zip::from(&mut self.w).and(&dw).for_each(|w, &dw| *w = *w - lr*dw);
                Zip::from(&mut self.b).and(&db).for_each(|b, &db| *b = *b - lr*db);
                // self.w = self.w.clone() - lr * dw;
                // self.b = self.b.clone() - lr * db;
            },
            Adam { lr, beta1, beta2} => {
                Zip::from(&mut self.m).and(&dw).for_each(|m, &dw| *m = *m * beta1 + (1. - beta1) * dw);
                Zip::from(&mut self.v).and(&dw).for_each(|v, &dw| *v = *v * beta2 + (1. - beta2) * dw.powi(2));
                Zip::from(&mut self.m_b).and(&db).for_each(|m, &db| *m = *m * beta1 + (1. - beta1) * db);
                Zip::from(&mut self.v_b).and(&db).for_each(|v, &db| *v = *v * beta2 + (1. - beta2) * db.powi(2));
                let m_hat = &self.m/(1. - beta1.powi(epoch));
                let v_hat = &self.v/(1. - beta2.powi(epoch));
                let mb_hat = &self.m_b/(1. - beta1.powi(epoch));
                let vb_hat = &self.v_b/(1. - beta2.powi(epoch));
                let gradw: Array2<f64> = m_hat / (v_hat.mapv(|v| v.sqrt()+ 1e-8));
                let gradb: Array2<f64> = mb_hat / (vb_hat.mapv(|v| v.sqrt()+ 1e-8));
                Zip::from(&mut self.w).and(&gradw).for_each(|w: &mut f64, &dw| *w -= lr * dw);
                Zip::from(&mut self.b).and(&gradb).for_each(|b: &mut f64, &db| *b -= lr * db);
            },
            None => (),
        }
    }
}