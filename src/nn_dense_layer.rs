use crate::prelude::*;

pub trait LayerTrait {
    fn new(units: usize, prev: usize, activation: Activation) -> Self;

    fn typ(&self) -> String;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Dense {
    pub w: Array2<f64>,
    pub b: Array2<f64>,
    pub activation: Activation,
}

impl LayerTrait for Dense {
    fn new(units: usize, prev: usize, activation: Activation) -> Self {
        Self {
            w: rand_array!(prev, units),
            b: rand_array!(1, units),
            activation,
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
            Relu => relu(z.clone()),
            Sigmoid => sigmoid(z.clone()),
            Tanh => tanh(z.clone()),
            Softmax => softmax(z.clone()),
        };

        // returns
        (z, a)
    }

    pub fn backward(
        &self,
        z: Array2<f64>,
        a: Array2<f64>,
        da: Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // dz = da * g(z)
        use Activation::*;
        let dz = match self.activation {
            Linear => da,
            Relu => da * relu_backward(z),
            Sigmoid => da * sigmoid_backward(z),
            Tanh => da * z.mapv(|z| 1.0 - z.tanh().powf(2.0)),
            Softmax => da + 1.0 * 1e-8,
        };

        // dw = dz.a , db = dz , da = w.dz
        let dw = (a.reversed_axes().dot(&dz)) / (dz.len() as f64) + 1.0 * 1e-8;
        let db = dz.clone().sum_axis(Axis(0)).insert_axis(Axis(0)) / (dz.len() as f64) + 1.0 * 1e-8;
        let da = dz.dot(&self.w.t()) + 1.0 * 1e-8;

        (dw, db, da)
    }
}

impl Optimization for Dense {
    fn optimize(&mut self, dw: Array2<f64>, db: Array2<f64>, optimizer: Optimizer) {
        use Optimizer::*;
        match optimizer {
            SGD(lr) => {
                self.w = self.w.clone() - lr * dw;
                self.b = self.b.clone() - lr * db;
            }
            Adam {
                lr,
                beta1,
                beta2,
                epsilon,
            } => {
                unimplemented!(
                    "Adam optimizer not implemented yet lr={}, beta1={}, beta2={}, epsilon={}",
                    lr,
                    beta1,
                    beta2,
                    epsilon
                );
            }
            None => (),
        }
    }
}
