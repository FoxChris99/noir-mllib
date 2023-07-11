#[allow(unused)]
use crate::nn_prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Activation {
    Linear,
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
}


pub fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|z| 1. / (1. + e.powf(-z)))
}

pub fn sigmoid_backward(z: &Array2<f64>) -> Array2<f64> {
    sigmoid(&z) * (1.0 - sigmoid(z))
}

pub fn relu(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|z| if z > 0.0 {z} else {0.0})
}

pub fn relu_backward(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|z| if z > 0.0 {1.0} else {0.0})
}

pub fn tanh(z: &Array2<f64>) -> Array2<f64> {
    z.mapv(|z| z.tanh())
}

pub fn tanh_backward(z: &Array2<f64>) -> Array2<f64> {
    1.-tanh(z)*tanh(z)
}

pub fn softmax(mut z: Array2<f64>) -> Array2<f64> {
    for i in 0..z.shape()[0]{
        let mut sum_exp = 0.;
        for k in 0..z.shape()[1]{
            sum_exp += e.powf(z[[i,k]]);}
        for j in 0..z.shape()[1]{
            z[[i,j]] = e.powf(z[[i,j]])/(sum_exp+1e-8);
        }
    }
    z.clone()
}

pub fn softmax_backward(z: Array2<f64>) -> Array2<f64> {
    let softmax_z = softmax(z);
    softmax_z.clone() * (1.0-softmax_z)
}
