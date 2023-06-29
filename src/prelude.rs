pub use serde::{Deserialize, Serialize};
pub use std::fs::File;
pub use std::io::Read;
pub use std::io::Write;

pub use std::f64::consts::E as e;

pub use ndarray::prelude::*;
pub use ndarray::*;
pub use ndarray_rand::rand_distr::Uniform;
pub use ndarray_rand::RandomExt;

pub use crate::nn_dense_layer::*;
//pub use crate::models::*;
pub use crate::nn_activations::*;
pub use crate::nn_losses::*;
pub use crate::nn_optimizers::*;
pub use crate::nn_utils::*;

pub use crate::rand_array;
pub use crate::Dense;
pub use crate::Model;
