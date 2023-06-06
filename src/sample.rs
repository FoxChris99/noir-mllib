use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::ops::{AddAssign,Div,Sub,Mul,Add, SubAssign, DivAssign, MulAssign};


//create our custom type of vector which represents a sample of the dataset
#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub struct Sample(pub Vec<f64>);

//implement standard operations for our sample vector

impl AddAssign for Sample {
    fn add_assign(&mut self, other: Self) {
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");
        for (a, b) in self.0.iter_mut().zip(other.0.into_iter()) {
            *a += b;
        }
    }
}

impl SubAssign for Sample {
    fn sub_assign(&mut self, other: Self) {
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");
        for (a, b) in self.0.iter_mut().zip(other.0.into_iter()) {
            *a -= b;
        }
    }
}

impl MulAssign for Sample {
    fn mul_assign(&mut self, other: Self) {
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");
        for (a, b) in self.0.iter_mut().zip(other.0.into_iter()) {
            *a *= b;
        }
    }
}

impl DivAssign for Sample {
    fn div_assign(&mut self, other: Self) {
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");
        for (a, b) in self.0.iter_mut().zip(other.0.into_iter()) {
            *a /= b;
        }
    }
}

impl Sub for Sample {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");
        Sample(self.0.iter().zip(other.0.iter()).map(|(a, b)| a - b).collect())
    }
}

impl Sub<f64> for Sample {
    type Output = Self;

    fn sub(self, other: f64) -> Self::Output {
        Sample(self.0.iter().map(|a| a/other).collect())
    }
}

impl Div for Sample {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");
        Sample(self.0.iter().zip(other.0.iter()).map(|(a, b)| a/b).collect())
    }
}

impl Div<f64> for Sample {
    type Output = Self;

    fn div(self, other: f64) -> Self::Output {
        Sample(self.0.iter().map(|a| a/other).collect())
    }
}


impl Mul for Sample {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");
        Sample(self.0.iter().zip(other.0.iter()).map(|(a, b)| a*b).collect())
    }
}



impl Mul<f64> for Sample {
    type Output = Self;

    fn mul(self, other: f64) -> Self::Output {
        Sample(self.0.iter().map(|a| a*other).collect())
    }
}

impl Add for Sample {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(self.0.len(), other.0.len(), "Vectors must have the same length");
        Sample(self.0.iter().zip(other.0.iter()).map(|(a, b)| a + b).collect())
    }
}

impl Add<f64> for Sample {
    type Output = Self;

    fn add(self, other: f64) -> Self::Output {
        Sample(self.0.iter().map(|a| a+other).collect())
    }
}











#[derive(Clone, Serialize, Deserialize, Default, Debug)]
pub struct SampleArray(pub Array1<f64>);

//implement standard operations for our sample vector

impl AddAssign for SampleArray {
    fn add_assign(&mut self, other: Self) {
        self.0 = &self.0 + other.0;
    }
}

impl Div<f64> for SampleArray {
    type Output = Self;

    fn div(self, other: f64) -> Self::Output {
        SampleArray(self.0/other)
    }
}