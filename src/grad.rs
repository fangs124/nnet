use std::{iter::zip, ops::Add};

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::Network;

#[derive(Serialize, Deserialize, Clone)]
pub struct Gradient {
    pub dws: Vec<DMatrix<f32>>,
    pub dbs: Vec<DVector<f32>>,
    pub dws_shape: Vec<(usize, usize)>,
    pub dbs_shape: Vec<usize>,
}

impl Gradient {
    pub fn new() -> Self {
        Gradient { dbs: Vec::new(), dbs_shape: Vec::new(), dws: Vec::new(), dws_shape: Vec::new() }
    }

    pub fn zero<T>(net: &Network<T>) -> Self {
        let mut grad = Gradient::new();
        for layer in &net.layers {
            grad.dbs.push(DVector::zeros(layer.b.nrows()));
            grad.dbs_shape.push(layer.b.nrows());
            grad.dws.push(DMatrix::zeros(layer.w.nrows(), layer.w.ncols()));
            grad.dws_shape.push((layer.w.nrows(), layer.w.ncols()));
        }
        return grad;
    }

    pub fn sum_pairs(pairs: &mut Vec<(Gradient, f32)>) -> Gradient {
        let (mut grad, r) = pairs.pop().unwrap();
        grad.scalar_mul(&r);
        let mut sum: Gradient = grad;
        for (grad, r) in pairs.iter_mut() {
            grad.scalar_mul(&r);
            sum = sum + grad.clone();
        }
        return sum;
    }

    pub fn sum(grads: &mut Vec<Gradient>) -> Gradient {
        let mut sum: Gradient = grads.pop().unwrap();
        for grad in grads.iter() {
            sum = sum + grad;
        }
        return sum;
    }

    pub fn scalar_mul(&mut self, r: &f32) {
        for i in 0..self.dbs.len() {
            self.dws[i] *= *r;
            self.dbs[i] *= *r;
        }
    }
}

impl Add<Gradient> for Gradient {
    type Output = Self;

    fn add(self, rhs: Gradient) -> Self::Output {
        assert!((self.dbs_shape == rhs.dbs_shape) && (self.dws_shape == rhs.dws_shape));
        let mut Gradient = Gradient::new();
        Gradient.dbs_shape = self.dbs_shape;
        Gradient.dws_shape = self.dws_shape;
        for (db_l, db_r) in zip(self.dbs, rhs.dbs) {
            Gradient.dbs.push(db_l + db_r);
        }
        for (dw_l, dw_r) in zip(self.dws, rhs.dws) {
            Gradient.dws.push(dw_l + dw_r);
        }
        return Gradient;
    }
}

//FIXME
impl std::ops::Add<&Gradient> for Gradient {
    type Output = Gradient;

    fn add(self, rhs: &Gradient) -> Self::Output {
        return self + rhs.clone();
    }
}
