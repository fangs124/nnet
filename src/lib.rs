extern crate nalgebra as na;

pub mod grad;
mod phi;

use std::marker::PhantomData;

use na::base::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

pub use grad::Gradient;
use phi::PhiT;

use crate::phi::safesoftmax;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Network<T> {
    pub input_dim: usize,
    pub node_counts: Vec<usize>,
    pub layers: Vec<Layer>,
    _phantom: PhantomData<T>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Layer {
    pub w: DMatrix<f32>,
    pub b: DVector<f32>,
    pub z: DVector<f32>, //z = w*phi(z') + b
    index: usize,
    ty: LayerT,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LayerT {
    Pi,
    Act(PhiT),
}

pub trait InputType {
    fn to_vector(&self) -> DVector<f32>;
}

pub trait SparseInputType {
    fn to_sparse_vec(&self) -> SparseVec;
}

#[derive(Serialize, Deserialize, Debug, Clone, Eq, PartialEq)]
pub struct SparseVec {
    data: Vec<usize>,
}

impl IntoIterator for SparseVec {
    type Item = usize;

    type IntoIter = <Vec<usize> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl SparseVec {
    pub fn new() -> Self {
        SparseVec { data: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        SparseVec { data: Vec::with_capacity(capacity) }
    }

    pub fn push(&mut self, value: usize) {
        self.data.push(value);
    }
}

impl<T: InputType> Network<T> {
    //const TRESHOLD: f32 = 0.0005;
    //const DEFAULT_ALPHA: f32 = 0.5;
    //const DEFAULT_GAMMA: f32 = 0.90;
    const REG_COEFF: f32 = 0.0001;
    const DEFAULT_IN_PHI: PhiT = PhiT::LReLU6;
    const DEFAULT_IN_TY: LayerT = LayerT::Act(Network::<T>::DEFAULT_IN_PHI);
    const DEFAULT_OUT_PHI: PhiT = PhiT::Tanh;
    const DEFAULT_OUT_TY: LayerT = LayerT::Act(Network::<T>::DEFAULT_OUT_PHI);
    const PI_TY: LayerT = LayerT::Pi;
    //pub fn update_sum(&mut self, pairs: &mut Vec<(Gradient, f32)>) {
    //    let grad_count = pairs.len();
    //    let total = Gradient::sum_pairs(pairs);
    //    //TODO itertools used here, maybe remove dependency?
    //    for (l, db, dw) in itertools::izip!(&mut self.layers, total.dbs, total.dws) {
    //        l.b += db / (grad_count as f32);
    //        l.w += dw / (grad_count as f32);
    //    }
    //}

    pub fn update(&mut self, grad: Gradient, r: f32) {
        //TODO itertools used here, maybe remove dependency?
        for (l, db, dw) in itertools::izip!(&mut self.layers, grad.dbs, grad.dws) {
            l.b += r * db;
            l.w += r * dw;
        }
    }

    pub fn z(&self) -> Vec<f32> {
        let i = self.layers.len() - 1;
        self.layers[i].z.data.as_vec().to_vec()
    }

    pub fn phi_z(&self) -> Vec<f32> {
        let i = self.layers.len() - 1;
        self.layers[i].phi().data.as_vec().to_vec()
    }

    pub fn phi_z_vector(&self) -> DVector<f32> {
        let i = self.layers.len() - 1;
        self.layers[i].phi()
    }

    pub fn new(input_dim: usize, node_counts: Vec<usize>) -> Self {
        let mut layers: Vec<Layer> = Vec::with_capacity(node_counts.len());
        let mut j = input_dim;
        for (nth, &i) in node_counts.iter().enumerate() {
            layers.push(Layer::new(i, j, nth, Network::<T>::DEFAULT_IN_TY));
            j = i;
        }

        layers[node_counts.len() - 1].ty = Network::<T>::DEFAULT_OUT_TY;
        Network { input_dim, node_counts, layers, _phantom: PhantomData }
    }

    pub fn new_pi_net(input_dim: usize, node_counts: Vec<usize>) -> Self {
        let mut layers: Vec<Layer> = Vec::with_capacity(node_counts.len());
        let mut j = input_dim;
        for (nth, &i) in node_counts.iter().enumerate() {
            layers.push(Layer::new(i, j, nth, Network::<T>::DEFAULT_IN_TY));
            j = i;
        }

        layers[node_counts.len() - 1].ty = Network::<T>::PI_TY;
        Network { input_dim, node_counts, layers, _phantom: PhantomData }
    }

    #[inline(always)]
    pub fn forward_prop(&mut self, input: &impl InputType) {
        self.forward_prop_vector(input.to_vector());
    }

    pub fn forward_prop_vector(&mut self, input: DVector<f32>) {
        let mut prev_phiz = input;
        for layer in self.layers.iter_mut() {
            layer.compute_z(&prev_phiz);
            prev_phiz = layer.phi();
        }
    }

    #[inline(always)]
    pub fn forward_prop_sparse(&mut self, input: &impl InputType) {
        self.forward_prop_vector(input.to_vector());
    }

    pub fn forward_prop_sparse_vec(&mut self, input: SparseVec) {
        let mut layers = self.layers.iter_mut();
        let first_layer = layers.next().unwrap();
        let d = first_layer.z.len();
        let mut sum = DVector::from_element(d, 0.0);
        for index in input {
            sum += first_layer.w.column(index)
        }
        first_layer.z = sum;
        let mut prev_phiz = first_layer.phi();
        for layer in layers {
            layer.compute_z(&prev_phiz);
            prev_phiz = layer.phi();
        }
    }

    //#[inline(always)]
    //pub fn forward_prop_phi_mutless(&self, input: &impl InputType) -> Vec<f32> {
    //    let mut prev_phiz = input.to_vector();
    //    for layer in self.layers.iter() {
    //        let current_phiz = layer.compute_z_mutless(&prev_phiz);
    //        assert!(layer.ty != LayerT::Pi);
    //        let LayerT::Act(ty) = layer.ty.clone() else { unreachable!() };
    //        prev_phiz = current_phiz.map(ty.phi());
    //    }
    //    prev_phiz.data.as_vec().to_vec()
    //}

    #[inline(always)]
    pub fn backward_prop(&mut self, input: &impl InputType, target: DVector<f32>, r: f32) -> Gradient {
        return self.backward_prop_vector(input.to_vector(), target, r);
    }

    pub fn backward_prop_vector(&mut self, input: DVector<f32>, target: DVector<f32>, r: f32) -> Gradient {
        //z = w*phi(z') + b
        //a = phi(z)

        // dphi/da
        self.forward_prop_vector(input.clone());
        let mut dphida = r.abs() * (DVector::from(self.phi_z()) - target);

        let mut grad = Gradient::new();
        for layer in self.layers.iter().rev() {
            //  dphi_n/dz_k    = dphi_n/da_k     * da_k/dz_k
            let dphidz = dphida.component_mul(&layer.dphi());
            //let dphidz = match layer.ty {
            //    LayerT::Pi => layer.dphi() * dphida,
            //    LayerT::Act(_) => dphida.component_mul(&layer.dphi()),
            //};

            // dz/dw           = a_{k-1}
            let dzdw = match layer.index {
                0 => input.clone(),
                _ => self.layers[layer.index - 1].phi(),
            };

            // we do this first so we can borrow dphidz here.. otherwise it makes more sense to do it at the end of the loop.
            // dphi_n/da_{k-1} = dphi_n/da_k     * da/dz   * dz/da_{k-1}
            dphida = layer.w.tr_mul(&dphidz);

            // dN/dw           = dphi_n/da_k     * da/dz   * dz/dw
            //                 = dphi_n/da_k     * dphi(z) * a

            //no L2 regularization
            //grad.dws.push(&dphidz * dzdw.transpose());
            //L2 regularization
            let dw = &dphidz * dzdw.transpose();
            let norm = dw.norm();
            grad.dws.push(dw - (norm / 1000.0) * layer.w.clone());

            // dN/db           = dphi_n/da_k     * da/dz   * dz/db
            //                 = dphi_n/da_k     * dphi(z) * 1
            grad.dbs.push(dphidz.clone());
        }

        grad.dbs.reverse();
        grad.dws.reverse();
        for (db, dw) in grad.dbs.iter().zip(&grad.dws) {
            grad.dws_shape.push(dw.shape());
            grad.dbs_shape.push(db.len());
        }

        return grad;
    }

    #[inline(always)]
    pub fn backward_prop_sparse(&mut self, input: &impl SparseInputType, target: DVector<f32>, r: f32) -> Gradient {
        return self.backward_prop_sparse_vec(input.to_sparse_vec(), target, r);
    }

    pub fn backward_prop_sparse_vec(&mut self, input: SparseVec, target: DVector<f32>, r: f32) -> Gradient {
        //z = w*phi(z') + b
        //a = phi(z)

        // dphi/da
        self.forward_prop_sparse_vec(input.clone());
        let mut dphida = r.abs() * (DVector::from(self.phi_z()) - target);

        //convert to sparse vector
        let d = self.input_dim;
        let mut input_vector = DVector::from_element(d, 0.0);
        for index in input {
            input_vector[index] = 1.0;
        }
        let mut grad = Gradient::new();

        for layer in self.layers.iter().rev() {
            //  dphi_n/dz_k    = dphi_n/da_k     * da_k/dz_k
            let dphidz = dphida.component_mul(&layer.dphi());
            //let dphidz = match layer.ty {
            //    LayerT::Pi => layer.dphi() * dphida,
            //    LayerT::Act(_) => dphida.component_mul(&layer.dphi()),
            //};

            // dz/dw           = a_{k-1}
            let dzdw = match layer.index {
                0 => input_vector.clone(),
                _ => self.layers[layer.index - 1].phi(),
            };

            // we do this first so we can borrow dphidz here.. otherwise it makes more sense to do it at the end of the loop.
            // dphi_n/da_{k-1} = dphi_n/da_k     * da/dz   * dz/da_{k-1}
            dphida = layer.w.tr_mul(&dphidz);

            // dN/dw           = dphi_n/da_k     * da/dz   * dz/dw
            //                 = dphi_n/da_k     * dphi(z) * a

            //no L2 regularization
            //grad.dws.push(&dphidz * dzdw.transpose());
            //L2 regularization
            let dw = &dphidz * dzdw.transpose();
            let norm = dw.norm();
            grad.dws.push(dw - (norm / 1000.0) * layer.w.clone());

            // dN/db           = dphi_n/da_k     * da/dz   * dz/db
            //                 = dphi_n/da_k     * dphi(z) * 1
            grad.dbs.push(dphidz.clone());
        }

        grad.dbs.reverse();
        grad.dws.reverse();
        for (db, dw) in grad.dbs.iter().zip(&grad.dws) {
            grad.dws_shape.push(dw.shape());
            grad.dbs_shape.push(db.len());
        }

        return grad;
    }
}

impl Layer {
    pub fn new(i: usize, j: usize, index: usize, ty: LayerT) -> Self {
        let w = DMatrix::new_random(i, j) - DMatrix::from_element(i, j, -0.5);
        let b = DVector::new_random(i) - DVector::from_element(i, -0.5);
        let z = DVector::zeros(i);
        Layer { w, b, z, index, ty }
    }

    pub fn compute_z(&mut self, prev_phiz: &DVector<f32>) {
        self.z = &self.w * prev_phiz + &self.b;
    }

    pub fn phi(&self) -> DVector<f32> {
        match &self.ty {
            LayerT::Pi => Layer::safesoftmax_vector(&self.z),
            LayerT::Act(phi_t) => self.z.map(phi_t.phi()),
        }
    }

    pub fn dphi(&self) -> DVector<f32> {
        match &self.ty {
            LayerT::Pi => DVector::from_element(self.z.len(), 1.0), //TODO might need to fix this
            LayerT::Act(phi_t) => self.z.map(phi_t.dphi()),
        }
    }

    fn safesoftmax_vector(xs: &DVector<f32>) -> DVector<f32> {
        let mut total: f32 = 0.0;
        let mut output_vec: Vec<f32> = Vec::with_capacity(xs.len());
        let max_x = xs.iter().max_by(|&a, &b| a.total_cmp(&b)).unwrap();

        for &x in xs {
            let val: f32 = (x - max_x).exp();
            total += val;
            output_vec.push(val);
        }

        DVector::from_vec(output_vec.iter().map(|x: &f32| x / total).collect::<Vec<f32>>())
    }
}
