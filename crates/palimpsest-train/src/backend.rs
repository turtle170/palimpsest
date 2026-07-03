//! Concrete backend selection — the only place in the workspace where a
//! backend is named. Everything else is generic over `B: Backend`.
//! Priority: tch > wgpu > ndarray if several features are enabled.

use burn::backend::Autodiff;

#[cfg(feature = "tch")]
pub type Inference = burn::backend::LibTorch<f32>;

#[cfg(all(feature = "wgpu", not(feature = "tch")))]
pub type Inference = burn::backend::Wgpu;

#[cfg(all(feature = "ndarray", not(feature = "wgpu"), not(feature = "tch")))]
pub type Inference = burn::backend::NdArray<f32>;

pub type Train = Autodiff<Inference>;

pub fn device() -> burn::tensor::Device<Inference> {
    Default::default()
}
