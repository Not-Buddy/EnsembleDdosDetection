//! ONNX inference + score normalization + LR combiner.
//!
//! Loads 3 ONNX models (IF, VAE, SVM), normalizes their outputs,
//! and combines via learned Logistic Regression.

use crate::config::{EnsembleConfig, NormalizationParams};
use anyhow::{Context, Result};
use ort::session::Session;
use ort::session::input::SessionInputValue;
use ort::value::Tensor;
use std::path::Path;

/// Detection result for a single flow.
#[derive(Debug)]
pub struct DetectionResult {
    pub if_score: f64,
    pub ae_score: f64,
    pub svm_score: f64,
    pub combined_prob: f64,
    pub is_attack: bool,
}

/// ONNX-based ensemble inference engine.
pub struct InferenceEngine {
    if_session: Session,
    ae_session: Session,
    svm_session: Session,
    norm_params: NormalizationParams,
    ensemble: EnsembleConfig,
    n_features: usize,
}

impl InferenceEngine {
    /// Load all 3 ONNX models and configs.
    pub fn load(
        models_dir: &Path,
        norm: NormalizationParams,
        ensemble: EnsembleConfig,
        n_features: usize,
    ) -> Result<Self> {
        tracing::info!("Loading ONNX models from {}", models_dir.display());

        let if_session = Session::builder()?
            .commit_from_file(models_dir.join("isolation_forest.onnx"))
            .context("Failed to load IF ONNX model")?;

        let ae_session = Session::builder()?
            .commit_from_file(models_dir.join("autoencoder.onnx"))
            .context("Failed to load AE ONNX model")?;

        let svm_session = Session::builder()?
            .commit_from_file(models_dir.join("one_class_svm.onnx"))
            .context("Failed to load SVM ONNX model")?;

        tracing::info!("All 3 ONNX models loaded successfully");

        Ok(Self {
            if_session,
            ae_session,
            svm_session,
            norm_params: norm,
            ensemble,
            n_features,
        })
    }

    /// Run inference on a preprocessed feature vector.
    pub fn predict(&mut self, features: &[f64]) -> Result<DetectionResult> {
        let input_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();

        let if_score = self.run_if(&input_f32)?;
        let ae_score = self.run_ae(&input_f32)?;
        let svm_score = self.run_svm(&input_f32)?;

        // LR Combiner: sigmoid(coef·scores + intercept)
        let coefs = &self.ensemble.coefficients;
        let logit = coefs[0] * if_score
            + coefs[1] * ae_score
            + coefs[2] * svm_score
            + self.ensemble.intercept;
        let combined_prob = sigmoid(logit);
        let is_attack = combined_prob >= self.ensemble.threshold;

        Ok(DetectionResult {
            if_score,
            ae_score,
            svm_score,
            combined_prob,
            is_attack,
        })
    }

    /// Create a SessionInputValue from f32 data.
    fn make_input(&self, data: &[f32]) -> Result<SessionInputValue<'static>> {
        let tensor =
            Tensor::from_array(([1usize, self.n_features], data.to_vec().into_boxed_slice()))
                .context("Failed to create input tensor")?;
        Ok(SessionInputValue::from(tensor))
    }

    fn run_if(&mut self, input: &[f32]) -> Result<f64> {
        let inp = self.make_input(input)?;
        let outputs = self.if_session.run([inp])?;

        // IF ONNX outputs: [0]=labels, [1]=score_samples
        let (_shape, scores) = outputs[1]
            .try_extract_tensor::<f32>()
            .context("Failed to extract IF scores")?;
        let raw = *scores.first().unwrap_or(&0.0) as f64;

        let p1 = self.norm_params.isolation_forest.p1;
        let p99 = self.norm_params.isolation_forest.p99;
        let denom = p99 - p1;
        let norm = if denom.abs() > 1e-10 {
            (raw - p1) / denom
        } else {
            0.0
        };
        Ok((1.0_f64 - norm).clamp(0.0, 1.0))
    }

    fn run_ae(&mut self, input: &[f32]) -> Result<f64> {
        let inp = self.make_input(input)?;
        let outputs = self.ae_session.run([inp])?;

        let (_shape, recon) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract AE output")?;

        // MSE reconstruction error
        let mse: f64 = input
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| {
                let d = (*a as f64) - (*b as f64);
                d * d
            })
            .sum::<f64>()
            / input.len() as f64;

        let s_min = self.norm_params.autoencoder.score_min;
        let s_max = self.norm_params.autoencoder.score_max;
        let denom = s_max - s_min;
        let norm = if denom > 1e-10 {
            (mse - s_min) / denom
        } else {
            0.0
        };
        Ok(norm.clamp(0.0_f64, 1.0))
    }

    fn run_svm(&mut self, input: &[f32]) -> Result<f64> {
        let inp = self.make_input(input)?;
        let outputs = self.svm_session.run([inp])?;

        // SVM ONNX outputs: [0]=labels, [1]=decision_function
        let (_shape, scores) = outputs[1]
            .try_extract_tensor::<f32>()
            .context("Failed to extract SVM scores")?;
        let raw = *scores.first().unwrap_or(&0.0) as f64;

        let s_min = self.norm_params.one_class_svm.score_min;
        let s_max = self.norm_params.one_class_svm.score_max;
        let denom = s_max - s_min;
        let norm = if denom.abs() > 1e-10 {
            (raw - s_min) / denom
        } else {
            0.0
        };
        Ok((1.0_f64 - norm).clamp(0.0, 1.0))
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
