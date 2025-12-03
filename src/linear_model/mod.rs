///! Linear regression modelling
/// 
/// Goal: Given a dataset with features X and target y, we seek to find a set of parameters (beta) that minimise the 
/// squared error
/// Formally: min_{w,b} \sum^n_{i=1} (y_i - (w * x_i + b))^2
/// More compactly argmin_{beta} ||y - X * beta||^2 (MSE)
/// We find the optimal beta using the normal equation: beta = (X^T * X)^-1 * X^T * y, **OLS estimator** (ordinary least squares)
///* This is considered a closed form solution as we can directly compute the optimal beta without iterative methods
/// The normal equation is derived by rewriting the loss function and taking the partial derivative wrt beta, setting it to 0 and solving for beta

use crate::error::Result;
use crate::primitives::{Vector, Matrix};
use serde::{Serialize, Deserialize};
use crate::metrics::r_squared;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term.
    intercept: f32,
    /// Whether to fit an intercept.
    fit_intercept: bool,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegression {
    pub fn new() -> Self {
        Self { 
            coefficients: None, 
            intercept: 0.0, 
            fit_intercept: true
        }
    }

    // Sets whether an intercept term should be used during fit
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn coefficients(&self) -> &Vector<f32> {
        self.coefficients.as_ref().expect("Model not fitted. Call fit() first.")
    }

    pub fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    /// add intercept/bias term to feature matrix
    fn add_intercept_column(x: &Matrix<f32>) -> Matrix<f32> {
        let (n_rows, n_cols) = x.shape();
        let mut data = Vec::with_capacity(n_rows * (n_cols + 1));

        for i in 0..x.n_rows() {
            data.push(1.0);
            for j in 0..x.n_cols() {
                data.push(x.get(i, j));
            }
        }

        Matrix::from_vec(n_rows, n_cols+1, data).expect("could not construct matrix due to internal error")
    }

    /// attempt to find beta to best fit input samples X to minimise error from y
    pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let (num_samples, num_features) = x.shape();

        // confirm that we are given data samples to fit
        if num_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        // confirm that every sample has a target value
        if num_samples != y.len() {
            return Err("Number of data samples must match target length".into());
        }

        // Fitting is basically trying to find a vector of parameters
        // beta = (X^T * X)^-1 * X^T * y
        // rewrite as X^T * X * B = X^Ty
        let x_fit = if self.fit_intercept {
            Self::add_intercept_column(x)
        } else {
            x.clone()
        };

        let x_t = x_fit.transpose();
        let xt_x = x_t.matmul(&x_fit).unwrap();
        let x_t_y = x_t.matvec(&y).unwrap();
        let beta = xt_x.cholesky_solve(&x_t_y).unwrap();

        // Extract intercept and coefficients
        if self.fit_intercept {
            self.intercept = beta[0];
            self.coefficients = Some(beta.slice(1, num_features + 1));
        } else {
            self.intercept = 0.0;
            self.coefficients = Some(beta);
        }

        Ok(())
    }

    /// predict y given x using beta => y = X * beta + b
    pub fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        let coefficients = self
            .coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.");

        let result = x
            .matvec(coefficients)
            .expect("Matrix dimensions don't match coefficients");

        result.add_scalar(self.intercept)
    }

    /// Computes the R² score.
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32 {
        let y_pred = self.predict(x);
        r_squared(&y_pred, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let model = LinearRegression::new();
        assert!(!model.is_fitted());
        assert!(model.fit_intercept)
    }

    #[test]
    fn test_simple_regression() {
        // y = 3x + 1 (This is the what it should figure out)
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("Should work, just constructing vector");
        let y = Vector::from_slice(&[4.0, 7.0, 10.0, 13.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).expect("Fit should succeed with the given test data");
        
        assert!(model.is_fitted());

        let coef = model.coefficients();
        assert!((coef[0] - 3.0).abs() < 1e-4);
        assert!((model.intercept - 1.0) < 1e-4);

        // confirm predictions
        let predictions = model.predict(&x);
        for i in 0..y.len() {
            assert!(predictions[i] == y[i]);
        }

        // confirm R²
        let score = model.score(&x, &y);
        assert!((score - 1.0).abs() < 1e-4);
    }
}