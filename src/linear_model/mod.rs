use std::cmp::max;

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
use crate::metrics::r_squared;
use crate::primitives::{Matrix, Vector};
use serde::{Deserialize, Serialize};

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

///! LinearRegression
//* - Use when features are INDEPENDENT (aka no colinearity between variables)
//* - The relationship between the features should be linear
//* - Super fast as it just has to compute a closed form expression
impl LinearRegression {
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
        }
    }

    // Sets whether an intercept term should be used during fit
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn coefficients(&self) -> &Vector<f32> {
        self.coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
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

        Matrix::from_vec(n_rows, n_cols + 1, data)
            .expect("could not construct matrix due to internal error")
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

/// Implements Ridge regression
/// Uses L2 regularisation shrinks all coefficients smoothly
/// Instead of minimizing the error and a penalty term over the weights (beta)
pub struct Ridge {
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term.
    intercept: f32,
    /// Whether to fit an intercept.
    fit_intercept: bool,
}

impl Ridge {
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
        }
    }

    // Sets whether an intercept term should be used during fit
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn coefficients(&self) -> &Vector<f32> {
        self.coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    pub fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>, lambda: f32) -> Result<()> {
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
        // beta = (X^T * X + lambda * I)^-1 * X^T * y
        // rewrite as X^T * X * B = X^Ty
        let x_fit = if self.fit_intercept {
            LinearRegression::add_intercept_column(&x)
        } else {
            x.clone()
        };

        let num_params = if self.fit_intercept {
            num_features + 1
        } else {
            num_features
        };

        // (X^T * X + lambda * I)
        let mut x_xt = x_fit.
            transpose().
            matmul(&x_fit).
            unwrap();

        // only update non-intercept parameters
        for i in 0..num_params {
            if self.fit_intercept && i == 0 {
                continue;
            }
            
            // only scale main diagonal
            let current = x_xt.get(i,i);
            x_xt.set(i, i, current + lambda);
        }

        // X^T * y 
        let x_xt_y = x_fit.
            transpose().
            matvec(y).
            unwrap();

        // solve for beta
        let beta = x_xt.cholesky_solve(&x_xt_y).unwrap();

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
    pub fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32 {
        let y_pred = self.predict(x);
        r_squared(&y_pred, y)
    }
}

pub struct Lasso {
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term.
    intercept: f32,
    /// Whether to fit an intercept.
    fit_intercept: bool,
    /// Maximum number of iterations for coordinate descent.
    max_iter: usize,
    /// Tolerance for convergence.
    tol: f32,
    /// Regularization strength
    alpha: f32,
}

impl Lasso {
    fn new(&self, max_iter: usize, tolerance: f32) -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
            max_iter: max_iter,
            tol: tolerance,
            alpha: 0.0,
        }
    }

    // Sets whether an intercept term should be used during fit
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn coefficients(&self) -> &Vector<f32> {
        self.coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    pub fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    /// Soft-thresholding operator for L1 regularization.
    fn soft_threshold(x: f32, lambda: f32) -> f32 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }

    // fit model using coordinate descent algorithm
    pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let (num_samples, num_features) = x.shape();

        if num_samples == 0 {
            return Err("Can not fit model with no samples".into())
        }

        if num_samples != y.len() {
            return Err("Number of samples is not equal to number of targets".into());
        }

        // Center data if fitting intercept
        let (x_centered, y_centered, y_mean) = if self.fit_intercept {
            // Compute means
            let mut x_mean = vec![0.0; num_features];
            let mut y_sum = 0.0;
            for i in 0..num_samples {
                // matrix average - sum across every feature in row
                for (j, mean_j) in x_mean.iter_mut().enumerate() {
                    *mean_j += x.get(i, j);
                }

                // vector average
                y_sum += y[i];
            }

            for mean in &mut x_mean {
                *mean /= num_samples as f32;
            }
            let y_mean = y_sum / num_samples as f32;

            // Center data
            let mut x_data = vec![0.0; num_samples * num_features];
            let mut y_data = vec![0.0; num_samples];

            for i in 0..num_samples {
                for j in 0..num_features {
                    x_data[i * num_features + j] = x.get(i, j) - x_mean[j];
                }
                y_data[i] = y[i] - y_mean;
            }

            (
                Matrix::from_vec(num_samples, num_features, x_data)
                    .expect("Valid matrix dimensions for property test"),
                Vector::from_vec(y_data),
                y_mean,
            )
        } else {
            (x.clone(), y.clone(), 0.0)
        };

        let mut beta = vec![0.0; num_features];

        let mut col_norms_sq = vec![0.0; num_features];
        for (j, norm_sq) in col_norms_sq.iter_mut().enumerate() {
            for i in 0..num_samples {
                let val = x_centered.get(i, j);
                *norm_sq += val * val;
            }
        }

        // iterative algorrithm with max limit (max_iter) and soft limit when max_change is within certain range
        for _ in 0..self.max_iter {
            let mut max_change = 0.0f32;

            // iterate through every feature (j is the current feature we optimise for, freeze everything else)
            for j in 0..num_features {
                if col_norms_sq[j] < 1e-10 {
                    continue; // skip zero features
                }

                // rho is a weighted sum of losses on a particular sample on the feature j
                // so the greater the loss the more important the feature is, since the prediction is far off the target
                // thus the lower the loss the less important the feature is, since the prediction is closer to the target
                let mut rho = 0.0;
                for i in 0..num_samples {
                    
                    // calculate prediction for sample (excluding the feature j)
                    let mut pred  = 0.0;
                    for (k, &beta_k) in beta.iter().enumerate() {
                        if k != j { // if k is not the current feature (j) then calculate prediction based on current beta
                            pred += x_centered.get(i,k) * beta_k;
                        }
                    }
                    
                    // calculate residual
                    let difference = y_centered[i] - pred;
                    
                    // update rho for feature j 
                    rho += x_centered.get(i, j) * difference;
                }

                let old_beta = beta[j];
                beta[j] = Self::soft_threshold(rho, self.alpha) / col_norms_sq[j];

                let change = (beta[j] - old_beta).abs();
                if change > max_change {
                    max_change = change;
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        Ok(())
    }
}

impl Default for Lasso {
    fn default() -> Self {
        Self { 
            coefficients: None, 
            intercept: 0.0, 
            fit_intercept: true, 
            max_iter: 1000, 
            tol: 1e-4, 
            alpha: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_new() {
        let model = LinearRegression::new();
        assert!(!model.is_fitted());
        assert!(model.fit_intercept)
    }

    #[test]
    fn test_simple_regression() {
        // y = 3x + 1 (This is the what it should figure out)
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
            .expect("Should work, just constructing vector");
        let y = Vector::from_slice(&[4.0, 7.0, 10.0, 13.0]);

        let mut model = LinearRegression::new();
        model
            .fit(&x, &y)
            .expect("Fit should succeed with the given test data");

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

    proptest! {
        #[test]
        fn ols_works(
            // generate 10-20 x values in the range -100-100
            x_vals in prop::collection::vec(-100.0f32..100.0f32, 10..20),
            true_slope in -10.0f32..10.0f32,
            true_intercept in -10.0f32..10.0f32,
        ) {
            let num_features = x_vals.len();

            // construct features from generated input
            let x = Matrix::from_vec(num_features, 1, x_vals.clone()).unwrap();

            // construct target matrix using generated input, slope and intercept
            let y: Vec<f32> = x_vals.iter()
                .map(|&x_val| true_slope * x_val + true_intercept)
                .collect();

            // construct vector
            let y = Vector::from_vec(y);

            let mut model = LinearRegression::new();
            model.fit(&x, &y).expect("Fit should succeed on the valid input data");
            let coef = model.coefficients();
            prop_assert!((coef[0] - true_slope).abs() < 0.01);
            prop_assert!((model.intercept - true_intercept).abs() < 0.01);
        }
    }

    /// Ridge tests
    #[test]
    fn test_simple_ridge() {
        // y = 2x + 1
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
            .expect("Valid matrix dimensions for test");
        let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

        let mut model = Ridge::new(); 
        model
            .fit(&x, &y, 0.0) // No regularization = OLS
            .expect("Fit should succeed with valid test data");

        assert!(model.is_fitted());

        let r2 = model.score(&x, &y);
        assert!(r2 > 0.99);
    }

    #[test]
    fn test_shrinking_coefficients() {
        // Test that higher alpha shrinks coefficients
        let x = Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
            .expect("Valid matrix dimensions for test");
        let y = Vector::from_slice(&[4.0, 8.0, 12.0, 16.0, 20.0]);
        
        // Low regularization
        let mut low_reg = Ridge::new();
        low_reg
            .fit(&x, &y, 0.01)
            .expect("Fit should succeed with valid test data");

        // High regularization
        let mut high_reg = Ridge::new();
        high_reg
            .fit(&x, &y, 100.0)
            .expect("Fit should succeed with valid test data");

        // Higher regularization should produce smaller coefficient magnitudes
        let low_coef = low_reg.coefficients();
        let high_coef = high_reg.coefficients();
        let low_norm: f32 = (0..low_coef.len()).map(|i| low_coef[i] * low_coef[i]).sum();
        let high_norm: f32 = (0..high_coef.len())
            .map(|i| high_coef[i] * high_coef[i])
            .sum();

        assert!(
            high_norm < low_norm,
            "High regularization should shrink coefficients: {high_norm} < {low_norm}"
        );
    }

    #[test]
    // confirm that we can handle case where matrix is no longer invertible
    fn test_non_invertible_ridge() {
        // 3 samples, 5 features 
        let x = Matrix::from_vec( 3, 5, vec![ 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0, ], ) .expect("Valid matrix dimensions for test"); 
        let y = Vector::from_vec(vec![10.0, 20.0, 30.0]);    

        let mut model = Ridge::new();
        let res = model.fit(&x, &y, 1.0);
        assert!(res.is_ok());
    }
}
