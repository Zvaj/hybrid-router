# Statistics for Data Science

## Descriptive Statistics

Descriptive Statistics summarize what is in a dataset without making claims beyond it. The central tendency measures — Mean, Median, and Mode — each capture different aspects of a distribution's center. Mean is sensitive to outliers; Median is robust. Spread is quantified by Range, Variance, and Standard Deviation. Variance is the average squared deviation from the mean; Standard Deviation is its square root, restoring the original units. Quartiles and Percentiles partition ranked data; the Interquartile Range (IQR) measures spread for the middle 50%.

Shape matters beyond center and spread. Skewness measures asymmetry — a right-skewed distribution has a long tail extending right, pulling the mean above the median. Kurtosis measures tail heaviness relative to a Normal Distribution. The Distribution of a variable is fully characterized by its probability density function, but Descriptive Statistics compress it into a handful of interpretable numbers.

Correlation measures the linear relationship between two variables. The Pearson Correlation coefficient ranges from −1 to +1; Spearman Correlation extends this to ranked data. A Correlation Matrix displays pairwise correlations among all variables simultaneously. Critically, correlation does not establish causation — two variables can be correlated because one causes the other, because both share a common cause, or by coincidence.

## Inferential Statistics

Inferential statistics extend conclusions from a Sample to a Population. The Central Limit Theorem guarantees that sample means follow a Normal Distribution as sample size grows, regardless of the population's shape — this is why normal-theory methods apply so broadly. Sampling must be done carefully: Random Sampling gives each individual an equal chance of selection; Stratified Sampling ensures key subgroups are represented proportionally.

Hypothesis Testing provides a formal framework for making binary decisions under uncertainty. A P-Value is the probability of observing data at least as extreme as what was seen, assuming the null hypothesis is true. Statistical Significance is declared when the p-value falls below a threshold (typically 0.05). Confidence Intervals provide an alternative that conveys both the estimated effect and its uncertainty: a 95% Confidence Interval captures the true parameter value in 95% of repeated experiments.

## Regression Analysis

Regression Analysis estimates the relationship between an Independent Variable (or predictor) and a Dependent Variable (or outcome). Linear Regression fits a Regression Line y = mx + b using the Least Squares Method, minimizing the Sum of Squared Errors. The Slope and Intercept are the Regression Coefficients; the Scikit-learn Library implements this via the LinearRegression Class, which exposes a Fit Method to estimate coefficients and a Predict Method to apply them.

Residuals — differences between Fitted Values and actual observations — diagnose whether the Assumptions of Regression hold: Linearity Assumption, Homoscedasticity (constant variance), Independence Assumption, and Normality of Residuals. Residual Analysis via a Residual Plot reveals systematic violations. R-Squared measures the proportion of variance explained; Mean Squared Error and Root Mean Squared Error measure average prediction error in original units.

## Model Evaluation

Overfitting occurs when a model learns the Training Data too well, capturing noise rather than signal, and fails to Generalize to new observations. The Train Test Split discipline addresses this: reserve Testing Data the model never sees during training, and report performance on that held-out set. Model Performance on training data is always optimistic; Test Error is the honest measure.

Cross-Validation improves on a single Train Test Split by rotating which portion of data serves as the validation set. K-Fold Cross-Validation splits data into k groups, training k times each with a different fold held out, and averages the results. This reduces dependence on any single random split and makes better use of limited data. The Bias-Variance Tradeoff governs Model Complexity: simple models have high Bias (systematic error) but low Variance; complex models have low bias but high Variance (sensitivity to training data).
