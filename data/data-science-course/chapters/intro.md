# Introduction to Data Science

## What Data Science Is

Data science is the intersection of statistics, programming, and domain expertise applied to the problem of extracting insight from data. It differs from traditional statistics in its emphasis on large, messy, real-world datasets and on building systems that automate decisions at scale. It differs from pure software engineering in its central concern with uncertainty, inference, and the quality of conclusions rather than just the correctness of code. The data science workflow — moving from problem definition through data collection, cleaning, modeling, and evaluation — is iterative and rarely linear. Understanding that workflow is as important as mastering any individual technique.

## Python and the Data Science Stack

Python has become the dominant language for data science because it combines readable syntax with an ecosystem of purpose-built libraries. Jupyter Notebooks enable an exploratory style of work in which code, output, and narrative coexist in the same document: code cells execute computations while markdown cells explain the reasoning. NumPy provides the numerical foundation — fast array operations that underlie almost every other library. Together these tools allow a data scientist to move fluidly from loading raw data to producing a finished analysis.

Variables, data types, and data structures are the basic vocabulary of this work. Numerical data, categorical data, ordinal data, and nominal data each carry different assumptions about what mathematical operations make sense. Measurement scales determine what you can validly compute — averaging an ordinal variable like a satisfaction rating is meaningful, but averaging a nominal variable like a zip code is not.

## Statistics Foundations

Statistical thinking separates data science from ad hoc pattern matching. Descriptive statistics — mean, median, variance — summarize what is in a dataset without making claims beyond it. Inferential statistics, including hypothesis testing and confidence intervals, allow conclusions to extend from a sample to a population, but only when assumptions about the data generating process are satisfied. Normal distribution assumptions, for instance, underlie many standard tests and must be checked. Correlation measures the strength of linear relationships between variables but does not establish causation — a distinction that matters enormously in practice.

## Machine Learning

Machine learning extends statistical modeling to problems where the goal is prediction rather than description. Supervised learning uses labeled training data — pairs of features and target variables — to learn a mapping that generalizes to new observations. Regression analysis predicts continuous outcomes; classification predicts categories. A train/test split is the basic discipline of model evaluation: a model that memorizes training data but fails on held-out test data has learned nothing useful. More robust evaluation uses cross-validation to reduce dependence on any single split.

## Visualization

Visualization serves two distinct purposes. In exploration, charts reveal patterns, outliers, and relationships that summary statistics obscure — a dataset's distribution, for instance, is almost always clearer in a histogram than in a table of numbers. In communication, visualization translates complex findings into forms that non-technical audiences can act on. Plot customization — careful choice of axes, labels, and color — is not decoration but argument.
