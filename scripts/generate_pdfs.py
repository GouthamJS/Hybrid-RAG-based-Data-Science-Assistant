"""
Generates synthetic PDFs for testing the hybrid retrieval system.
Requires: pip install fpdf2
"""
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
from fpdf import FPDF

PDFS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")

def generate_pdf(filename: str, title: str, content: str):
    """Creates a basic PDF using fpdf2."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16, style="B")
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, txt=content)
    
    os.makedirs(PDFS_DIR, exist_ok=True)
    pdf.output(os.path.join(PDFS_DIR, filename))
    print(f"Generated: {filename}")

ML_BASICS = """
Machine Learning Fundamentals

1. Overfitting: Overfitting occurs when a model learns the detailed and noise in the training data to the extent that it negatively impacts the performance of the model on new data. This means the noise or random fluctuations in the training data is picked up and learned as concepts by the model.

2. Underfitting: Underfitting happens when a model cannot capture the underlying trend of the data. It's usually the result of a very simple model that fails to see the complexity of the data, like trying to fit a linear model to a non-linear dataset.

3. Bias-Variance Tradeoff: The bias-variance tradeoff is the property of a set of predictive models whereby models with a lower bias in parameter estimation have a higher variance of the parameter estimates across samples, and vice versa. Finding the right balance is key to preventing both underfitting (high bias) and overfitting (high variance).
"""

DEEP_LEARNING = """
Deep Learning Architectures

1. Convolutional Neural Networks (CNN): CNNs extract spatial features from grid-like data such as images using convolutional filters. They are widely used in image and video recognition, recommender systems, and natural language processing.

2. Recurrent Neural Networks (RNN): RNNs are designed for sequential data context, like time series or natural language. They process sequential data by maintaining a hidden state across time steps, but suffer from the vanishing gradient problem.

3. Long Short-Term Memory (LSTM): LSTMs are a special kind of RNN capable of learning long-term dependencies. They use gating mechanisms (input, forget, and output gates) to solve the vanishing gradient problem.

4. Transformers: Transformers rely entirely on self-attention mechanisms to draw global dependencies between input and output. They have largely replaced RNNs in natural language processing (e.g., BERT, GPT).
"""

STATISTICS = """
Statistical Concepts in Data Science

1. Probability Distributions: A mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment. The Normal (Gaussian) distribution is characterized by its bell-shaped curve and is fundamental to many statistical methods.

2. Hypothesis Testing: A method of statistical inference used to decide whether the data at hand sufficiently support a particular hypothesis. It involves setting a Null Hypothesis (H0) and an Alternative Hypothesis (H1).

3. P-values: The p-value is the probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct. A smaller p-value (< 0.05) typically rejects the null hypothesis.
"""

SQL_INTERVIEW = """
Advanced SQL For Interviews

1. JOINs: SQL joins are used to combine rows from two or more tables. An INNER JOIN returns records that have matching values in both tables. A LEFT JOIN returns all records from the left table, and the matched records from the right table.

2. Subqueries: A subquery is a SQL query nested inside a larger query. It can be used for checking existence, filtering aggregations, and data manipulation.

3. Window Functions: Window functions operate on a set of rows and return a single value for each row from the underlying query. Examples include ROW_NUMBER(), RANK(), DENSE_RANK(), and LEAD().

4. Indexing: A database index is a data structure that improves the speed of data retrieval operations on a database table at the cost of additional writes and storage space.
"""

PYTHON_DS = """
Python Data Science Ecosystem

1. NumPy: NumPy is the fundamental package for scientific computing with Python. It provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions.

2. Pandas: Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language. It uses DataFrame and Series abstractions.

3. Matplotlib: A comprehensive library for creating static, animated, and interactive visualizations in Python. 

4. Scikit-learn: Simple and efficient tools for predictive data analysis. It provides implementations for many machine learning algorithms like random forests, SVMs, and k-means clustering.
"""

if __name__ == "__main__":
    generate_pdf("ml_basics.pdf", "Machine Learning Basics", ML_BASICS)
    generate_pdf("deep_learning.pdf", "Deep Learning Architectures", DEEP_LEARNING)
    generate_pdf("statistics.pdf", "Statistics for Data Science", STATISTICS)
    generate_pdf("sql_interview.pdf", "SQL Interview Prep", SQL_INTERVIEW)
    generate_pdf("python_ds.pdf", "Python Data Science Stack", PYTHON_DS)
    print("PDF generation complete.")
