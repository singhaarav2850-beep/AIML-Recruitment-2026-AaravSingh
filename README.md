# AI-ML Club Recruitment Task 2026

## 1. Candidate Details
* **Name:** AARAV SINGH
* **Register No.:** RA2511003010667
* **Year:** 2nd Year/3rd Sem
* **Branch/Degree:** B.Tech(CSE-Core)

## 2. Tasks Completed
* **Task 1:** Air Quality Forecasting
* **Task 2:** Neural Network (MNIST)

---

## 3. Problem Statements
* **Task 1:** Analyze historical air quality measurements from the UCI dataset and build a machine learning regression model to accurately predict future Carbon Monoxide (CO) levels based on temporal and lagged features.
* **Task 2:** Build, train, and evaluate a Deep Learning Neural Network to classify handwritten digits (0-9) using the classic MNIST dataset, and analyze how architectural changes affect performance.

## 4. Approach
* **Task 1 (Time-Series):** Handled invalid `-200` sensor readings by converting them to NaNs and forward-filling them. Engineered a 1-hour lag feature (`CO_lag_1`) and a 4-hour rolling average. Performed a strict chronological 80/20 train-test split to avoid data leakage, and trained a Random Forest Regressor.
* **Task 2 (Deep Learning):** Loaded the MNIST dataset and normalized the pixel values (0 to 1). Built a baseline Sequential model using TensorFlow/Keras with a Flatten layer, a ReLU hidden layer, and a Softmax output layer. Evaluated performance using a Confusion Matrix. Conducted an experiment by adding a second hidden layer and increasing neuron count to observe the impact on accuracy.

## 5. Technologies Used
* **Python**
* **Data & Machine Learning:** Pandas, NumPy, Scikit-Learn (Task 1)
* **Deep Learning:** TensorFlow, Keras (Task 2)
* **Visualization:** Matplotlib, Seaborn

## 6. Results
* **Task 1 Results:** Achieved an $R^2$ Score of 0.8787 and an MAE of 0.3010. The 1-hour lagged feature was overwhelmingly the strongest predictor of future pollution levels.
* **Task 2 Results:** The baseline neural network successfully classified digits with high accuracy. The confusion matrix revealed that misclassifications mostly occurred between visually similar numbers (e.g., 4s and 9s). The experimental model (more layers/neurons) converged faster on complex patterns.

## 7. Key Learnings
1. **Data Leakage:** Understood the critical importance of chronological splitting to prevent data leakage in time-series forecasting.
2. **Feature Engineering:** Learned how to engineer lagged and rolling-window features to give machine learning models historical context.
3. **Neural Network Architecture:** Learned the specific roles of activation functions—using ReLU in hidden layers to learn complex patterns, and Softmax in the output layer for multi-class probability.

## 8. Challenges & Solutions
* **Challenge 1 (Task 1):** Handling missing values in a time-series format without randomly dropping rows and breaking the timeline.
  * **Solution:** Used forward-filling (`ffill()`) to carry the last valid observation forward, which is a logical assumption for gradually changing environmental data.
* **Challenge 2 (Task 2):** Formatting the 2D image data so a standard neural network could process it.
  * **Solution:** Used a `Flatten` layer as the very first layer in the Keras Sequential model to convert the 28x28 pixel grid into a 1D array of 784 pixels.
