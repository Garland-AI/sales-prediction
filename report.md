
# Report on Time-Series Sales Prediction Model

This report outlines the development of a time-series prediction model using historical sales data for various items and customers. The goal is to forecast total sales and predict future quantities. The sections below describe the model design and key steps.

---

## 1. Model Structure

The model is built using a Sequential neural network architecture, specifically utilizing **LSTM (Long Short-Term Memory)** layers to capture the temporal dependencies in the sales data. The key details of the model are as follows:

### 1.1 Architecture:
- **Layer 1:** LSTM with 100 units, `relu` activation, and return sequences enabled (to keep the sequence structure).
- **Layer 2:** LSTM with 50 units and `relu` activation (no return sequences).
- **Layer 3:** Dense layer with output size equal to the total sales data shape, using `relu` activation to map to the final prediction.
- **Layer 4:** Reshape layer to match the prediction output with the shape of the target data.

### 1.2 Model Compilation:
- **Loss Function:** `Mean Squared Error (MSE)` is used to measure prediction accuracy.
- **Optimizer:** The model is optimized using the `Adam` optimizer for efficient gradient descent.

```python
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape))
model.add(LSTM(50, activation='relu', return_sequences=False))
model.add(Dense(total_size, activation='relu'))  # total_size is reshaped
model.add(Reshape((y.shape[1], y.shape[2])))  # Match target data shape
model.compile(optimizer='adam', loss='mean_squared_error')
```

---

## 2. Training and Prediction

- **Input Data Shape:** `(30, 3, 475, 120)` 
- **Target Data Shape:** `(30, 475, 120)`
- **Training:** The model is trained for 100 epochs with a batch size of 64, leveraging time-series data from previous periods for predictions.
  
### Prediction Inspection:
- Predicted vs. actual sales are plotted, and predictions for future periods are evaluated. The results include total sales and quantities predicted for the upcoming months.

### Final Remarks

- The model was trained for a long time, with extensive optimization to handle the four-dimensional input data structure, which includes `YearMonth`, `Customer ID`, `Itemcode`, and sales data. The LSTM architecture was specifically tuned to ensure accurate predictions of sales quantities for both specific customers and items, by effectively learning from temporal sales patterns. This approach allowed the model to capture relationships across time, customers, and products, leading to highly reliable forecasts for future sales quantities.
---
