# Stock Price Prediction

This project leverages **Machine Learning** and **Deep Learning** techniques to predict and visualize stock prices.  
It includes both a **web application** and a **Jupyter Notebook** that enable experimentation with stock market trends using trained **Keras** models.

---

## Features

- Fetches real-time or historical stock data using APIs such as `yfinance`
- Predicts future stock trends using pre-trained deep learning models
- Provides interactive and dynamic data visualizations
- Offers a user-friendly web interface (Flask or Streamlit)
- Supports model retraining and experimentation through a Jupyter Notebook

---

## Project Structure

```
Stock-Price-Prediction/
│
├── app.py                           # Main application script
├── stock_market_prediction_us.ipynb # Research and model training notebook
├── Stock Prediction.keras            # Pre-trained Keras model
├── Stock_Prediction_new.keras        # Updated Keras model
├── requirements.txt                  # Project dependencies
└── prevcode.txt                      # Previous reference code and notes
```

---

## Installation

Follow the steps below to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/hvmt2003/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
```

Activate the virtual environment:

**For Windows:**
```bash
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the application based on the framework used:

**If the application uses Streamlit:**
```bash
streamlit run app.py
```

**If the application uses Flask:**
```bash
python app.py
```

After running, open the local URL displayed in the terminal (commonly `http://127.0.0.1:5000`) to access the web interface.

---

## Model Information

The project utilizes deep learning models developed using **Keras** and **TensorFlow** to forecast future stock prices based on historical data.

**Available Models:**
- `Stock Prediction.keras`
- `Stock_Prediction_new.keras`

**Typical Workflow:**
1. Fetch historical stock data using `yfinance` or another API  
2. Preprocess the data (scaling and feature engineering)  
3. Prepare time-series sequences for the model  
4. Train or load pre-trained models  
5. Generate and visualize predictions  

All model training and analysis can be performed in the Jupyter notebook `stock_market_prediction_us.ipynb`.

---

## Configuration

If your setup requires API keys or environment variables, create a `.env` file in the root directory:

```
API_KEY=your_api_key_here
```

Ensure the `.env` file is added to `.gitignore` to prevent sensitive information from being committed.

---

## Troubleshooting

| Issue | Possible Cause | Solution |
|--------|----------------|-----------|
| `ModuleNotFoundError` | Missing dependency | Run `pip install -r requirements.txt` |
| App not launching | Incorrect framework command | Try both `streamlit run app.py` and `python app.py` |
| No data displayed | Invalid ticker or network issue | Verify stock symbol and internet connection |
| Shape or model errors | Data preprocessing mismatch | Check scaler parameters and input sequence length |

---

## Contributing

Contributions are welcome. You may:
- Improve documentation
- Add support for new data sources
- Introduce new ML or DL models (e.g., GRU, Transformer, LSTM variants)
- Enhance UI or visualization features

To contribute:
1. Fork the repository  
2. Create a new branch  
3. Commit and push your changes  
4. Open a Pull Request for review  

---

## License

This project is distributed under the MIT License.  
You may freely modify and distribute the software with appropriate attribution.

---

## Acknowledgements

Developed using open-source technologies including:
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Streamlit / Flask**
- **yfinance**

Special thanks to the open-source community for continuous innovation and learning resources.

---

## Author

**Harshvardhan Mani Tripathi**  
[GitHub Profile](https://github.com/hvmt2003)

For queries, collaborations, or suggestions, feel free to raise an issue or pull request in this repository.
