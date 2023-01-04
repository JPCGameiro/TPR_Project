# TPR_Project
SSH Attacks

# Project Files
1. Commands to create the metrics, windows and features are in commands.txt file
2. NewProfileClass.py and ProfileClass.py provide the two approaches for testing the anomaly detection
3. Files for the best PCA include bestPCA.py and ProfileClassFunctions.py
4. Files for the best threshold include bestThreshold.py and ProfileClassFunctions.py
5. Files for the ensemble test include the testEnsemble.py and Ensemble.py


# How to install

1. Create a virtual environment (venv)
```bash
python3 -m venv venv
```

2. Activate the virtual environment (you need to repeat this step, and this step only, every time you start a new terminal/session):
```bash
source venv/bin/activate
```

3. Install the game requirements:
```bash
pip3 install -r requirements.txt
```

# Run the project

1. Run the the anomaly detection
```bash
python3 NewProfileClass.py
python3 ProfileClass.py
```

2. Run the the bestPCA
```bash
python3 bestPCA.py
```

3. Run the the bestThreshold
```bash
python3 bestThreshold.py
```

4. Run the the ensemble
```bash
python3 testEnsemble.py
```





