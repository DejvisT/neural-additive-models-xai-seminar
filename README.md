# neural-additive-models-xai-seminar
---
## Installation Steps

### 1. Install requirements
Make sure you have the basic dependencies installed:

```bash
pip install -r requirements.txt
```

### 2. Clone the repository

```bash
git clone https://github.com/agarwl/google-research.git
cd google-research/neural_additive_models
```

### 3. Fix the sklearn dependency

Open the file `setup.py` inside `neural_additive_models` folder and find the line:
```bash
install_requires = [
    'tensorflow>=1.15',
    'numpy>=1.15.2',
    'sklearn',
    'pandas>=0.24',
    'absl-py',
]
```

Change it to:
```bash
install_requires = [
    'tensorflow>=1.15',
    'numpy>=1.15.2',
    'scikit-learn',
    'pandas>=0.24',
    'absl-py',
]
```

### 4. Install the package
From inside the `neural_additive_models` directory, run:
```bash
pip install -e .
```
