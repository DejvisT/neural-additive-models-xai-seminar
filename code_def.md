# Contains explnations  to some part of the NAM_train.py code  $\hat{y}$ that causes  confusion

### num_splits
It refers to the function $\{split_training_dataset}$ function 
Which depends on the sklearn function:

```python
import numpy as np
from sklearn.model_selection import ShuffleSplit
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
y = np.array([1, 2, 1, 2, 1, 2])
rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
rs.get_n_splits(X)
5
print(rs)
ShuffleSplit(n_splits=5, random_state=0, test_size=0.25, train_size=None)
for i, (train_index, test_index) in enumerate(rs.split(X)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
Fold 0:
  Train: index=[1 3 0 4]
  Test:  index=[5 2]
Fold 1:
  Train: index=[4 0 2 5]
  Test:  index=[1 3]
Fold 2:
  Train: index=[1 2 4 0]
  Test:  index=[3 5]
Fold 3:
  Train: index=[3 4 1 0]
  Test:  index=[5 2]
Fold 4:
  Train: index=[3 5 1 0]
  Test:  index=[2 4]


```

#### chatgpt related explanation
<img width="728" height="626" alt="image" src="https://github.com/user-attachments/assets/e3b906af-c93b-4d81-b243-26c75461a172" /><br>
This means we two stages:<br><br>
<strong> 1.random shuffle of the 4 folds <br>
2.split the 4 folds into 80,20 train and validation.
</strong><br>
So that while training we get to choose,the validation dataset using data_split argument.

