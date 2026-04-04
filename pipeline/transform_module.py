import tensorflow as tf
import tensorflow_transform as tft

# Feature keys for the Adult Census Dataset
NUMERIC_FEATURE_KEYS = [
    'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
]

CATEGORICAL_FEATURE_KEYS = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]

LABEL_KEY = 'target'

def preprocessing_fn(inputs):
  
    outputs = {}

    # Scaling numeric features to z-scores (mean=0, std=1)
    for key in NUMERIC_FEATURE_KEYS:
        outputs[key] = tft.scale_to_z_score(inputs[key])

    # Converting categorical strings to integer indices
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[key] = tft.compute_and_apply_vocabulary(inputs[key])

    # Transforming the label (target column)
    outputs[LABEL_KEY] = tft.compute_and_apply_vocabulary(inputs[LABEL_KEY])

    return outputs
