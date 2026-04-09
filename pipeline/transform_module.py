import tensorflow as tf
import tensorflow_transform as tft

# Defining feature groups for processing
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

    # Scaling numeric features to have mean 0 and variance 1 (Z-score)
    for key in NUMERIC_FEATURE_KEYS:
        outputs[key] = tft.scale_to_z_score(inputs[key])

    # Converting categorical strings to integer IDs using a generated vocabulary
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[key] = tft.compute_and_apply_vocabulary(inputs[key])
    
    # Mapping the label (target) to integer IDs (0 or 1)
    outputs[LABEL_KEY] = tft.compute_and_apply_vocabulary(inputs[LABEL_KEY])

    return outputs