import tensorflow as tf
import tensorflow_transform as tft

# Defining feature groups for processing
NUMERIC_FEATURE_KEYS = [
    'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
]
BUCKETIZED_FEATURE_KEYS = ['age']
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

    # Bucketize age into 5 life-stage groups: 0-20, 21-40, 41-60, 61-80, 81+
    outputs['age'] = tft.bucketize(inputs['age'], num_buckets=5)

    # Converting categorical strings to integer IDs using a generated vocabulary
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[key] = tft.compute_and_apply_vocabulary(inputs[key])
    
    # Converting the label string to a float (0.0 or 1.0) for the Evaluator
    label = tf.strings.strip(inputs[LABEL_KEY])
    outputs[LABEL_KEY] = tf.where(
        tf.strings.regex_full_match(label, r'.*>50K.*'),
        tf.cast(1.0, tf.float32), 
        tf.cast(0.0, tf.float32)
    )

    return outputs