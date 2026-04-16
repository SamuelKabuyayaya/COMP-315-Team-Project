import tensorflow as tf
import tensorflow_transform as tft

# feature groups used in preprocessing
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

    # scale numeric features with z score
    for key in NUMERIC_FEATURE_KEYS:
        outputs[key] = tft.scale_to_z_score(inputs[key])

    # map categorical strings to vocab ids
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[key] = tft.compute_and_apply_vocabulary(inputs[key])
    
    # map label string to float for evaluator
    label = tf.strings.strip(inputs[LABEL_KEY])
    outputs[LABEL_KEY] = tf.where(
        tf.equal(label, '>50K'), 
        tf.cast(1.0, tf.float32), 
        tf.cast(0.0, tf.float32)
    )

    return outputs
