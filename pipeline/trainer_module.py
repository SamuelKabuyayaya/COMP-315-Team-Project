import os
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

# Keys for the Adult Census dataset
FEATURE_KEYS = [
    'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]
LABEL_KEY = 'target'

def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Generates features and labels for training/eval."""
    if isinstance(file_pattern, list):
        file_pattern = file_pattern[0]
    
    base_dir = file_pattern.rstrip('*').rstrip('/')
    filenames = tf.io.gfile.glob(os.path.join(base_dir, '*.gz'))
    
    if not filenames:
        filenames = tf.io.gfile.glob(os.path.join(base_dir, '*', '*.gz'))

    transformed_feature_spec = tf_transform_output.transformed_feature_spec()

    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    dataset = dataset.repeat().shuffle(1000)
    dataset = dataset.map(lambda x: tf.io.parse_single_example(x, transformed_feature_spec))
    dataset = dataset.batch(batch_size)

    def extract_inputs_and_label(features):
        label = tf.cast(features.pop(LABEL_KEY), tf.float32)
        flattened_features = []
        for key in FEATURE_KEYS:
            f = features[key]
            if isinstance(f, tf.SparseTensor):
                f = tf.sparse.to_dense(f)
            f = tf.cast(f, tf.float32)
            f = tf.reshape(f, [-1, 1])
            flattened_features.append(f)
        return tf.concat(flattened_features, axis=1), label

    return dataset.map(extract_inputs_and_label).prefetch(tf.data.AUTOTUNE)

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""
    
    # We get the spec of already transformed features
    transformed_feature_spec = tf_transform_output.transformed_feature_spec()
    
    # We remove the label from the spec as it won't be provided during inference
    if LABEL_KEY in transformed_feature_spec:
        transformed_feature_spec.pop(LABEL_KEY)

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in serving using transformed features."""
        # Parsing the incoming examples using the transformed spec (numeric/IDs)
        parsed_features = tf.io.parse_example(serialized_tf_examples, transformed_feature_spec)
        
        # Formatting features for the Keras model input layer
        flattened_features = []
        for key in FEATURE_KEYS:
            f = parsed_features[key]
            if isinstance(f, tf.SparseTensor):
                f = tf.sparse.to_dense(f)
            f = tf.cast(f, tf.float32)
            f = tf.reshape(f, [-1, 1])
            flattened_features.append(f)
            
        return model(tf.concat(flattened_features, axis=1))

    return serve_tf_examples_fn

def run_fn(args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(args.transform_output)
    
    train_dataset = _input_fn(args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = _input_fn(args.eval_files, tf_transform_output, batch_size=64)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_KEYS),), name='inputs'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['BinaryAccuracy']
    )
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.model_run_dir, update_freq='batch'
    )

    train_steps = args.train_steps if args.train_steps else 100
    eval_steps = args.eval_steps if args.eval_steps else 50

    model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=eval_dataset,
        validation_steps=eval_steps,
        epochs=5,
        callbacks=[tensorboard_callback]
    )

    # Creating the signature that expects transformed examples (matching Evaluator output)
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }
    
    model.save(args.serving_model_dir, save_format='tf', signatures=signatures)