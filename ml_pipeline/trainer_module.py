import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

FEATURE_KEYS = [
    'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week',
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]
LABEL_KEY = 'target'


def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Reads transformed examples and returns (concatenated_tensor, label)."""
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=lambda f: tf.data.TFRecordDataset(f, compression_type='GZIP'),
        label_key=LABEL_KEY,
        num_epochs=None,
    )

    def concat_features(features, label):
        flattened = []
        for key in FEATURE_KEYS:
            f = features[key]
            if isinstance(f, tf.SparseTensor):
                f = tf.sparse.to_dense(f)
            f = tf.cast(f, tf.float32)
            f = tf.reshape(f, [-1, 1])
            flattened.append(f)
        return tf.concat(flattened, axis=1), tf.cast(label, tf.float32)

    return dataset.map(concat_features).prefetch(tf.data.AUTOTUNE)


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Serving signature that accepts RAW examples.

    The Evaluator now feeds raw_examples,
    so we parse using raw_feature_spec and apply tft_layer.
    """
    # Use raw_feature_spec to handle raw strings from the dataset
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    # We don't need the label for inference
    raw_feature_spec.pop(LABEL_KEY, None)
    
    # Load the transform graph to apply it during inference
    tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        # 1. Parse raw serialized ASCII strings
        parsed_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        
        # 2. Apply the TFX Transform graph (this handles strings -> ints, scaling, etc.)
        transformed_features = tft_layer(parsed_features)

        # 3. Prepare the input tensor for the model (keeping your concatenation logic)
        flattened = []
        for key in FEATURE_KEYS:
            # Note: transform_module.py does NOT add _xf, so we use original keys
            f = transformed_features[key]
            if isinstance(f, tf.SparseTensor):
                f = tf.sparse.to_dense(f)
            f = tf.cast(f, tf.float32)
            f = tf.reshape(f, [-1, 1])
            flattened.append(f)

        outputs = model(tf.concat(flattened, axis=1))
        return {'outputs': outputs}

    return serve_tf_examples_fn


def run_fn(args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(args.transform_output)

    train_dataset = _input_fn(args.train_files, tf_transform_output, batch_size=64)
    eval_dataset  = _input_fn(args.eval_files,  tf_transform_output, batch_size=64)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_KEYS),), name='inputs'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1,  activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.model_run_dir, update_freq='batch'
    )

    train_steps = args.train_steps if args.train_steps else 100
    eval_steps  = args.eval_steps  if args.eval_steps  else 50

    model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=eval_dataset,
        validation_steps=eval_steps,
        epochs=5,
        callbacks=[tensorboard_callback]
    )

    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        )
    }

    model.save(args.serving_model_dir, save_format='tf', signatures=signatures)