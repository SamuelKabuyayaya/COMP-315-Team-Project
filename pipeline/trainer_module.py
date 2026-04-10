import os
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

# feature columns used by the model
FEATURE_KEYS = [
    'age',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country'
]

# label column
LABEL_KEY = 'target'


def _combine_transformed_features(features):
    # stack transformed features into one dense input tensor
    flattened_features = []

    for key in FEATURE_KEYS:
        feature_value = features[key]

        # turn sparse into dense
        if isinstance(feature_value, tf.SparseTensor):
            feature_value = tf.sparse.to_dense(feature_value)

        # make everything float32
        feature_value = tf.cast(feature_value, tf.float32)
        feature_value = tf.reshape(feature_value, [-1, 1])
        flattened_features.append(feature_value)

    return tf.concat(flattened_features, axis=1)


def _input_fn(file_pattern, tf_transform_output, batch_size=32):
    """reads transformed records and returns features with labels"""

    if isinstance(file_pattern, list):
        file_pattern = file_pattern[0]

    base_dir = file_pattern.rstrip('*').rstrip('/')
    filenames = tf.io.gfile.glob(os.path.join(base_dir, '*.gz'))

    if not filenames:
        filenames = tf.io.gfile.glob(os.path.join(base_dir, '*', '*.gz'))

    transformed_feature_spec = tf_transform_output.transformed_feature_spec()

    dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    dataset = dataset.repeat().shuffle(1000)
    dataset = dataset.map(
        lambda x: tf.io.parse_single_example(x, transformed_feature_spec)
    )
    dataset = dataset.batch(batch_size)

    def extract_inputs_and_label(features):
        label = features.pop(LABEL_KEY)

        if isinstance(label, tf.SparseTensor):
            label = tf.sparse.to_dense(label)

        label = tf.cast(tf.reshape(label, [-1]), tf.float32)
        combined_features = _combine_transformed_features(features)
        return combined_features, label

    return dataset.map(extract_inputs_and_label).prefetch(tf.data.AUTOTUNE)


def _get_tf_examples_serving_signature(model, tf_transform_output):
    """serving signature that accepts raw tf.Examples"""

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec().copy()

        # label is not available at serving time
        raw_feature_spec.pop(LABEL_KEY, None)

        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)

        # apply transform graph
        transformed_features = model.tft_layer(raw_features)
        combined_features = _combine_transformed_features(transformed_features)

        outputs = model(combined_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transformed_labels_signature(model, tf_transform_output):
    """preprocessing fn for tfma to get numeric labels"""

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transformed_labels_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)

        # apply same transform graph used in training
        transformed_features = model.tft_layer(raw_features)
        labels = transformed_features[LABEL_KEY]

        if isinstance(labels, tf.SparseTensor):
            labels = tf.sparse.to_dense(labels)

        labels = tf.cast(tf.reshape(labels, [-1]), tf.float32)

        # override raw label during tfma evaluation
        return {LABEL_KEY: labels}

    return transformed_labels_fn


def _export_serving_model(tf_transform_output, model, output_dir):
    """exports a savedmodel with tfma-friendly signatures"""

    # keep transform layer tracked by the model
    model.tft_layer = tf_transform_output.transform_features_layer()

    signatures = {
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            _get_tf_examples_serving_signature(model, tf_transform_output),
        'transformed_labels':
            _get_transformed_labels_signature(model, tf_transform_output),
    }

    tf.saved_model.save(model, output_dir, signatures=signatures)


def run_fn(args: FnArgs):
    # load transform output
    tf_transform_output = tft.TFTransformOutput(args.transform_output)

    # build train and eval datasets
    train_dataset = _input_fn(args.train_files, tf_transform_output, batch_size=64)
    eval_dataset = _input_fn(args.eval_files, tf_transform_output, batch_size=64)

    # build simple binary classifier
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_KEYS),), name='inputs'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    # send logs to tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.model_run_dir,
        update_freq='batch'
    )

    train_steps = args.train_steps if args.train_steps else 100
    eval_steps = args.eval_steps if args.eval_steps else 50

    # train model
    model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=eval_dataset,
        validation_steps=eval_steps,
        epochs=5,
        callbacks=[tensorboard_callback]
    )

    # export model for serving and tfma
    _export_serving_model(tf_transform_output, model, args.serving_model_dir)