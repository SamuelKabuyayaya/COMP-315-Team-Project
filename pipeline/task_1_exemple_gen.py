import os
import shutil

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)
from tfx.types import Channel, standard_artifacts
from tfx.proto import trainer_pb2
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.local.local_dag_runner import LocalDagRunner

# pipeline paths
PIPELINE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(PIPELINE_DIR)
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'Dataset1_adult', 'source'))
OUTPUT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, 'tfx_pipeline_output_v2'))
TRANSFORM_MODULE = os.path.abspath(os.path.join(PIPELINE_DIR, 'transform_module.py'))
TRAINER_MODULE = os.path.abspath(os.path.join(PIPELINE_DIR, 'trainer_module.py'))
METADATA_PATH = os.path.join(OUTPUT_DIR, 'metadata.db')

# label used for evaluation
LABEL_KEY = 'target'


def _create_eval_config():
    # config for tfma evaluator
    return tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name='serving_default',
                label_key=LABEL_KEY,
                prediction_key='outputs',
                preprocessing_function_names=['transformed_labels']
            )
        ],
        slicing_specs=[
            tfma.SlicingSpec(),  # overall
            tfma.SlicingSpec(feature_keys=['sex']),
            tfma.SlicingSpec(feature_keys=['race'])
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name='BinaryAccuracy',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': 0.5}
                            )
                        )
                    ),
                    tfma.MetricConfig(
                        class_name='AUC',
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={'value': 0.5}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={'value': -1e-10}
                            )
                        )
                    )
                ]
            )
        ]
    )


def create_pipeline(pipeline_name, pipeline_root, data_root, transform_module, trainer_module, metadata_path):
    """builds the full tfx pipeline"""

    # step 1: read csv data
    example_gen = CsvExampleGen(input_base=data_root)

    # step 2: make data stats
    stats_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # step 3: infer schema
    schema_gen = SchemaGen(statistics=stats_gen.outputs['statistics'])

    # step 4: check for anomalies
    validator = ExampleValidator(
        statistics=stats_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )

    # step 5: transform features
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module
    )

    # step 6: train model
    trainer = Trainer(
        module_file=trainer_module,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50)
    )

    # step 7: get latest blessed model
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=standard_artifacts.Model),
        model_blessing=Channel(type=standard_artifacts.ModelBlessing)
    ).with_id('latest_blessed_model_resolver')

    # step 8: evaluate model with tfma
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=_create_eval_config(),
        schema=schema_gen.outputs['schema']
    )

    # build pipeline
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            stats_gen,
            schema_gen,
            validator,
            transform,
            trainer,
            model_resolver,
            evaluator
        ],
        enable_cache=False,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path)
    )


if __name__ == '__main__':
    # clear old output to avoid lock problems
    if os.path.exists(OUTPUT_DIR):
        print(f"cleaning old output directory: {OUTPUT_DIR}")
        try:
            shutil.rmtree(OUTPUT_DIR)
        except PermissionError:
            print("warning: access denied during cleanup. close anything using the metadata db and try again.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # pipeline name
    unique_pipeline_name = 'adult_census_pipeline_v2'
    print(f"starting tfx pipeline: {unique_pipeline_name}")

    try:
        # run pipeline locally
        LocalDagRunner().run(
            create_pipeline(
                pipeline_name=unique_pipeline_name,
                pipeline_root=OUTPUT_DIR,
                data_root=DATA_ROOT,
                transform_module=TRANSFORM_MODULE,
                trainer_module=TRAINER_MODULE,
                metadata_path=METADATA_PATH
            )
        )
        print("pipeline execution finished successfully.")
    except Exception as e:
        print(f"pipeline execution failed: {e}")