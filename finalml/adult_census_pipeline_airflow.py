"""
TFX Pipeline for Adult Census Dataset — Airflow DAG runner
"""
import datetime
import os

import absl
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen,
    ExampleValidator, Transform, Trainer, Evaluator, Pusher
)
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import metadata, pipeline
from tfx.proto import pusher_pb2, trainer_pb2, example_gen_pb2
from tfx.v1.dsl import Resolver
from tfx.v1.dsl.experimental import LatestBlessedModelStrategy
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner, AirflowPipelineConfig

airflow_config = {
    "schedule_interval": None,
    "start_date": datetime.datetime(2020, 4, 17),
}

PIPELINE_NAME    = "adult_census_pipeline_v2"
PIPELINE_ROOT    = os.path.join(os.path.expanduser('~'), 'COMP315', 'airflow_pipeline_outputs', 'adult_census_tfx_output')
DATA_ROOT        = '/home/manjula/airflow/dags/315_team_project/data/Dataset1_adult/source'
METADATA_PATH    = os.path.join(os.path.expanduser('~'), 'COMP315', 'airflow_pipeline_outputs', 'metadata', 'adult_census_pipeline_v2', 'metadata.db')
SERVING_MODEL_DIR= os.path.join(os.path.expanduser('~'), 'COMP315', 'airflow_pipeline_outputs', 'tfx_output', 'serving_model', 'adult_census_pipeline_v2')

_THIS_DIR        = os.path.dirname(os.path.abspath(__file__))
TRANSFORM_MODULE = os.path.join(_THIS_DIR, 'transform_module.py')
TRAINER_MODULE   = os.path.join(_THIS_DIR, 'trainer_module.py')

TRAIN_NUM_STEPS  = 1000
EVAL_NUM_STEPS   = 300
LABEL_KEY        = 'target'


def create_pipeline(
    pipeline_name, pipeline_root, data_root,
    transform_module, trainer_module,
    metadata_path, serving_model_dir,
    train_num_steps, eval_num_steps,
    enable_cache=True,
):
    # 1. Ingest CSV data
    #80:20 split using hash buckets to ensure deterministic splits across runs
    example_gen = CsvExampleGen(
        input_base=data_root,
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2),
            ])
        )
    )

    # 2. Compute statistics from RAW examples
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # 3. Infer schema
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True,
    )

    # 4. Validate examples
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'],
    )

    # 5. Feature engineering on RAW examples
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=transform_module,
    )

    # 6. Train on TRANSFORMED examples
    trainer = Trainer(
        module_file=trainer_module,
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=train_num_steps),
        eval_args=trainer_pb2.EvalArgs(num_steps=eval_num_steps),
    )

    # 7. Resolve latest blessed model
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id('latest_blessed_model_resolver')

    # 8. Evaluate on TRANSFORMED examples so labels are already float 0/1
    eval_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                label_key=LABEL_KEY,
                signature_name='serving_default',
                prediction_key='outputs'
            )
        ],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(feature_keys=['sex']),
            tfma.SlicingSpec(feature_keys=['race']),
        ],
        metrics_specs=[
            tfma.MetricsSpec(metrics=[
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='Precision'),
                tfma.MetricConfig(class_name='Recall'),
                tfma.MetricConfig(
                    class_name='BinaryAccuracy',
                    threshold=tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={'value': 0.80}
                        ),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={'value': -0.01}
                        ),
                    ),
                ),
            ])
        ],
    )

    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],  
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config=eval_config,
    )

    # 9. Push blessed model
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen, statistics_gen, schema_gen,
            example_validator, transform, trainer,
            model_resolver, evaluator, pusher,
        ],
        enable_cache=enable_cache,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
    )


absl.logging.set_verbosity(absl.logging.INFO)

DAG = AirflowDagRunner(
    AirflowPipelineConfig(airflow_dag_config=airflow_config)
).run(
    create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        transform_module=TRANSFORM_MODULE,
        trainer_module=TRAINER_MODULE,
        metadata_path=METADATA_PATH,
        serving_model_dir=SERVING_MODEL_DIR,
        train_num_steps=TRAIN_NUM_STEPS,
        eval_num_steps=EVAL_NUM_STEPS,
    )
)