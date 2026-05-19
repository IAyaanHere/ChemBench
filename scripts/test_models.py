import logging
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from chembench.models.baselines.sklearn_models import (
    GradientBoostingModel,
    LightGBMModel,
    LinearRegressionModel,
    MLPModel,
    RandomForestModel,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info('=' * 70)
    logger.info('Testing Baseline Model Initialization')
    logger.info('=' * 70)

    models = [
        LinearRegressionModel(),
        RandomForestModel(),
        GradientBoostingModel(),
        LightGBMModel(),
        MLPModel(),
    ]

    logger.info('\n--- Model Configurations ---\n')
    for model in models:
        logger.info('Model: %s', model.model_name)
        logger.info('  Task Type: %s (auto-detect on fit)', model.task_type)
        logger.info('  Is Fitted: %s', model.is_fitted)
        logger.info('  Representation: %s\n', repr(model))

    logger.info('=' * 70)
    logger.info('All models initialized successfully!')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
