"""Run the evaluation benchmark and print results."""
from src.evaluation.benchmark import run_benchmark
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Running evaluation benchmark...")
    results = run_benchmark()
    print("Evaluation results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
