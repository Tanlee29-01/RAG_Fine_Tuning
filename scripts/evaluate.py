"""Run the evaluation benchmark and print results."""
import sys
from src.evaluation.benchmark import run_benchmark
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Running evaluation benchmark...")
    try:
        results = run_benchmark()
    except Exception as exc:
        logger.error("Benchmark failed: %s", exc)
        sys.exit(1)

    if not results:
        logger.warning("Benchmark returned no results.")
        return

    print("Evaluation results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
