import argparse
import json
import os


NUMERIC_TASKS = {
    "all_global_pop_task",
    "all_global_gdp_task",
    "all_global_carbon_task",
    "all_house_price_task",
    "US_bachelor_ratio_task",
    "US_crime_violent_task",
}


def resolve_input_file(task_name: str) -> str:
    if task_name not in NUMERIC_TASKS:
        raise ValueError(f"Unsupported task_name: {task_name}")
    return f"../../CityLens_data/UrbanSensing_data/{task_name}.json"


def load_dataset(task_name: str, start: int, end: int):
    input_file = resolve_input_file(task_name)
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal packaged batch runner for CityLens-style tasks.")
    parser.add_argument("--task", default="all_global_gdp_task")
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--end", type=int, default=60)
    parser.add_argument("--output-dir", default="../outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_dataset(args.task, args.start, args.end)

    print("[UAgent batch runner]")
    print(f"task={args.task}")
    print(f"selected_samples={len(data)}")
    print(f"output_dir={os.path.abspath(args.output_dir)}")
    print("")
    print("This packaged runner is intentionally lightweight.")
    print("Use the original root-level experiment scripts for full benchmark execution and exact paper reproduction.")


if __name__ == "__main__":
    main()
