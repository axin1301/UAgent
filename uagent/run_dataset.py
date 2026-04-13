import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import tqdm

from func_sup import (
    LLM,
    apply_stv_prefilter,
    filter_requirements_by_modalities,
    filter_tool_list_text,
    make_image_registry,
    run_analysis_agent,
    run_closed_loop_pipeline,
    run_conclusion_agent,
    run_init_agent,
    run_question_spec_agent,
    run_reasoning_agent,
    run_task_router,
    run_tool_shortlister_satellite,
    run_tool_shortlister_street_view,
)
from tool_list_short import TOOL_LIST
from config import DEFAULT_END, DEFAULT_START, DEFAULT_TASK


AVAILABLE_SAT_TOOL_NAMES = [
    "Satellite Image Automatic Description Generator",
    "Satellite Image Object Detection Tool",
    "Area Estimator",
    "Building Footprint Extractor",
    "Building Height Extractor",
    "Road Network Extractor",
    "Satellite Image Land Use Inference Tool",
    "Structure Layout Analyzer",
    "Special Target Recognizer",
    "Satellite Image Landmark Extraction Tool",
    "Satellite Image Geo-Region Localizer",
]

AVAILABLE_STV_TOOL_NAMES = [
    "Street Object Detector",
    "Street View Semantic Segmentation Tool",
    "Building Facade Extractor",
    "Text Sign OCR",
    "Building Type Cue Detector",
    "Street View Image Captioner",
    "Pedestrian Density Estimator",
    "Commercial Clue Extractor",
    "Street View Ground Level Detail Recognizer",
]


def _to_jsonable(obj, max_str=20000):
    try:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            if isinstance(obj, str) and len(obj) > max_str:
                return obj[:max_str] + f"...(truncated,len={len(obj)})"
            return obj
        if isinstance(obj, dict):
            return {str(k): _to_jsonable(v, max_str=max_str) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_jsonable(v, max_str=max_str) for v in obj]
        if hasattr(obj, "model_dump"):
            return _to_jsonable(obj.model_dump(), max_str=max_str)
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            return _to_jsonable(obj.dict(), max_str=max_str)
        s = repr(obj)
        if len(s) > max_str:
            s = s[:max_str] + f"...(truncated,len={len(s)})"
        return {"__repr__": s, "__type__": type(obj).__name__}
    except Exception:
        return {"__repr__": "<unserializable>", "__type__": type(obj).__name__}


def safe_step(traj, step_name, func, *args, **kwargs):
    import time
    import traceback

    t0 = time.time()
    try:
        out = func(*args, **kwargs)
        traj.append(
            {
                "step": step_name,
                "status": "ok",
                "time_sec": round(time.time() - t0, 4),
                "output": _to_jsonable(out),
            }
        )
        return out
    except Exception as e:
        traj.append(
            {
                "step": step_name,
                "status": "error",
                "time_sec": round(time.time() - t0, 4),
                "output": None,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        raise


def resolve_image_paths(image_refs, image_dir):
    resolved = []
    for ref in image_refs:
        if os.path.isabs(ref):
            resolved.append(ref)
        else:
            resolved.append(os.path.join(image_dir, ref))
    return resolved


def get_question(item):
    if "prompt" in item:
        return item["prompt"]
    if "text" in item:
        return item["text"]
    raise ValueError("Each sample must contain either 'prompt' or 'text'.")


def process_one_item(task):
    idx, item, image_dir = task
    item = dict(item)
    traj = []

    try:
        images = safe_step(traj, "resolve_images", resolve_image_paths, item["images"], image_dir)
        question = safe_step(traj, "load_question", get_question, item)
        registry = safe_step(traj, "make_registry", make_image_registry, images)
        alias_list = registry["aliases"]

        image_init = safe_step(traj, "init_agent", run_init_agent, question, alias_list, registry)
        q_spec = safe_step(traj, "question_spec_agent", run_question_spec_agent, question, image_init)
        init_output = {**image_init, **q_spec}
        init_output = safe_step(traj, "stv_prefilter", apply_stv_prefilter, init_output, 5)

        analysis_output = safe_step(
            traj,
            "analysis_agent",
            run_analysis_agent,
            init_output["normalized_question"],
            init_output["answer_spec"],
            init_output["image_roles"],
        )
        analysis_output = safe_step(
            traj,
            "filter_requirements",
            filter_requirements_by_modalities,
            analysis_output,
            init_output["image_roles"],
        )

        task_route = safe_step(
            traj,
            "task_router",
            run_task_router,
            analysis_output,
            init_output["image_roles"],
            [],
            LLM,
        )

        sat_shortlist = safe_step(
            traj,
            "sat_shortlist",
            run_tool_shortlister_satellite,
            task_route,
            analysis_output,
            TOOL_LIST,
            AVAILABLE_SAT_TOOL_NAMES,
            init_output["image_roles"],
            [],
            4,
            LLM,
        )
        stv_shortlist = safe_step(
            traj,
            "stv_shortlist",
            run_tool_shortlister_street_view,
            task_route,
            analysis_output,
            TOOL_LIST,
            AVAILABLE_STV_TOOL_NAMES,
            init_output["image_roles"],
            [],
            4,
            LLM,
        )

        allowed_tools = sat_shortlist["shortlist"] + stv_shortlist["shortlist"]
        filtered_tool_list = safe_step(
            traj,
            "filtered_tool_list",
            filter_tool_list_text,
            TOOL_LIST,
            allowed_tools,
        )

        image_paths = {}
        if init_output["image_roles"].get("satellite"):
            image_paths["satellite"] = init_output["image_roles"]["satellite"]
        if init_output["image_roles"].get("street_view"):
            image_paths["street_view"] = init_output["image_roles"]["street_view"]

        bundle = safe_step(
            traj,
            "closed_loop_pipeline",
            run_closed_loop_pipeline,
            init_output,
            analysis_output,
            filtered_tool_list,
            image_paths,
            1,
            None,
            None,
            registry,
        )

        reasoning_output = safe_step(
            traj,
            "reasoning_agent",
            run_reasoning_agent,
            init_output["normalized_question"],
            init_output["answer_spec"],
            analysis_output,
            bundle["urban_state"],
            bundle["reflection_output"],
        )
        final_answer = safe_step(
            traj,
            "conclusion_agent",
            run_conclusion_agent,
            init_output["answer_spec"],
            reasoning_output,
        )

        item["prediction"] = final_answer
        item["traj"] = traj
        return idx, item
    except Exception as e:
        item["prediction"] = None
        item["traj"] = traj
        item["_error"] = {"type": type(e).__name__, "message": str(e)}
        return idx, item


def load_json_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples.")
    return data


def infer_image_dir(input_json, image_dir):
    if image_dir:
        return image_dir
    base_dir = os.path.dirname(os.path.abspath(input_json))
    return os.path.join(base_dir, "images")


def infer_output_json(input_json, output_json):
    if output_json:
        return output_json
    base_dir = os.path.dirname(os.path.abspath(input_json))
    stem = os.path.splitext(os.path.basename(input_json))[0]
    return os.path.join(base_dir, f"{stem}_uagent_predictions.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the packaged UAgent pipeline on a JSON dataset.")
    parser.add_argument("--input-json", required=True, help="Path to a JSON file containing a list of samples.")
    parser.add_argument("--image-dir", default=None, help="Directory containing referenced image files. Defaults to a sibling 'images/' directory next to the input JSON.")
    parser.add_argument("--output-json", default=None, help="Where to save predictions and traces. Defaults to '<input_name>_uagent_predictions.json' next to the input JSON.")
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--end", type=int, default=DEFAULT_END)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-name", default=DEFAULT_TASK, help="Optional descriptive task name for logs only.")
    args = parser.parse_args()

    image_dir = infer_image_dir(args.input_json, args.image_dir)
    output_json = infer_output_json(args.input_json, args.output_json)

    print(f"[UAgent dataset runner] task_name={args.task_name}", flush=True)
    print(f"input_json={args.input_json}", flush=True)
    print(f"image_dir={image_dir}", flush=True)
    print(f"output_json={output_json}", flush=True)
    print(f"slice=({args.start}, {args.end}) workers={args.workers}", flush=True)

    data = load_json_samples(args.input_json)
    data = data[args.start:args.end]
    tasks = [(i, data[i], image_dir) for i in range(len(data))]
    results = [None] * len(data)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one_item, task) for task in tasks]
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
            idx, updated = fut.result()
            results[idx] = updated

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {os.path.abspath(output_json)}", flush=True)


if __name__ == "__main__":
    main()
