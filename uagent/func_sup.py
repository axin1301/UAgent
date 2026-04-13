from prompt_list import *
from llm_api import LLM
from tool_function_map import *
from tool_function_map import TOOL_API_MAP
import uuid
import os
from PIL import Image
from tool_function_map import TOOL_API_MAP
available_tool_names = list(TOOL_API_MAP.keys())

import hashlib
import json

def run_no_tool_visual_state(normalized_question, analysis_output, image_roles, registry):
    """
    Minimal urban_state without external tools.
    Only use VL caption to generate coarse cues.
    """

    targets = {}
                # Focus on land use, building density, road layout, greenery, and commercial cues.
    for modality in ["satellite", "street_view"]:
        for img_ref in image_roles.get(modality, []):
            real_path = resolve_image_ref(img_ref, registry)
            alias = registry["path_to_alias"].get(os.path.abspath(real_path), img_ref)

            # ===== 调用你现有的 VL 模型做 caption =====
            caption = VLM(
                real_path,
                prompt=f"""
                Based on the question: {normalized_question}
                Provide a concise description of urban characteristics visible in this image.
                """
            )

            targets[f"{modality}.{alias}"] = {
                "modality": modality,
                "target": alias,
                "features": {
                    "scene_summary": caption,
                    "road_or_layout_summary": "",
                    "building_or_structure_summary": "",
                    "vegetation_or_nature_summary": "",
                    "water_or_open_space_summary": "",
                    "notable_objects_or_landmarks": []
                },
                "evidence": [
                    {
                        "tool_name": "VL Caption (No External Tool)",
                        "purpose": "Direct visual summary without tool decomposition",
                        "key_finding": caption,
                        "output_ref": alias
                    }
                ],
                "uncertainty": []
            }

    return {
        "targets": targets,
        "global_notes": ["No external tool execution. Visual summary only."]
    }

def compress_records(execution_output):
    recs = execution_output.get("records", []) or []
    keep = []
    for r in recs:
        keep.append({
            "modality": r.get("modality"),
            "target": r.get("target"),
            "tool_name": r.get("tool_name"),
            "purpose": r.get("purpose"),
            "input_image_alias": r.get("input_image_alias", r.get("input_image")),
            "output": r.get("output"),
        })
    return {"records": keep}

def resolve_image_ref(img_ref, registry):
    p = registry["alias_to_path"].get(img_ref, img_ref)
    # 如果你看到这种情况，说明你把 repr/json 字符串当路径了
    if "\\\\" in p:
        # 还原：把双反斜杠变单反斜杠（Windows）
        p = p.replace("\\\\", "\\")
    return p

def make_image_registry(image_list):
    """
    image_list: list[str] real paths in the same order as <image> placeholders
    returns registry dict with alias mappings
    """
    alias_to_path = {}
    path_to_alias = {}
    alias_list = []

    for i, p in enumerate(image_list, start=1):
        alias = f"IMG{i:02d}"
        ap = os.path.abspath(p)
        alias_to_path[alias] = ap
        path_to_alias[ap] = alias
        alias_list.append(alias)

    return {
        "alias_to_path": alias_to_path,
        "path_to_alias": path_to_alias,
        "aliases": alias_list
    }

def resolve_image_ref(img_ref, registry):
    """img_ref may be alias or real path; return real absolute path."""
    if registry is None:
        return img_ref
    return registry["alias_to_path"].get(img_ref, os.path.abspath(img_ref))

def register_derived_image(real_path, registry, alias):
    """register a derived image (crop/quadrant) with a new alias."""
    ap = os.path.abspath(real_path)
    registry["alias_to_path"][alias] = ap
    registry["path_to_alias"][ap] = alias
    return alias

def make_image_alias_map(image_list):
    """
    image_list: list[str] real paths (order matches question)
    returns:
      alias_to_path: dict[str,str]
      path_to_alias: dict[str,str]
      alias_list: list[str]
    """
    alias_to_path = {}
    path_to_alias = {}
    alias_list = []

    for i, p in enumerate(image_list, start=1):
        alias = f"IMG{i:02d}"
        ap = os.path.abspath(p)
        alias_to_path[alias] = ap
        path_to_alias[ap] = alias
        alias_list.append(alias)

    return alias_to_path, path_to_alias, alias_list

def safe_format(template: str, **kwargs) -> str:
    """
    Safely format a prompt that contains JSON braces.
    It escapes all braces first, then unescapes the placeholders we actually want.
    """
    escaped = template.replace("{", "{{").replace("}", "}}")
    for k, v in kwargs.items():
        escaped = escaped.replace("{{" + k + "}}", "{" + k + "}")
    return escaped.format(**kwargs)


def split_into_quadrants(image_path, output_folder="split_images"):
    """
    Split a satellite image into four quadrants:
    top_left, top_right, bottom_left, bottom_right.

    Returns:
        dict: {
            "top_left": path,
            "top_right": path,
            "bottom_left": path,
            "bottom_right": path
        }
    """
    os.makedirs(output_folder, exist_ok=True)

    quadrant_paths = {}

    try:
        img = Image.open(image_path)
        width, height = img.size

        half_w = width // 2
        half_h = height // 2

        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)

        # 明确写出每个 quadrant（避免 for-loop 语义错误）
        quadrants = {
            "top_left": (0, 0, half_w, half_h),
            "top_right": (half_w, 0, width, half_h),
            "bottom_left": (0, half_h, half_w, height),
            "bottom_right": (half_w, half_h, width, height),
        }

        for quad_name, (left, upper, right, lower) in quadrants.items():
            cropped_img = img.crop((left, upper, right, lower))
            output_path = os.path.join(
                output_folder,
                f"{name}_{quad_name}{ext}"
            )
            cropped_img.save(output_path)
            quadrant_paths[quad_name] = output_path

        return quadrant_paths

    except Exception as e:
        raise RuntimeError(f"Failed to split image {image_path}: {e}")

def apply_stv_prefilter(init_output, keep_default=5):
    """
    根据 init_agent 输出的 semantics（sat_landuse / stv_scene）做早期 STV 过滤。
    真正缩减 init_output["image_roles"]["street_view"]，从而影响后续 analysis/router/tool/closed-loop。
    """
    roles = init_output.get("image_roles", {}) or {}
    stvs = roles.get("street_view", []) or []
    if not stvs:
        return init_output

    sem = init_output.get("semantics", {}) or {}
    pre = init_output.get("prefilter", {}) or {}

    sat_landuse = sem.get("sat_landuse", "unknown") or "unknown"
    stv_scene = sem.get("stv_scene", {}) or {}

    max_keep = pre.get("max_keep", keep_default)
    try:
        max_keep = int(max_keep)
    except Exception:
        max_keep = keep_default
    max_keep = max(1, min(max_keep, len(stvs)))

    # 一致性规则（你可以后续微调）
    def consistent(sat, stv):
        if sat in ("residential", "commercial"):
            return stv in ("residential", "commercial", "transport")
        if sat == "industrial":
            return stv in ("industrial", "transport")
        if sat == "nature":
            return stv in ("nature", "transport")
        if sat == "transport":
            return stv == "transport"
        # mixed / unknown：不强筛
        return True

    # 先用 LLM 给的 keep（并校验必须是子集）
    keep = (pre.get("stv_keep") or [])
    keep = [a for a in keep if a in stvs]
    keep = keep[:max_keep]

    # 若 LLM 没给 keep 或给得很差：用一致性+置信度打分挑选
    if not keep:
        scored = []
        for a in stvs:
            info = stv_scene.get(a, {}) if isinstance(stv_scene.get(a), dict) else {}
            st = info.get("scene_type", "unknown") or "unknown"
            try:
                conf = float(info.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0

            score = (1.0 if consistent(sat_landuse, st) else 0.0) + 0.01 * conf
            scored.append((score, conf, a, st))

        scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
        keep = [a for _, _, a, _ in scored[:max_keep]]

    # 保底：至少留 1 张
    if not keep:
        keep = [stvs[0]]

    dropped = [a for a in stvs if a not in keep]

    # 真正缩减
    init_output["image_roles"]["street_view"] = keep

    # 记录一下（方便 debug）
    init_output["_prefilter_runtime"] = {
        "sat_landuse": sat_landuse,
        "max_keep": max_keep,
        "kept": keep,
        "dropped": dropped,
    }
    return init_output


def split_and_register_quadrants(sat_alias, registry, output_dir="split_images"):
    """
    sat_alias: e.g. "IMG01"
    return quadrant_aliases: ["IMG01_TL","IMG01_TR","IMG01_BL","IMG01_BR"]
    """
    real_path = resolve_image_ref(sat_alias, registry)
    quad_paths = split_into_quadrants(real_path, output_dir=output_dir)  # 你已有的分割函数改成支持 output_dir

    # 你 split_into_quadrants 返回的路径顺序要固定（建议 TL, TR, BL, BR）
    quad_aliases = []
    suffixes = ["TL", "TR", "BL", "BR"]

    for suf, qp in zip(suffixes, quad_paths):
        qa = f"{sat_alias}_{suf}"
        register_derived_image(qp, registry, qa)
        quad_aliases.append(qa)

    return quad_aliases

answer_spec = '''{
  "answer_type": "single_choice" | "multi_choice" | "numeric" | "free_form" | "boolean",
  "choices": ["A","B","C","D"],            # optional
  "choice_meaning": {                      # optional
    "A": "top_left",
    "B": "top_right",
    "C": "bottom_left",
    "D": "bottom_right"
  },
  "output_constraints": {
    "format": "one_letter_only",           # optional
    "allowed_values": ["A","B","C","D"]    # optional
  }
}'''

# def run_init_agent(question: str, image_list: list):
def run_init_agent(question, image_list, registry=None):
    image_type_results = [classify_image_type(p, registry=registry) for p in image_list]

    prompt = safe_format(
        init_agent_prompt,
        question=question,
        image_list=image_list,
        image_type_results=image_type_results
    )
    out = LLM(prompt)
    print("image_init_output:", out)
    return safe_json_loads(out)


def run_question_spec_agent(question, image_init_output):
    prompt = safe_format(
        question_spec_prompt,
        question=question,
        image_roles=image_init_output["image_roles"],
        preprocess=image_init_output["preprocess"]
    )
    out = LLM(prompt)
    print("question_spec_output:", out)
    return safe_json_loads(out)


def run_analysis_agent(normalized_question: str, answer_spec: dict, image_roles: dict):
    prompt = safe_format(analysis_agent_prompt,
        normalized_question=normalized_question,
        answer_spec=str(answer_spec),
        image_roles=image_roles
    )
    return safe_json_loads(LLM(prompt))

def run_planning_agent(analysis_output, tool_list, image_roles, tool_requests=None):
    if tool_requests is None:
        tool_requests = []
    prompt = safe_format(planning_agent_prompt,
        analysis_output=json.dumps(analysis_output, ensure_ascii=False),
        image_roles=json.dumps(image_roles, ensure_ascii=False),
        tool_list=tool_list,
        tool_requests=json.dumps(tool_requests or [], ensure_ascii=False)
        )
    planning_output = LLM(prompt)
    print('Planning Output: ',planning_output)
    return safe_json_loads(planning_output)



# def _normalize_targets(modality: str, targets, image_paths: dict):
#     """
#     Convert plan targets into concrete image targets.
#     Returns list of (target_name, image_path).
#     """
#     if targets is None:
#         targets = []

#     # Treat ["all"] the same as []
#     if targets == ["all"]:
#         targets = []

#     if modality == "satellite":
#         quad_dict = image_paths.get("satellite_quadrants", {}) or {}
#         if not quad_dict:
#             # If no quadrants, fall back to whole satellite image list
#             sat_list = image_paths.get("satellite", []) or []
#             if not sat_list:
#                 return []
#             # Use a synthetic target name for whole image
#             return [("satellite_full", sat_list[0])]

#         # If targets not specified, run all quadrants
#         if not targets:
#             return list(quad_dict.items())

#         # Else run specified quadrants
#         return [(t, quad_dict[t]) for t in targets if t in quad_dict]

def _normalize_targets(modality: str, targets, image_paths: dict):
    """
    Convert plan targets into concrete image targets.
    Returns list of (target_name, image_path_or_alias).
    """
    if targets is None:
        targets = []

    # Treat ["all"] the same as []
    if targets == ["all"]:
        targets = []

    if modality == "satellite":
        quad_dict = image_paths.get("satellite_quadrants", {}) or {}
        if not quad_dict:
            sat_list = image_paths.get("satellite", []) or []
            if not sat_list:
                return []
            return [("satellite_full", sat_list[0])]

        if not targets:
            return list(quad_dict.items())

        return [(t, quad_dict[t]) for t in targets if t in quad_dict]

    elif modality == "street_view":
        stv_list = image_paths.get("street_view", []) or []
        if not stv_list:
            return []

        # ✅ 1) 默认：targets 为空 → 跑全部 street views
        if not targets:
            return [(f"street_view_{i}", p) for i, p in enumerate(stv_list)]

        # ✅ 2) 支持 planner 写 ["street_view"] / ["stv"] 表示全部
        norm = [str(t).strip().lower() for t in targets]
        if any(t in ["street_view", "stv", "all"] for t in norm):
            return [(f"street_view_{i}", p) for i, p in enumerate(stv_list)]

        # ✅ 3) 支持指定 index：["0","1"] 或 ["stv_0","stv_1"]
        out = []
        for t in targets:
            ts = str(t).strip().lower()
            if ts.startswith("stv_"):
                ts = ts.replace("stv_", "")
            if ts.startswith("street_view_"):
                ts = ts.replace("street_view_", "")
            if ts.isdigit():
                idx = int(ts)
                if 0 <= idx < len(stv_list):
                    out.append((f"street_view_{idx}", stv_list[idx]))

        if out:
            return out

        # ✅ 4) 支持直接指定 alias（例如 IMG02、IMG03）
        alias_set = set(targets)
        out = [(f"street_view_{i}", p) for i, p in enumerate(stv_list) if p in alias_set]
        if out:
            return out

        # fallback：至少跑第一张
        return [("street_view_0", stv_list[0])]

    else:
        return []


def _make_tool_prompt(tool_step: dict):
    """Ask LLM to generate a tool-specific prompt for this step."""
    # prompt = safe_format(execution_agent_prompt,tool_step=tool_step)
    # resp = safe_json_loads(LLM(prompt))
    # return resp["tool_prompt"]
    return tool_step["purpose"]

def _pack_output(tool_out):
    if tool_out is None:
        return {"type": "none"}
    if isinstance(tool_out, (dict, list)):
        return {"type": "json", "data": tool_out}
    return {"type": "text", "text": str(tool_out)}

# def run_execution_agent(planning_output, image_paths, cache_records=None, force_rerun=None):
def run_execution_agent(planning_output, image_paths, cache_records=None, force_rerun=None, registry=None):

    """
    Execute planned tools on image targets, with optional caching.

    Args:
      planning_output: dict or JSON string
      image_paths: dict
      cache_records: list of previous records (optional). If provided, reuse existing results.
      force_rerun: iterable of keys (modality, target, tool_name) that MUST be rerun (optional).

    Returns:
      {"records": [...]}  (same schema as before)
    """

    planning_output = safe_json_loads(planning_output)

    if cache_records is None:
        cache_records = []
    cache_index = _build_cache_index(cache_records)

    if force_rerun is None:
        force_rerun = set()
    else:
        force_rerun = set(force_rerun)

    records_out = []

    def _run_step(modality: str, step: dict):
        tool_name = step["tool_name"]
        tool_fn = TOOL_API_MAP[tool_name]

        targets = step.get("targets", [])
        image_targets = _normalize_targets(modality, targets, image_paths)

        tool_prompt = _make_tool_prompt(step)

        for target_name, img_path in image_targets:
            tool_name_norm = str(tool_name).strip()
            target_norm = str(target_name).strip().lower().replace(" ", "_").replace("-", "_")
            if target_norm in ["all", "full", "whole", "satellite", "satellite_full"]:
                target_norm = "satellite_full"

            k = (modality, target_norm, tool_name_norm)

            run_id = uuid.uuid4().hex[:6]
            print("[PIPELINE RUN]", run_id)
            # ✅ cache hit
            if k in cache_index and k not in force_rerun:
                print("[CACHE HIT]", k)
                rec = cache_index[k]

                # 可选：保证 record 里的 target/tool_name 也一致（避免后续再错）
                rec = dict(rec)
                rec["target"] = target_norm
                rec["tool_name"] = tool_name_norm

                records_out.append(rec)
                continue

            # print("[RUN_exec]", k)
            print("[RUN_exec]",run_id, k)  #

            # ✅ run tool
            real_img_path = resolve_image_ref(img_path, registry)  # ✅ alias -> real path
            try:
                _,tool_out,_,_,_ = tool_fn(real_img_path, tool_prompt)
            except TypeError:
                _,tool_out,_,_,_ = tool_fn(real_img_path)

            rec = {
                "modality": modality,
                "target": target_norm,
                "tool_name": tool_name_norm,
                "purpose": step.get("purpose", ""),
                "prompt": tool_prompt,
                "input_image_alias": img_path,     # ✅ alias
                "input_image": real_img_path,      # ✅ real path
                "output": _pack_output(tool_out)
            }

            records_out.append(rec)
            cache_index[k] = rec

    # Satellite
    for step in planning_output.get("satellite_plan", []) or []:
        if step.get("tool_name") not in available_tool_names:
            continue
        _run_step("satellite", step)

    # Street-view
    for step in planning_output.get("street_view_plan", []) or []:
        # 有时 planner 可能输出空 tool_name 的占位项，直接跳过
        if not isinstance(step, dict): 
            continue
        if step.get("tool_name") not in available_tool_names:
            continue
        if not step.get("tool_name"):
            continue
        _run_step("street_view", step)

    return {"records": records_out}


# def run_state_agent(normalized_question: str, analysis_output: dict, execution_output: dict):
#     prompt = safe_format(state_agent_prompt,
#         normalized_question=normalized_question,
#         analysis_output=analysis_output,
#         execution_output=execution_output
#     )
#     urban_state = LLM(prompt)
#     return urban_state

def run_state_agent(normalized_question: str, analysis_output: dict, execution_output: dict):
    execution_compact = compress_records(execution_output)
    execution_output=json.dumps(execution_compact, ensure_ascii=False)

    prompt = safe_format(
        state_agent_prompt,
        normalized_question=normalized_question,
        analysis_output=json.dumps(analysis_output, ensure_ascii=False),
        execution_output=json.dumps(execution_output, ensure_ascii=False),
    )
    urban_state = LLM(prompt)
    return urban_state

def run_reflection_agent(normalized_question, answer_spec, analysis_output,
                         planning_output, execution_output, urban_state, tool_list):
    
    execution_output_clean = strip_real_paths(execution_output)
    urban_state_clean = strip_real_paths(urban_state)

    prompt = safe_format(reflection_agent_prompt,
        normalized_question=normalized_question,
        answer_spec=str(answer_spec),
        analysis_output=analysis_output,
        planning_output=planning_output,
        execution_output=execution_output_clean, #execution_output,
        urban_state=urban_state_clean, #urban_state,
        tool_list=tool_list
    )    
    return safe_json_loads(LLM(prompt))


# def run_closed_loop_pipeline(
#     init_output,
#     analysis_output,
#     tool_list,
#     image_paths,
#     max_iters=2,
#     cache_records_init=None,
#     planning_output_init=None
# ):
    
#     normalized_question = init_output["normalized_question"]
#     answer_spec = init_output["answer_spec"]

#     planning_output = planning_output_init or run_planning_agent(
#     analysis_output=analysis_output,
#     tool_list=tool_list,
#     image_roles=init_output["image_roles"],
#     tool_requests=[]
# )

#     cache_records = cache_records_init or []   # ✅ 复用外部缓存
#     force_rerun = set()

#     for it in range(max_iters):
#         print("[ITER]", it)

#         execution_output = run_execution_agent(
#             planning_output=planning_output,
#             image_paths=image_paths,
#             cache_records=cache_records,
#             force_rerun=force_rerun
#         )

#         cache_records = execution_output["records"]
#         force_rerun = set()  # 默认清空，除非 reflection 再指定

#         urban_state = run_state_agent(normalized_question, analysis_output, execution_output)
        
#         def should_reflect(analysis_output, urban_state):
#             # 如果 uncertainty 不为空就 reflect
#             if '"uncertainty": [' in urban_state and '"uncertainty": []' not in urban_state:
#                 return True
#             return False
        
#         if should_reflect(analysis_output, urban_state):

#             reflection_output = run_reflection_agent(
#                 normalized_question=normalized_question,
#                 answer_spec=answer_spec,
#                 analysis_output=analysis_output,
#                 planning_output=planning_output,
#                 execution_output=execution_output,
#                 urban_state=urban_state,
#                 tool_list=tool_list
#             )
#         else:
#             reflection_output = {"status": "PASS", "confidence": 0.9, "actions": {"replan_required": False}}

#         last_bundle = {
#             "planning_output": planning_output,
#             "execution_output": execution_output,
#             "urban_state": urban_state,
#             "reflection_output": reflection_output
#         }

#         actions = reflection_output.get("actions", {}) or {}

#         if reflection_output.get("status") == "PASS":
#             break

#         # ❌ 没要求 replan → 不动 planning_output（也不改 force_rerun）
#         if not actions.get("replan_required", False):
#             continue

#         tool_requests = actions.get("tool_requests", []) or []

#         # parse rerun_same_plan into force_rerun keys
#         force_rerun = set()
#         for item in (actions.get("rerun_same_plan", []) or []):
#             modality = item.get("modality")
#             tool_name = item.get("tool_name")
#             for t in (item.get("targets", []) or []):
#                 force_rerun.add((modality, t, tool_name))

#         # ✅ only when replan_required
#         planning_output = run_planning_agent(
#             analysis_output, tool_list, init_output["image_roles"], tool_requests=tool_requests
#         )

#         # next loop will run incrementally; you can pass force_rerun if desired
#         # simplest: store it on the loop variable (if you want)
#         # For now, you can integrate force_rerun into the incremental call.

#     return last_bundle

def run_closed_loop_pipeline(
    init_output,
    analysis_output,
    tool_list,
    image_paths,
    max_iters=2,
    cache_records_init=None,
    planning_output_init=None,
    registry=None
):
    def _count_unique_targets(records):
        return len({(r.get("modality"), r.get("target")) for r in (records or [])})

    def _count_runs(records):
        # 粗略：一条 record 就算一次 run（你现在没有 hit 字段）
        return len(records or [])

    def _issues_by_type(reflection_output):
        issues = reflection_output.get("issues", []) or []
        m = {}
        for it in issues:
            m.setdefault(it.get("type"), []).append(it)
        return m

    def _topk_tool_requests(tool_requests, k=0):
        # k=0 表示直接不允许新增工具请求（最严格、也最省时）
        if k <= 0:
            return []
        # priority 大的优先
        tool_requests = sorted(tool_requests, key=lambda x: x.get("priority", 0), reverse=True)
        return tool_requests[:k]

    def _shrink_targets_to_existing(tool_requests, existing_targets):
        # existing_targets: dict {modality: set(targets)}
        out = []
        for tr in tool_requests:
            mod = tr.get("modality")
            tgt = tr.get("targets", []) or []
            # 把 "all" 变成“只允许已存在 targets”
            if tgt == ["all"] or tgt == ["ALL"] or tgt == ["All"] or tgt == ["*"] or tgt == "all":
                tr = dict(tr)
                tr["targets"] = sorted(list(existing_targets.get(mod, set())))
            else:
                # 显式 targets 也要过滤到已存在集合
                tr = dict(tr)
                tr["targets"] = [t for t in tgt if t in existing_targets.get(mod, set())]
            if tr["targets"]:
                out.append(tr)
        return out

    def _limit_rerun(rerun_same_plan, existing_targets, k=1):
        # 每轮最多 rerun k 个 target（默认 1）
        items = []
        for item in (rerun_same_plan or []):
            mod = item.get("modality")
            tool = item.get("tool_name")
            tgts = item.get("targets", []) or []
            if tgts == ["all"] or tgts == "all":
                tgts = sorted(list(existing_targets.get(mod, set())))
            else:
                tgts = [t for t in tgts if t in existing_targets.get(mod, set())]
            for t in tgts:
                items.append((mod, t, tool))
        # 去重 + 截断
        dedup = []
        seen = set()
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            dedup.append(x)
        return set(dedup[:k])

    normalized_question = init_output["normalized_question"]
    answer_spec = init_output["answer_spec"]
    image_roles = init_output["image_roles"]

    # ✅ 用外部的 planning_output（如果给了），否则自己规划一次
    if planning_output_init is not None:
        planning_output = planning_output_init
    else:
        planning_output = run_planning_agent(
            analysis_output=analysis_output,
            tool_list=tool_list,
            image_roles=image_roles,
            tool_requests=[]
        )
        planning_output = filter_plan_by_image_roles(planning_output, init_output["image_roles"])

    cache_records = cache_records_init or []
    force_rerun = set()
    last_bundle = None

    # =========================
    # ✅【新增】预算参数：你可以按机器/实验节奏调
    # ========================= # xyx
    # MAX_UNIQUE_TARGETS = 6   # 建议：卫星1 + 街景<=5（避免全量10+）
    # MAX_RUNS = 8             # 总工具调用上限（records 条数近似）
    MAX_RUNS = 20
    MAX_UNIQUE_TARGETS = 10


    for it in range(max_iters):
        print("[ITER]", it)

        execution_output = run_execution_agent(
            planning_output=planning_output,
            image_paths=image_paths,
            cache_records=cache_records,
            force_rerun=force_rerun,
            registry=registry,          # ✅ 新增
        )

        # ✅ 用最新 records 作为下一轮缓存（records 里应包含 hit + run）
        cache_records = execution_output["records"]
        force_rerun = set()

        urban_state = run_state_agent(
            normalized_question=normalized_question,
            analysis_output=analysis_output,
            execution_output=execution_output
        )

        # 你已有 should_reflect 就用你那套；这里先保持原样
        reflection_output = run_reflection_agent(
            normalized_question=normalized_question,
            answer_spec=answer_spec,
            analysis_output=analysis_output,
            planning_output=planning_output,
            execution_output=execution_output,
            urban_state=urban_state,
            tool_list=tool_list
        )

        last_bundle = {
            "planning_output": planning_output,
            "execution_output": execution_output,
            "urban_state": urban_state,
            "reflection_output": reflection_output
        }

    #     if reflection_output.get("status") == "PASS":
    #         break

    #     actions = reflection_output.get("actions", {}) or {}
    #     if not actions.get("replan_required", False):
    #         continue

    #     tool_requests = actions.get("tool_requests", []) or []

    #     # parse rerun_same_plan into force_rerun keys
    #     for item in (actions.get("rerun_same_plan", []) or []):
    #         modality = item.get("modality")
    #         tool_name = item.get("tool_name")
    #         for t in (item.get("targets", []) or []):
    #             force_rerun.add((modality, t, tool_name))

    #     # ✅ 只有需要 replan 才重新调用 planner
    #     planning_output = run_planning_agent(
    #         analysis_output=analysis_output,
    #         tool_list=tool_list,
    #         image_roles=image_roles,
    #         tool_requests=tool_requests
    #     )
    #     planning_output = filter_plan_by_image_roles(planning_output, init_output["image_roles"])

    # return last_bundle
    # =========================
        # ✅【关键插入点 #1】在你原来的 PASS 判断之后、replan 之前：
        #    做“预算早停 / 软问题早停 / rerun限额 / 禁止扩张”
        # =========================

        # 原逻辑：PASS 就退出
        if reflection_output.get("status") == "PASS":
            break

        # 1) 预算早停：跑太多就不再进入下一轮（直接返回当前bundle）
        records = execution_output.get("records", []) or []
        unique_targets = _count_unique_targets(records)
        num_runs = _count_runs(records)

        if unique_targets >= MAX_UNIQUE_TARGETS or num_runs >= MAX_RUNS:
            # 不再replan，避免越跑越多图
            reflection_output["status"] = "STOP_EARLY"
            reflection_output["stop_reason"] = {
                "unique_targets": unique_targets,
                "num_runs": num_runs
            }
            last_bundle["reflection_output"] = reflection_output
            break

        # 2) 如果主要问题是 missing_evidence/low_confidence（软问题），直接收敛输出
        issues_map = _issues_by_type(reflection_output)
        hard_issue_types = {"missing_evidence","contradiction", "tool_failure", "inconsistency"}  # 你例子里有这些就算硬
        soft_issue_types = { "low_confidence"}

        has_hard = any(t in issues_map for t in hard_issue_types)
        has_soft = any(t in issues_map for t in soft_issue_types)

        if has_soft and (not has_hard):
            reflection_output["status"] = "PASS_WITH_UNCERTAINTY"
            reflection_output["stop_reason"] = "soft_issues_only"
            last_bundle["reflection_output"] = reflection_output
            break

        # =========================
        # ✅【关键插入点 #2】从这里开始才进入你原来的 replan 流程
        # =========================

        actions = reflection_output.get("actions", {}) or {}
        if not actions.get("replan_required", False):
            continue

        # 3) 禁止新增工具/禁止扩张targets：把 tool_requests 直接清空（最严格、最省时）
        # tool_requests = []  # 你说尽量不增加新内容，就别新增工具 # xyx
        tool_requests = _topk_tool_requests(actions.get("tool_requests", []) or [], k=1)

        # 4) rerun_same_plan 限额：最多 rerun 1 个 target（而不是 all）
        existing_targets = {}
        for r in records:
            existing_targets.setdefault(r.get("modality"), set()).add(r.get("target"))

        force_rerun = _limit_rerun(
            actions.get("rerun_same_plan", []) or [],
            existing_targets=existing_targets,
            k=1
        )

        # 如果没有允许的动作，就早停，别白跑下一轮
        if (not tool_requests) and (not force_rerun):
            reflection_output["status"] = "STOP_EARLY"
            reflection_output["stop_reason"] = "no_allowed_actions"
            last_bundle["reflection_output"] = reflection_output
            break

        # ✅ 只有需要 replan 且你允许 tool_requests 时才调用 planner
        # 你这里我们禁止新增 tool_requests，所以默认不 replan，下一轮只做少量 rerun
        # 如果你未来想允许“最多新增1个工具”，把上面 tool_requests=[] 改成 top1 即可
        if tool_requests:
            planning_output = run_planning_agent(
                analysis_output=analysis_output,
                tool_list=tool_list,
                image_roles=image_roles,
                tool_requests=tool_requests
            )
            planning_output = filter_plan_by_image_roles(planning_output, init_output["image_roles"])

    return last_bundle


def run_reasoning_agent(normalized_question, answer_spec, analysis_output, urban_state, reflection_output=None):
    if reflection_output is None:
        reflection_output = {}

    prompt = safe_format(reasoning_agent_prompt,
        normalized_question=normalized_question,
        answer_spec=str(answer_spec),
        analysis_output=analysis_output,
        urban_state=urban_state,
        reflection_output=reflection_output
    )
    return safe_json_loads(LLM(prompt))


def run_conclusion_agent(answer_spec, reasoning_output):
    prompt = safe_format(
        conclusion_agent_prompt,
        answer_spec=str(answer_spec),
        reasoning_output=reasoning_output
    )
    out = LLM(prompt)
    if isinstance(out, dict):
        # 有些 LLM 可能返回 {"answer": "A"}，兜底一下
        out = out.get("answer", "")
    return str(out).strip()


# def safe_json_loads(x):
#     """Parse dict or JSON-ish string into dict/list, with robust cleanup."""
#     if isinstance(x, (dict, list)):
#         return x
#     if not isinstance(x, str):
#         raise TypeError(f"safe_json_loads expected dict/list/str, got {type(x)}")

#     s = x.strip()

#     # Strip code fences if present
#     if s.startswith("```"):
#         # remove leading ```json or ``` and trailing ```
#         s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
#         s = re.sub(r"\n?```$", "", s)
#         s = s.strip()

#     # Extract largest JSON object/array region
#     obj_start = s.find("{")
#     arr_start = s.find("[")
#     if obj_start == -1 and arr_start == -1:
#         # not JSON, return raw string
#         return s

#     if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
#         start = obj_start
#         end = s.rfind("}")
#     else:
#         start = arr_start
#         end = s.rfind("]")

#     if start != -1 and end != -1 and end > start:
#         s = s[start:end+1].strip()

#     # First attempt
#     try:
#         return json.loads(s)
#     except json.JSONDecodeError:
#         # ---- Cleanup pass 1: remove illegal \' (common from LLM) ----
#         s2 = s.replace("\\'", "'")

#         # ---- Cleanup pass 2: escape any remaining invalid backslash escapes ----
#         # JSON valid escapes: \" \\ \/ \b \f \n \r \t \uXXXX
#         s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s2)

#         # Try again
#         return json.loads(s2)

# def safe_json_loads(x):
#     """Parse dict or JSON-ish string into dict/list, with robust cleanup."""
#     if isinstance(x, (dict, list)):
#         return x
#     if not isinstance(x, str):
#         raise TypeError(f"safe_json_loads expected dict/list/str, got {type(x)}")

#     s = x.strip()

#     # ---- Strip code fences ----
#     if s.startswith("```"):
#         s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
#         s = re.sub(r"\n?```$", "", s)
#         s = s.strip()

#     # ---- Extract largest JSON block ----
#     obj_start = s.find("{")
#     arr_start = s.find("[")
#     if obj_start == -1 and arr_start == -1:
#         return s

#     if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
#         start = obj_start
#         end = s.rfind("}")
#     else:
#         start = arr_start
#         end = s.rfind("]")

#     if start != -1 and end != -1 and end > start:
#         s = s[start:end+1].strip()

#     # =====================================================
#     # ✅ NEW: Remove JS/C++ style comments (// ...)
#     # =====================================================
#     s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)

#     # =====================================================
#     # First attempt
#     # =====================================================
#     try:
#         s = re.sub(r'[\x00-\x1f\x7f]', '', s)
#         return json.loads(s,strict=False)
#     except json.JSONDecodeError:
#         # ---- Cleanup pass 1: illegal \' ----
#         s2 = s.replace("\\'", "'")

#         # ---- Cleanup pass 2: invalid backslash escapes ----
#         s2 = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s2)

#         # ---- Cleanup pass 3: remove trailing commas ----
#         s2 = re.sub(r",\s*([}\]])", r"\1", s2)
#         s2 = re.sub(r'[\x00-\x1f\x7f]', '', s2)
#         return json.loads(s2,strict=False)


import json, re

def safe_json_loads(s: str):
    if not isinstance(s, str):
        return s

    # 1) try direct
    try:
        return json.loads(s, strict=False)
    except Exception:
        pass

    # 2) extract first {...} block
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)

    # 3) common fixes
    s2 = s.strip()

    # Python literals -> JSON literals
    s2 = re.sub(r"\bNone\b", "null", s2)
    s2 = re.sub(r"\bTrue\b", "true", s2)
    s2 = re.sub(r"\bFalse\b", "false", s2)

    # Remove trailing commas before } or ]
    s2 = re.sub(r",(\s*[}\]])", r"\1", s2)

    # If it uses single quotes heavily, try a conservative conversion:
    # replace single-quoted keys: 'key': -> "key":
    s2 = re.sub(r"(?P<pre>[\{\s,])'(?P<key>[^']+?)'\s*:", r'\g<pre>"\g<key>":', s2)
    # replace single-quoted string values: :"val"  (only when it looks like a value)
    s2 = re.sub(r":\s*'([^']*)'", r': "\1"', s2)

    # 4) final try
    return json.loads(s2, strict=False)

def strip_real_paths(obj):
    """
    remove or mask windows-like paths from nested dict/list/string
    """
    import re
    def _clean_str(x):
        # mask like D:\xxx\yyy.png or D:\\xxx\\yyy.png
        x = re.sub(r"[A-Za-z]:\\\\[^\"'\s]+", "<PATH>", x)
        x = re.sub(r"[A-Za-z]:\\[^\"'\s]+", "<PATH>", x)
        return x

    if isinstance(obj, str):
        return _clean_str(obj)
    if isinstance(obj, list):
        return [strip_real_paths(v) for v in obj]
    if isinstance(obj, dict):
        return {k: strip_real_paths(v) for k, v in obj.items()}
    return obj

def classify_image_type(image_path: str , registry=None):
    # print('image_path: ',image_path)
    real_path = resolve_image_ref(image_path, registry)  # ✅ IMG01 -> /abs/xxx.png
    resp,_,_,_ = VLM([real_path], vlm_image_type_prompt)
    data = safe_json_loads(resp)

    # 兜底字段
    img_type = data.get("type", "other")
    conf = float(data.get("confidence", 0.0))
    reason = data.get("reason", "")

    return {"path": image_path, "type": img_type, "confidence": conf, "reason": reason}


def run_execution_agent_incremental(planning_output, image_paths, cache_records=None, force_rerun=None,registry=None):
    """
    planning_output: dict
    cache_records: list of existing execution records (same schema as execution_output["records"])
    force_rerun: set of keys {(modality,target,tool_name), ...} that must be rerun even if cached
    """
    planning_output = safe_json_loads(planning_output)

    if cache_records is None:
        cache_records = []
    if force_rerun is None:
        force_rerun = set()

    # Build cache index
    cache_index = {}
    for r in cache_records:
        k = (r.get("modality"), r.get("target"), r.get("tool_name"))
        cache_index[k] = r

    new_records = []

    # helper to run one step on concrete targets
    def _run_step(modality, step):
        tool_name = step["tool_name"]
        tool_fn = TOOL_API_MAP[tool_name]

        targets = step.get("targets", [])
        image_targets = _normalize_targets(modality, targets, image_paths)
        tool_prompt = _make_tool_prompt(step)

        for target_name, img_path in image_targets:
            k = (modality, target_name, tool_name)
            if k in cache_index and k not in force_rerun:
                # reuse cached record
                new_records.append(cache_index[k])
                continue

            # execute (handle tools with 1-arg or 2-arg signatures)
            real_img_path = resolve_image_ref(img_path, registry)  # ✅ alias -> real path
            try:
                _,tool_out,_,_,_ = tool_fn(img_path, tool_prompt)
            except TypeError:
                _,tool_out,_,_,_ = tool_fn(img_path)

            rec = {
                "modality": modality,
                "target": target_name,
                "tool_name": tool_name,
                "purpose": step.get("purpose", ""),
                "prompt": tool_prompt,
                "input_image": img_path,
                "output": tool_out
            }
            new_records.append(rec)
            cache_index[k] = rec  # update cache

    for step in planning_output.get("satellite_plan", []):
        _run_step("satellite", step)

    for step in planning_output.get("street_view_plan", []):
        _run_step("street_view", step)

    return {"records": new_records}


def _record_key(rec: dict):
    """Cache key for one tool execution."""
    return (rec.get("modality"), rec.get("target"), rec.get("tool_name"))

def _build_cache_index(cache_records):
    """Build {key: record} index."""
    idx = {}
    for r in (cache_records or []):
        idx[_record_key(r)] = r
    return idx

_ALLOWED_TASK_TYPES = {
    "Population Prediction",
    "Infrastructure Inference",
    "Satellite Address Inference",
    "Satellite Land Use Inference",
    "Building Comparison",
    "Point of Interest Comparison",
    "Street View Address Inference",
    "Landmark Inference",
    "Street View Outlier Detection",
    "Street View Localization within Satellite Quadrants",
    "Satellite Image Retrieval given a Street View",
}

_ALLOWED_SIGNALS = {
    "land_use",
    "infrastructure_presence",
    "road_network_pattern",
    "building_density_footprints",
    "height_range",
    "block_morphology_layout",
    "waterfront_proximity",
    "street_scene_category",
    "facade_style_cues",
    "commercial_activity_cues",
    "object_presence_counts",
    "vegetation_presence",
    "text_sign_ocr",
    "cross_view_alignment",
    "similarity_retrieval",
    "outlier_detection",
}

import json
from typing import Any, Dict, List, Optional

def _safe_json_loads(s: str) -> Dict[str, Any]:
    s = s.strip()
    # strip ``` fences if present
    if s.startswith("```"):
        parts = s.split("```")
        s = parts[1] if len(parts) > 1 else s
        s = s.replace("json", "", 1).strip()
        s = re.sub(r'[\x00-\x1f\x7f]', '', s)
    return json.loads(s,strict=False)

# def run_task_router(
#     analysis_output: str,
#     image_roles: Dict[str, Any],
#     tool_requests: Optional[List[Any]] = None,
#     LLM=None,  # function: LLM(prompt) -> str
# ) -> Dict[str, Any]:
#     """
#     Returns routing JSON for downstream shortlisting/planning.
#     """
#     if LLM is None:
#         raise ValueError("LLM function must be provided: LLM(prompt) -> str")

#     prompt = TASK_ROUTER_PROMPT.format(
#         analysis_output=analysis_output,
#         image_roles=json.dumps(image_roles, ensure_ascii=False),
#         tool_requests=json.dumps(tool_requests or [], ensure_ascii=False),
#     )

#     raw = LLM(prompt)
#     data = _safe_json_loads(raw)

#     # ---- Sanity checks / normalization ----
#     task_type = data.get("task_type")
#     if task_type not in _ALLOWED_TASK_TYPES:
#         raise ValueError(f"Invalid task_type: {task_type}. Raw: {raw[:500]}")

#     # Normalize signals
#     needed_signals = data.get("needed_signals", [])
#     if not isinstance(needed_signals, list):
#         needed_signals = []
#     # Filter to allowed (fail-safe)
#     needed_signals = [s for s in needed_signals if s in _ALLOWED_SIGNALS]
#     data["needed_signals"] = needed_signals

#     # Defaults
#     data.setdefault("required_modalities", [])
#     data.setdefault("optional_modalities", [])
#     data.setdefault("need_street_view_summary_if_available", False)
#     data.setdefault("forbidden_claims", [
#         "named_street_address_identification",
#         "named_poi_identification",
#         "exact_geocoordinates"
#     ])
#     data.setdefault("reflection_requests", {"requested_tools": [], "requested_modalities": []})
#     data.setdefault("notes", "")

#     # Basic type checks
#     if not isinstance(data["required_modalities"], list):
#         data["required_modalities"] = []
#     if not isinstance(data["optional_modalities"], list):
#         data["optional_modalities"] = []

#     return data

def run_task_router(
    analysis_output: str,
    image_roles: Dict[str, Any],
    tool_requests: Optional[List[Any]] = None,
    LLM=None,  # function: LLM(prompt) -> str
) -> Dict[str, Any]:
    """
    Returns routing JSON for downstream shortlisting/planning.
    Robust: never raises on invalid LLM output. Falls back to a generic route.
    """
    if LLM is None:
        raise ValueError("LLM function must be provided: LLM(prompt) -> str")

    def _fallback(reason: str, raw: str = "", parsed: Any = None) -> Dict[str, Any]:
        has_sat = bool(image_roles.get("satellite"))
        has_stv = bool(image_roles.get("street_view"))
        # generic route that won't break downstream
        return {
            "router_status": "fallback",
            "task_type": "generic",
            "needed_signals": [],  # downstream can ignore or use generic defaults
            "required_modalities": [m for m, ok in [("satellite", has_sat), ("street_view", has_stv)] if ok],
            "optional_modalities": [],
            "need_street_view_summary_if_available": has_stv,
            "forbidden_claims": [
                "named_street_address_identification",
                "named_poi_identification",
                "exact_geocoordinates"
            ],
            "reflection_requests": {"requested_tools": [], "requested_modalities": []},
            "notes": f"[router_fallback] {reason}",
            "raw_llm_output": (raw or "")[:1200],
            "parsed_output": parsed,
        }

    prompt = TASK_ROUTER_PROMPT.format(
        analysis_output=analysis_output,
        image_roles=json.dumps(image_roles, ensure_ascii=False),
        tool_requests=json.dumps(tool_requests or [], ensure_ascii=False),
    )

    raw = ""
    try:
        raw = LLM(prompt)
        data = _safe_json_loads(raw)
        if not isinstance(data, dict):
            return _fallback("router_output_not_dict", raw=raw, parsed=data)
    except Exception as e:
        return _fallback(f"llm_or_parse_error: {e}", raw=raw)

    # ---- Sanity checks / normalization ----
    task_type = data.get("task_type")
    if task_type not in _ALLOWED_TASK_TYPES:
        return _fallback(f"invalid_task_type: {task_type}", raw=raw, parsed=data)

    # Normalize signals
    needed_signals = data.get("needed_signals", [])
    if not isinstance(needed_signals, list):
        needed_signals = []
    needed_signals = [s for s in needed_signals if s in _ALLOWED_SIGNALS]
    data["needed_signals"] = needed_signals

    # Defaults
    data.setdefault("router_status", "ok")
    data.setdefault("required_modalities", [])
    data.setdefault("optional_modalities", [])
    data.setdefault("need_street_view_summary_if_available", False)
    data.setdefault("forbidden_claims", [
        "named_street_address_identification",
        "named_poi_identification",
        "exact_geocoordinates"
    ])
    data.setdefault("reflection_requests", {"requested_tools": [], "requested_modalities": []})
    data.setdefault("notes", "")

    # Basic type checks
    if not isinstance(data["required_modalities"], list):
        data["required_modalities"] = []
    if not isinstance(data["optional_modalities"], list):
        data["optional_modalities"] = []

    return data


def _has_modality(image_roles: Dict[str, Any], key: str) -> bool:
    """
    Robust-ish check for modality existence.
    Supports:
      - {"satellite": [...]} / {"street_view": [...]}
      - {"satellite": {"image_paths":[...]}} style
    """
    if key not in image_roles or image_roles[key] is None:
        return False
    v = image_roles[key]
    if isinstance(v, list):
        return len(v) > 0
    if isinstance(v, dict):
        # common patterns
        for k in ("image_paths", "paths", "images"):
            if k in v and isinstance(v[k], list):
                return len(v[k]) > 0
        # fallback: any non-empty dict counts as present
        return len(v) > 0
    return True

def run_tool_shortlister_satellite(
    task_route: Dict[str, Any],
    analysis_output: str,
    tool_list: str,
    available_tool_names: List[str],
    image_roles: Dict[str, Any],
    tool_requests: Optional[List[Any]] = None,
    max_tools: int = 4,
    LLM=None,  # LLM(prompt)->str
) -> Dict[str, Any]:
    
    if task_route.get("router_status") != "ok":
        # 只在真的有 satellite 图像时返回工具
        if not image_roles.get("satellite"):
            return {"shortlist": []}

        return {
            "shortlist": [
                "Satellite Image Semantic Segmentation Tool",
                "Satellite Image Object Detection Tool",
                "Structure Layout Analyzer",
                "Satellite Image Land Use Inference Tool",
            ][:max_tools]
        }
    
    if LLM is None:
        raise ValueError("LLM function must be provided: LLM(prompt) -> str")

    has_sat = _has_modality(image_roles, "satellite")
    prompt = SAT_SHORTLISTER_PROMPT.format(
        task_route=json.dumps(task_route, ensure_ascii=False),
        analysis_output=analysis_output,
        image_roles=json.dumps(image_roles, ensure_ascii=False),
        available_tool_names=json.dumps(available_tool_names, ensure_ascii=False),
        tool_list=tool_list,
        tool_requests=json.dumps(tool_requests or [], ensure_ascii=False),
        max_tools=max_tools
    )

    raw = LLM(prompt)
    data = _safe_json_loads(raw)

    # Enforce modality gate in code as a safety net
    if not has_sat:
        # allow forced inclusion only if reflection requested satellite tools (best-effort)
        data["shortlist"] = data.get("shortlist", [])
        if not data["shortlist"]:
            data["shortlist"] = []
            data["rationales"] = []
            data["notes"] = (data.get("notes","") + " No satellite modality present.").strip()

    # Filter shortlist to allowed tool names & cap length
    allowed = set(available_tool_names)
    shortlist = [t for t in data.get("shortlist", []) if t in allowed][:max_tools]
    data["shortlist"] = shortlist

    # Rationales: keep only those for selected tools
    rats = data.get("rationales", [])
    if isinstance(rats, list):
        data["rationales"] = [r for r in rats if isinstance(r, dict) and r.get("tool_name") in shortlist]
    else:
        data["rationales"] = []

    data.setdefault("notes", "")
    return data

def run_tool_shortlister_street_view(
    task_route: Dict[str, Any],
    analysis_output: str,
    tool_list: str,
    available_tool_names: List[str],
    image_roles: Dict[str, Any],
    tool_requests: Optional[List[Any]] = None,
    max_tools: int = 4,
    LLM=None,  # LLM(prompt)->str
) -> Dict[str, Any]:
    
    # ====== ✅ Fallback 兜底（开头加）======
    if task_route.get("router_status") != "ok":
        if not image_roles.get("street_view"):
            return {"shortlist": []}

        return {
            "shortlist": [
                "Building Facade Extractor",
                "Street Object Detector",
                "Street View Semantic Segmentation Tool",
                "Text Sign OCR",
            ][:max_tools]
        }
    
    if LLM is None:
        raise ValueError("LLM function must be provided: LLM(prompt) -> str")

    has_stv = _has_modality(image_roles, "street_view")
    prompt = STV_SHORTLISTER_PROMPT.format(
        task_route=json.dumps(task_route, ensure_ascii=False),
        analysis_output=analysis_output,
        image_roles=json.dumps(image_roles, ensure_ascii=False),
        available_tool_names=json.dumps(available_tool_names, ensure_ascii=False),
        tool_list=tool_list,
        tool_requests=json.dumps(tool_requests or [], ensure_ascii=False),
        max_tools=max_tools
    )

    raw = LLM(prompt)
    data = _safe_json_loads(raw)

    # Enforce modality gate safety net
    if not has_stv:
        data["shortlist"] = data.get("shortlist", [])
        if not data["shortlist"]:
            data["shortlist"] = []
            data["rationales"] = []
            data["notes"] = (data.get("notes","") + " No street_view modality present.").strip()

    # Filter shortlist to allowed tool names & cap length
    allowed = set(available_tool_names)
    shortlist = [t for t in data.get("shortlist", []) if t in allowed][:max_tools]
    data["shortlist"] = shortlist

    rats = data.get("rationales", [])
    if isinstance(rats, list):
        data["rationales"] = [r for r in rats if isinstance(r, dict) and r.get("tool_name") in shortlist]
    else:
        data["rationales"] = []

    data.setdefault("notes", "")
    return data

# func_sup.py
from typing import List

def filter_plan_by_image_roles(planning_output: dict, image_roles: dict):
    has_sat = bool(image_roles.get("satellite"))
    has_stv = bool(image_roles.get("street_view"))

    if not has_sat:
        planning_output.get("selected_tools", {}).update({"satellite": []})
        planning_output["satellite_plan"] = []
    if not has_stv:
        planning_output.get("selected_tools", {}).update({"street_view": []})
        planning_output["street_view_plan"] = []
    return planning_output


def filter_tool_list_text(tool_list_text: str, allowed_tool_names: List[str]) -> str:
    """
    Best-effort: keep only tool blocks whose 'Name:' line matches allowed_tool_names.
    Assumes tool blocks are separated and contain 'Name:' lines.
    """
    allowed = set(allowed_tool_names)
    blocks = tool_list_text.split("\n=== Tool Description")
    kept = [blocks[0]]  # header part before first block (if any)

    for b in blocks[1:]:
        # reconstruct marker
        block = "=== Tool Description" + b
        name = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("Name:"):
                name = line[len("Name:"):].strip()
                break
        if name in allowed:
            kept.append(block)

    return "\n".join(kept).strip()


def filter_requirements_by_modalities(analysis_output: dict, image_roles: dict) -> dict:
    def has_modality(key: str) -> bool:
        v = image_roles.get(key, None)
        if v is None:
            return False
        if isinstance(v, list):
            return len(v) > 0
        if isinstance(v, dict):
            for k in ("image_paths", "paths", "images"):
                if isinstance(v.get(k, None), list):
                    return len(v[k]) > 0
            return len(v) > 0
        return True

    has_sat = has_modality("satellite")
    has_stv = has_modality("street_view")

    reqs = analysis_output.get("requirements", [])
    kept = []
    for r in reqs:
        mod = r.get("modality")
        if mod == "satellite" and not has_sat:
            continue
        if mod == "street_view" and not has_stv:
            continue
        kept.append(r)

    analysis_output["requirements"] = kept
    return analysis_output
