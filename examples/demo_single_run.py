import json
from func_sup import *
from tool_list_short import TOOL_LIST  # 完整工具池

# img_path_0 = "../ImageData/London_urbanllava/10893_16375.png"
# img_path_1 = "../ImageData/London_urbanllava/CdK7iPnYNV2l9Dh5t_3AoQ&51.52688203706936&-0.09047034014079254&15D.jpg"

# original_question_string = "You are given a satellite image <image> and a street view image <image>. Which quadrant of the satellite image shows the location of the street view image.\n            A. Top left\n            B. Top right\n            C. Bottom left\n            D. Bottom right\n            Only provide one letter as the answer and please select your answer from A, B, C, or D."

# images = [img_path_0, img_path_1]

# original_question_string = "How would you rate the overall overall population density of this satellite image on a scale of 0.0 to 9.9, with 9.9 being the highest? Only output the score 'X.X'.",

# img_path_0 = "../ImageData/Beijing_citybench/12404_26960.png"
# images = [img_path_0]

# images=[
# "../ImageData/NewYork_urbanllava/49232_38611.png",
# "../ImageData/NewYork_urbanllava/49231_38608.png",
# "../ImageData/NewYork_urbanllava/49227_38625.png",
# "../ImageData/NewYork_urbanllava/49241_38617.png"]

# original_question_string = '''In the provided four satellite images in urban area, which image probably shows most POIs (For example, ['restaurants', 'bakerys', 'foods', 'fast_foods', 'beveragess', 'food_courts', 'bars', 'cafes', 'coffees', 'vending_machines', 'nightclubs'])?\n        A. The first image <image>\n\n        B. The second image <image>\n\n        C. The third image <image>\n\n        D. The fourth image <image>\n\n        Only provide one letter as the answer and please select your answer from A, B, C, or D.'''

images = ["../ImageData/London_urbanllava/43578_65502.png"]
original_question_string = '''
The following is a multiple-choice question about selecting the most possible landuse type in the region of a satellite image.\n    A. Garages\n    B. Commercial\n    C. Residential\n    D. Recreation\n    Please choose the most suitable one among A, B, C and D as the answer to this question. \n    Please output the option directly. No need for explaination.
'''

print(images)
question = original_question_string

init_output = run_init_agent(question, images)
print('init output: ', init_output)

# # json.loads
# if init_output["preprocess"]["satellite_quadrants_required"]:
#     satellite_quadrants = split_into_quadrants(
#         init_output["image_roles"]["satellite"][0]
#     )
# else:
#     satellite_quadrants = {}

# image_paths = {
#     "satellite": init_output["image_roles"]["satellite"],
#     "satellite_quadrants": satellite_quadrants,
#     "street_view": init_output["image_roles"]["street_view"]
# }

image_paths = {}

if init_output["image_roles"].get("satellite"):
    image_paths["satellite"] = init_output["image_roles"]["satellite"]

if init_output["image_roles"].get("street_view"):
    image_paths["street_view"] = init_output["image_roles"]["street_view"]

# 只有当 Init 明确要求切 quadrant，且 satellite 存在
if (
    init_output.get("preprocess", {}).get("satellite_quadrants_required", False)
    and "satellite" in image_paths
):
    satellite_quadrants = split_into_quadrants(image_paths["satellite"][0])
    image_paths["satellite_quadrants"] = satellite_quadrants

# print("SAT for split:", init_output["image_roles"]["satellite"][0])

# print("STV:", init_output["image_roles"]["street_view"][0])

# print("First quad path:", list(image_paths["satellite_quadrants"].values())[0])


analysis_output = run_analysis_agent(
    normalized_question=init_output["normalized_question"],
    answer_spec=init_output["answer_spec"],
    image_roles=init_output["image_roles"]
)
print('analysis output: ', analysis_output)

planning_output = run_planning_agent(
    analysis_output=analysis_output,
    tool_list=TOOL_LIST,
    tool_requests=[]  # 第一轮没有reflection
)
print('planning output: ', planning_output)

execution_output = run_execution_agent(
    planning_output=planning_output,
    image_paths=image_paths
)
print('execution output: ', execution_output)

urban_state = run_state_agent(
    normalized_question=init_output["normalized_question"],
    analysis_output=analysis_output,
    execution_output=execution_output
)
print('urban state: ', urban_state)

bundle = run_closed_loop_pipeline(
    init_output=init_output,
    analysis_output=analysis_output,
    tool_list=TOOL_LIST,
    image_paths=image_paths,
    max_iters=1
)
print('bundle: ', bundle)

reasoning_output = run_reasoning_agent(
    normalized_question=init_output["normalized_question"],
    answer_spec=init_output["answer_spec"],
    analysis_output=analysis_output,
    urban_state=bundle["urban_state"],
    reflection_output=bundle["reflection_output"]
)
print('reasoning output: ', reasoning_output)

final_answer = run_conclusion_agent(
    answer_spec=init_output["answer_spec"],
    reasoning_output=reasoning_output
)
print('final answer: ', final_answer)