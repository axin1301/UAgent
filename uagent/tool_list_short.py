TOOL_LIST = '''
Satellite Image Tools
Satellite Image Automatic Description Generator
Ability: Generates high-level natural language descriptions of a satellite image, summarizing land use, urban style, block morphology, and overall spatial patterns. Does NOT identify precise street names, addresses, or named landmarks.

Satellite Image Object Detection Tool
Ability: Detects generic object categories such as buildings, vehicles, ships, tanks, and other large structures at a coarse level.
Does NOT identify specific instances or object identities.

Area Estimator
Ability: Estimates proportions of broad surface categories (roads, buildings, vegetation, water) using semantic segmentation.
Provides aggregate statistics only.

Building Footprint Extractor
Ability: Describes building footprint patterns, density, spatial distribution, and basic footprint geometry at block or neighborhood scale.
Does NOT output individual building identities.

Building Height Extractor
Ability: Estimates approximate building height ranges and vertical distribution patterns across an area. Does NOT determine exact heights of specific buildings.

Road Network Extractor
Ability: Characterizes road network structure, connectivity, orientation, and other linear transportation patterns. Does NOT identify or name individual roads or streets.

Satellite Image Land Use Inference Tool
Ability: Infers dominant land-use and functional categories (residential, industrial, commercial), including factory-like areas and large parking zones, at area level. Does NOT identify specific facilities or operators.

Structure Layout Analyzer
Ability: Analyzes spatial arrangement of urban elements such as block size, intersection density, compactness, and layout regularity. Focuses on structural patterns only.

Special Target Recognizer
Ability: Detects broad classes of special infrastructure, including oil tanks, water towers, railways, and rail-related facilities. Classification is generic and non-instance-specific.

Satellite Image Landmark Extraction Tool
Ability: Identifies visually prominent **LANDMARK** structures based on shape, scale, or spatial dominance. Landmarks are approximate and may not correspond to officially named POIs.

Satellite Image Geo-Region Localizer
Ability: Estimates coarse **GEOGRAPHIC** context and environmental adjacency (e.g., city-scale region, proximity to coastline or river) using large-scale visual cues. Provides probabilistic regional hypotheses only.

Street View Tools
Street Object Detector
Ability: Detects and counts generic street-level objects such as vehicles, people, buildings, signs, and infrastructure elements. All outputs are category-level and aggregate.

Street View Semantic Segmentation Tool
Ability: Performs pixel-level classification of broad semantic categories including road, building, vegetation, sky, and ground. Does NOT identify specific entities.

Building Facade Extractor
Ability: Describes facade characteristics and architectural style features at a visual-pattern level. Does NOT identify building names or addresses.

Text Sign OCR
Ability: Extracts visible text or numbers from signs when legible. OCR results may be partial, noisy, and non-authoritative.

Building Type Cue Detector 
Ability: Detects architectural and visual cues indicative of building type or function categories (residential, commercial, industrial). Inference is probabilistic.

Street View Image Captioner
Ability: Generates high-level scene descriptions and classifies street scenes into broad area types. Captions are descriptive, not authoritative.

Pedestrian Density Estimator
Ability: Estimates pedestrian density at scene level. Does NOT track or identify individuals.

Commercial Clue Extractor
Ability: Extracts visual cues suggesting commercial activity and enclosure characteristics (open vs fenced). Does NOT identify specific businesses.

Street View Ground Level Detail Recognizer
Ability: Describes ground-level visual patterns, street furniture, and approximate building height ranges. Focuses on perceptual patterns, not precise measurement.
'''