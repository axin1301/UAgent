Satellite Image Tools

=== Tool Description for S1 ===
Name: Satellite Image Automatic Description Generator
Ability: Generates high-level natural language descriptions of a satellite image, summarizing overall land use, spatial layout, and general urban characteristics. 
Does NOT identify precise street names, addresses, or uniquely named landmarks.

=== Tool Description for S2 ===
Name: Satellite Image Object Detection Tool
Ability: Detects generic object categories such as buildings, vehicles, ships, or tanks at a coarse level.
Does NOT identify specific instances, object identities, street names, or vehicle license information.

=== Tool Description for S3 ===
Name: Area Estimator
Ability: Estimates the proportion of broad surface categories (e.g., roads, buildings, vegetation, water) based on semantic segmentation.
Provides aggregate statistics only; NOT suitable for identifying specific streets or parcels.

=== Tool Description for S4 ===
Name: Building Footprint Extractor
Ability: Describes overall building footprint patterns, density, and spatial distribution at block or neighborhood scale.
Does NOT output individual building addresses or names.

=== Tool Description for S5 ===
Name: Building Height Extractor
Ability: Estimates approximate building height ranges and vertical distribution patterns across an area.
Does NOT determine the height of a specific named building.

=== Tool Description for S6 ===
Name: Road Network Extractor
Ability: Characterizes road network structure, connectivity, and orientation (e.g., grid-like vs organic layouts).
Does NOT identify or name individual roads or streets.

=== Tool Description for S8 ===
Name: Satellite Image Land Use Inference Tool
Ability: Infers dominant land-use categories (residential, industrial, commercial, mixed-use) at area level.
Does NOT infer precise zoning boundaries or addresses.

=== Tool Description for S9 ===
Name: Structure Layout Analyzer
Ability: Analyzes spatial arrangement of urban elements, including block size, intersection density, and building compactness.
Focuses on structural patterns rather than named locations.

=== Tool Description for S10 ===
Name: Image Style Classifier
Ability: Classifies the overall urban style of an area (e.g., dense urban core, suburban, industrial zone).
Provides coarse categorization only.

=== Tool Description for S11 ===
Name: Factory Recognizer
Ability: Detects the presence of factory-like structures as a category and their approximate locations.
Does NOT identify specific factory names or operators.

=== Tool Description for S12 ===
Name: Large Parking Area Detector
Ability: Identifies large-scale parking facilities or open parking zones.
Does NOT associate parking areas with specific businesses or addresses.

=== Tool Description for S13 ===
Name: Special Target Recognizer
Ability: Detects broad classes of special infrastructure (e.g., oil tanks, water towers).
Classification is generic and non-instance-specific.

=== Tool Description for S14 ===
Name: Satellite Image Roof and Building Footprint Detailer
Ability: Extracts roof shape patterns and building footprint details at structural level.
Does NOT identify buildings by name or function.

=== Tool Description for S15 ===
Name: Satellite Image Landmark Extraction Tool
Ability: Identifies visually prominent landmark-like structures based on shape and scale.
Landmark detection is approximate and NOT guaranteed to match officially named points of interest.

=== Tool Description for S16 ===
Name: Satellite Image Geospatial Named Entity Extractor
Ability: Attempts to infer coarse geospatial cues (e.g., possible street-like linear features) from satellite imagery.
Results are uncertain and should NOT be treated as reliable street or address identification.

=== Tool Description for S17 ===
Name: Satellite Image Railway Detector
Ability: Detects railway lines and rail-related infrastructure as a category.
Does NOT identify specific railway line names or stations.

=== Tool Description for S18 ===
Name: Satellite Image Geo-Region Localizer
Ability: Estimates the coarse geographic region of a satellite image (e.g., city-level or borough-level) based on large-scale urban morphology, spatial patterns, and contextual visual cues.
Provides probabilistic regional hypotheses rather than exact coordinates or addresses.

=== Tool Description for S19 ===
Name: Satellite Image Waterfront Proximity Analyzer
Ability: Determines whether a satellite image area is adjacent to a major water body (e.g., river or coastline) and estimates the relative direction of the water body.
Does NOT identify the name of the water body.

=== Tool Description for S20 ===
Name: Urban Block Morphology Classifier
Ability: Classifies urban block morphology (e.g., large public housing superblocks, fine-grained mixed-use blocks, industrial blocks) based on building size, spacing, and layout patterns.
Focuses on structural urban form rather than named neighborhoods.

---
Street View Tools

=== Tool Description for G1 ===
Name: Street Object Detector
Ability: Detects generic street-level object categories such as vehicles, people, buildings, and signs.
Does NOT identify specific brands, license plates, or named locations.

=== Tool Description for G2 ===
Name: Street View Semantic Segmentation Tool
Ability: Performs pixel-level classification of broad semantic categories (road, sky, building, vegetation).
Outputs are category-level only.

=== Tool Description for G3 ===
Name: Building Facade Extractor
Ability: Describes facade characteristics and architectural features at style or function level.
Does NOT identify specific building names or addresses.

=== Tool Description for G4 ===
Name: Text Sign OCR
Ability: Extracts visible text or numbers from signs when legible.
OCR results may be partial or noisy and are NOT guaranteed to correspond to official street names or addresses.

=== Tool Description for G5 ===
Name: Building Type Cue Detector
Ability: Detects architectural cues indicative of building type categories.
Operates at pattern level, not individual identification.

=== Tool Description for G6 ===
Name: Vehicle Type Classifier
Ability: Classifies vehicles into coarse types (car, bus, truck, motorcycle).
Does NOT identify manufacturers or license plates.

=== Tool Description for G7 ===
Name: Pedestrian Density Estimator
Ability: Estimates pedestrian density at scene level.
Does NOT track or identify individuals.

=== Tool Description for G8 ===
Name: Street View Image Captioner
Ability: Generates high-level scene descriptions summarizing visible elements.
Captions are descriptive, not authoritative.

=== Tool Description for G9 ===
Name: Building Function Inference Tool
Ability: Infers likely building function categories (residential, commercial, industrial).
Inference is probabilistic and non-specific.

=== Tool Description for G10 ===
Name: Street Scene Category Inference
Ability: Classifies street scenes into broad area types (residential, commercial, industrial).
Does NOT infer exact neighborhoods.

=== Tool Description for G11 ===
Name: Infrastructure Detector
Ability: Detects generic infrastructure elements (streetlights, fences, towers).
Detection is category-level only.

=== Tool Description for G12 ===
Name: Vegetation Detector
Ability: Identifies presence and distribution of vegetation types.
Does NOT identify specific plant species or named parks.

=== Tool Description for G13 ===
Name: Commercial Clue Extractor
Ability: Extracts visual cues suggestive of commercial activity.
Does NOT identify specific businesses.

=== Tool Description for G14 ===
Name: Enclosure Detector
Ability: Determines whether a scene appears enclosed or fenced.
Provides binary or coarse classification.

=== Tool Description for G15 ===
Name: Street View Ground Level Detail Recognizer
Ability: Describes distinctive ground-level visual patterns and street furniture.
Focuses on visual texture and arrangement, not naming.

=== Tool Description for G16 ===
Name: Street View Architectural Style Classifier
Ability: Classifies architectural styles based on visual features.
Style labels are approximate and period-level.

=== Tool Description for G17 ===
Name: Object Count Reporter
Ability: Counts approximate numbers of object categories in a scene.
Counts are aggregate estimates.

=== Tool Description for G18 ===
Name: Height Approximation Tool
Ability: Estimates approximate building height ranges and floor counts.
Does NOT provide exact measurements for named buildings.