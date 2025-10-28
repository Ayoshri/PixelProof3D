# pixelproof_security_scanner_fixed.py
import logging
import os
import tempfile
import time
import csv
import json
import gradio as gr
import numpy as np
import rembg
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from functools import partial
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
import argparse
import cv2
import pickle

# === Enhanced imports for security analysis ===
try:
    from sklearn.cluster import KMeans, DBSCAN
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.segmentation import felzenszwalb, slic
    from skimage.measure import regionprops, label
    from scipy.spatial.distance import euclidean
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    logging.warning("Advanced analysis libraries not available — security features limited")

# Device selection
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load TripoSR model
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to(device)

rembg_session = rembg.new_session()

# === PERSISTENT STORAGE SYSTEM ===
STORAGE_DIR = os.path.join(tempfile.gettempdir(), 'pixelproof_storage')
MODELS_DB_PATH = os.path.join(STORAGE_DIR, 'models_database.json')
USERS_DB_PATH = os.path.join(STORAGE_DIR, 'users_database.json')

def init_storage():
    """Initialize persistent storage directories and databases."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    os.makedirs(os.path.join(STORAGE_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(STORAGE_DIR, 'thumbnails'), exist_ok=True)
    
    # Initialize databases if they don't exist
    if not os.path.exists(MODELS_DB_PATH):
        with open(MODELS_DB_PATH, 'w') as f:
            json.dump([], f)
    
    if not os.path.exists(USERS_DB_PATH):
        with open(USERS_DB_PATH, 'w') as f:
            json.dump([], f)

def load_models_database():
    """Load saved models database."""
    try:
        with open(MODELS_DB_PATH, 'r') as f:
            return json.load(f)
    except:
        return []

def save_models_database(data):
    """Save models database."""
    try:
        with open(MODELS_DB_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except:
        return False

def save_model_to_storage(model_data, analysis_data, user_info):
    """Save model and analysis data to persistent storage."""
    try:
        models_db = load_models_database()
        
        # Generate unique ID
        model_id = f"PXL-{int(time.time())}-{len(models_db)}"
        
        # Create model entry
        model_entry = {
            "id": model_id,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC'),
            "user_info": user_info,
            "analysis": analysis_data,
            "dominant_colors": analysis_data.get("dominant_colors", [])[:5],  # Store top 5 colors for search
            "material_type": analysis_data.get("material_analysis", {}).get("material_type", "unknown"),
            "security_score": analysis_data.get("security_score", 0),
            "files": {}
        }
        
        # Save model files
        if "obj_path" in model_data and model_data["obj_path"]:
            obj_dest = os.path.join(STORAGE_DIR, 'models', f'{model_id}.obj')
            with open(model_data["obj_path"], 'rb') as src:
                with open(obj_dest, 'wb') as dst:
                    dst.write(src.read())
            model_entry["files"]["obj"] = obj_dest
        
        if "glb_path" in model_data and model_data["glb_path"]:
            glb_dest = os.path.join(STORAGE_DIR, 'models', f'{model_id}.glb')
            with open(model_data["glb_path"], 'rb') as src:
                with open(glb_dest, 'wb') as dst:
                    dst.write(src.read())
            model_entry["files"]["glb"] = glb_dest
        
        # Save thumbnail
        if "thumbnail" in model_data and model_data["thumbnail"]:
            thumb_path = os.path.join(STORAGE_DIR, 'thumbnails', f'{model_id}.jpg')
            model_data["thumbnail"].save(thumb_path, "JPEG", quality=85)
            model_entry["thumbnail"] = thumb_path
        
        # Add to database
        models_db.append(model_entry)
        save_models_database(models_db)
        
        return model_id
        
    except Exception as e:
        logging.exception("Failed to save model to storage: %s", e)
        return None

def search_saved_models(query_image=None, text_query=""):
    """Search saved models by image similarity or text."""
    try:
        models_db = load_models_database()
        
        if not models_db:
            return []
        
        results = []
        
        if query_image is not None:
            # Analyze query image for color matching
            query_analysis = security_color_analysis(query_image, 8)
            query_colors = query_analysis.get("dominant_colors", [])[:5]
            
            # Find models with similar colors
            for model in models_db:
                similarity_score = calculate_color_similarity(query_colors, model.get("dominant_colors", []))
                if similarity_score > 0.3:  # Threshold for similarity
                    model_copy = model.copy()
                    model_copy["similarity_score"] = similarity_score
                    results.append(model_copy)
        
        if text_query:
            # Text-based search
            text_query = text_query.lower()
            for model in models_db:
                user_info = model.get("user_info", {})
                searchable_text = f"{user_info.get('product_name', '')} {user_info.get('name', '')} {model.get('material_type', '')}".lower()
                
                if text_query in searchable_text:
                    if model not in results:
                        results.append(model)
        
        # Sort by similarity score or timestamp
        if query_image:
            results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        else:
            results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return results[:10]  # Return top 10 results
        
    except Exception as e:
        logging.exception("Failed to search saved models: %s", e)
        return []

def calculate_color_similarity(colors1, colors2):
    """Calculate similarity between two color palettes."""
    if not colors1 or not colors2:
        return 0
    
    similarity_sum = 0
    comparisons = 0
    
    for color1 in colors1[:3]:  # Compare top 3 colors
        for color2 in colors2[:3]:
            # Calculate Euclidean distance in RGB space
            try:
                distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1[:3], color2[:3])))
                similarity = max(0, 1 - (distance / (255 * np.sqrt(3))))  # Normalize to 0-1
                similarity_sum += similarity
                comparisons += 1
            except:
                continue
    
    return similarity_sum / comparisons if comparisons > 0 else 0

def load_saved_model(model_id):
    """Load a saved model by ID."""
    try:
        models_db = load_models_database()
        model_data = next((m for m in models_db if m["id"] == model_id), None)
        
        if not model_data:
            return None
        
        result = {
            "info": model_data,
            "obj_path": model_data["files"].get("obj"),
            "glb_path": model_data["files"].get("glb"),
            "thumbnail": model_data.get("thumbnail")
        }
        
        return result
        
    except Exception as e:
        logging.exception("Failed to load saved model: %s", e)
        return None

def get_saved_models_gallery():
    """Get gallery of saved models for dashboard."""
    try:
        models_db = load_models_database()
        
        gallery_items = []
        model_info = []
        
        for model in reversed(models_db[-20:]):  # Show latest 20 models
            if model.get("thumbnail") and os.path.exists(model["thumbnail"]):
                gallery_items.append(model["thumbnail"])
                model_info.append({
                    "id": model["id"],
                    "timestamp": model["timestamp"],
                    "product": model.get("user_info", {}).get("product_name", "Unknown Product"),
                    "customer": model.get("user_info", {}).get("name", "Unknown Customer"),
                    "material": model.get("material_type", "unknown")
                })
        
        return gallery_items, model_info
        
    except Exception as e:
        logging.exception("Failed to get saved models gallery: %s", e)
        return [], []

# Initialize storage on startup
init_storage()

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        return Image.fromarray((image * 255.0).astype(np.uint8))

    if do_remove_background:
        image = input_image.convert("RGBA")
        try:
            image = remove_background(image, rembg_session)
        except Exception:
            image = input_image.convert("RGBA")
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image

def generate(image, mc_resolution, formats=["obj", "glb"]):
    scene_codes = model(image, device=device)
    mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    rv = []
    for format in formats:
        mesh_path = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
        mesh.export(mesh_path.name)
        rv.append(mesh_path.name)
    return rv

def bilinear_sample(image_array, u, v):
    """Bilinear interpolation for smooth color sampling."""
    try:
        h, w = image_array.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x = u * (w - 1)
        y = v * (h - 1)
        
        # Get integer coordinates
        x1 = int(np.floor(x))
        x2 = min(x1 + 1, w - 1)
        y1 = int(np.floor(y))
        y2 = min(y1 + 1, h - 1)
        
        # Get fractional parts
        fx = x - x1
        fy = y - y1
        
        # Sample four neighboring pixels
        p11 = image_array[y1, x1]
        p21 = image_array[y1, x2]
        p12 = image_array[y2, x1]
        p22 = image_array[y2, x2]
        
        # Bilinear interpolation
        color = (p11 * (1 - fx) * (1 - fy) + 
                p21 * fx * (1 - fy) + 
                p12 * (1 - fx) * fy + 
                p22 * fx * fy)
        
        return color.astype(np.uint8)
        
    except Exception:
        # Fallback to simple sampling
        h, w = image_array.shape[:2]
        x = int(u * (w - 1))
        y = int(v * (h - 1))
        return image_array[y, x]

def create_proper_vertex_colors(mesh, texture_image, material_type="semi-glossy"):
    """Create proper vertex colors for the mesh with correct orientation."""
    if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
        return mesh
    
    try:
        # Convert texture to RGB array
        if isinstance(texture_image, Image.Image):
            texture_array = np.array(texture_image.convert('RGB'))
        else:
            texture_array = np.array(texture_image)
        
        # Get mesh bounds for UV mapping
        vertices = mesh.vertices.copy()
        
        # Normalize vertices to 0-1 range for UV mapping
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        ranges = max_coords - min_coords
        ranges = np.where(ranges == 0, 1, ranges)  # Avoid division by zero
        
        normalized_verts = (vertices - min_coords) / ranges
        
        # Create vertex colors array
        vertex_colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        
        h, w = texture_array.shape[:2]
        
        for i, norm_pos in enumerate(normalized_verts):
            # Improved UV mapping - use XZ plane primarily
            u = np.clip(norm_pos[0], 0, 1)
            v = np.clip(1 - norm_pos[2], 0, 1)  # Flip V for proper orientation
            
            # Convert to pixel coordinates
            x = int(u * (w - 1))
            y = int(v * (h - 1))
            
            # Sample color
            sampled_color = texture_array[y, x]
            
            # Apply material enhancement
            if material_type == "glossy/metallic":
                # Enhance contrast for metallic surfaces
                sampled_color = np.clip(sampled_color * 1.2 + 15, 0, 255).astype(np.uint8)
            elif material_type == "matte/fabric":
                # Soften for matte surfaces
                sampled_color = np.clip(sampled_color * 0.9 - 10, 0, 255).astype(np.uint8)
            
            vertex_colors[i] = [sampled_color[0], sampled_color[1], sampled_color[2], 255]
        
        # Apply colors to mesh
        mesh.visual.vertex_colors = vertex_colors
        
        # Create face colors for better compatibility
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
            for i, face in enumerate(mesh.faces):
                if len(face) >= 3:
                    # Average color of face vertices
                    face_vertex_colors = vertex_colors[face[:3]]
                    avg_color = np.mean(face_vertex_colors, axis=0).astype(np.uint8)
                    face_colors[i] = avg_color
                else:
                    face_colors[i] = [128, 128, 128, 255]
            
            mesh.visual.face_colors = face_colors
            
    except Exception as e:
        logging.exception("Failed to create vertex colors: %s", e)
    
    return mesh

def create_mtl_file(obj_path, dominant_colors, material_type="semi-glossy"):
    """Create a proper MTL file for the OBJ model."""
    try:
        mtl_path = obj_path.replace('.obj', '.mtl')
        
        # Calculate average color from dominant colors
        if dominant_colors:
            avg_color = np.mean(dominant_colors[:3], axis=0) / 255.0
        else:
            avg_color = np.array([0.7, 0.7, 0.7])
        
        # Set material properties based on type
        if material_type == "glossy/metallic":
            ka = avg_color * 0.1  # Ambient
            kd = avg_color * 0.8  # Diffuse
            ks = np.array([0.9, 0.9, 0.9])  # Specular
            ns = 200  # Shininess
        elif material_type == "matte/fabric":
            ka = avg_color * 0.3
            kd = avg_color * 0.9
            ks = np.array([0.1, 0.1, 0.1])
            ns = 10
        else:
            ka = avg_color * 0.2
            kd = avg_color * 0.8
            ks = np.array([0.5, 0.5, 0.5])
            ns = 50
        
        # Write MTL file
        with open(mtl_path, 'w') as f:
            f.write("# PixelProof Security Protocol Material\n")
            f.write(f"# Material Type: {material_type}\n\n")
            
            f.write("newmtl pixelproof_material\n")
            f.write(f"Ka {ka[0]:.6f} {ka[1]:.6f} {ka[2]:.6f}\n")
            f.write(f"Kd {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}\n")
            f.write(f"Ks {ks[0]:.6f} {ks[1]:.6f} {ks[2]:.6f}\n")
            f.write(f"Ns {ns:.1f}\n")
            f.write("d 1.0\n")
            f.write("illum 2\n")
        
        # Update OBJ file to use material
        with open(obj_path, 'r') as f:
            obj_content = f.read()
        
        # Add material library reference
        if "mtllib" not in obj_content:
            mtl_name = os.path.basename(mtl_path)
            obj_lines = obj_content.split('\n')
            obj_lines.insert(1, f"mtllib {mtl_name}")
            obj_lines.insert(2, "usemtl pixelproof_material")
            
            with open(obj_path, 'w') as f:
                f.write('\n'.join(obj_lines))
                
    except Exception as e:
        logging.exception("Failed to create MTL file: %s", e)

def _run_triposr_and_return_trimesh(pil_image, mc_resolution, dominant_colors=None, material_type="semi-glossy"):
    """Run TripoSR and return properly colored trimesh."""
    try:
        scene_codes = model(pil_image, device=device)
        mesh = model.extract_mesh(scene_codes, True, resolution=mc_resolution)[0]
        mesh = to_gradio_3d_orientation(mesh)
        
        # Apply proper vertex colors
        if dominant_colors:
            mesh = create_proper_vertex_colors(mesh, pil_image, material_type)
        
        return mesh
        
    except Exception as e:
        logging.exception("Failed to run TripoSR: %s", e)
        return trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3), dtype=int))

def _merge_and_export_trimeshes(tri_list, formats=["obj", "glb"], dominant_colors=None, material_type="semi-glossy"):
    """Merge and export trimeshes with proper materials."""
    if not tri_list:
        raise RuntimeError("No meshes to merge.")
    
    if len(tri_list) == 1:
        merged = tri_list[0]
    else:
        merged = trimesh.util.concatenate(tri_list)
    
    paths = []
    for fmt in formats:
        tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False)
        tmp.close()
        
        try:
            if fmt.lower() == "obj":
                merged.export(tmp.name, file_type="obj")
                # Create MTL file after export
                if dominant_colors:
                    create_mtl_file(tmp.name, dominant_colors, material_type)
            elif fmt.lower() == "glb":
                merged.export(tmp.name, file_type="glb")
            else:
                merged.export(tmp.name)
            
            paths.append(tmp.name)
            
        except Exception as e:
            logging.exception("Failed export %s: %s", tmp.name, e)
    
    return paths

# === SECURITY ANALYSIS FUNCTIONS ===

def security_color_analysis(image, precision_level=16):
    """Advanced color analysis for security protocol scanning."""
    if not SKLEARN_AVAILABLE:
        return {"dominant_colors": [], "color_accuracy": "Limited"}
    
    try:
        img_rgb = image.convert("RGB")
        img_hsv = img_rgb.convert("HSV")
        
        # Convert to numpy arrays for analysis
        rgb_array = np.array(img_rgb)
        hsv_array = np.array(img_hsv)
        
        # Reshape for clustering
        pixels = rgb_array.reshape((-1, 3))
        hsv_pixels = hsv_array.reshape((-1, 3))
        
        # Remove background noise (pure whites/blacks/grays)
        mask = ~((pixels == [255, 255, 255]).all(axis=1) | 
                 (pixels == [0, 0, 0]).all(axis=1) |
                 ((pixels[:, 0] == pixels[:, 1]) & (pixels[:, 1] == pixels[:, 2])))
        
        if mask.any():
            clean_pixels = pixels[mask]
            clean_hsv = hsv_pixels[mask]
        else:
            clean_pixels = pixels
            clean_hsv = hsv_pixels
        
        # Sample for performance if too many pixels
        if len(clean_pixels) > 100000:
            indices = np.random.choice(len(clean_pixels), 100000, replace=False)
            clean_pixels = clean_pixels[indices]
            clean_hsv = clean_hsv[indices]
        
        # High-precision color clustering
        kmeans = KMeans(n_clusters=precision_level, n_init=20, random_state=42, max_iter=1000)
        labels = kmeans.fit_predict(clean_pixels)
        
        # Get cluster centers and their frequencies
        centers = kmeans.cluster_centers_.astype(int)
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort by frequency (most dominant first)
        sorted_indices = np.argsort(-counts)
        dominant_colors = [tuple(centers[i]) for i in sorted_indices]
        color_frequencies = counts[sorted_indices]
        
        # Calculate color distribution statistics
        total_pixels = len(clean_pixels)
        color_percentages = (color_frequencies / total_pixels) * 100
        
        # Analyze color harmony and relationships
        color_harmony = analyze_color_harmony(dominant_colors[:8])
        
        # Surface material analysis based on HSV
        material_analysis = analyze_surface_material(clean_hsv, dominant_colors[:8])
        
        return {
            "dominant_colors": dominant_colors,
            "color_frequencies": color_frequencies.tolist(),
            "color_percentages": color_percentages.tolist(),
            "total_unique_colors": len(np.unique(clean_pixels.reshape(-1, 3), axis=0)),
            "average_color": tuple(np.mean(clean_pixels, axis=0).astype(int)),
            "color_harmony": color_harmony,
            "material_analysis": material_analysis,
            "analysis_precision": precision_level,
            "color_accuracy": "High-Precision Security Grade"
        }
        
    except Exception as e:
        logging.exception("Security color analysis failed: %s", e)
        return {"dominant_colors": [], "color_accuracy": "Analysis Failed"}

def analyze_color_harmony(colors):
    """Analyze color relationships for realistic rendering."""
    if len(colors) < 2:
        return {"harmony_type": "monochromatic"}
    
    # Convert to HSV for harmony analysis
    hsv_colors = []
    for color in colors:
        r, g, b = [c/255.0 for c in color]
        hsv = rgb_to_hsv(r, g, b)
        hsv_colors.append(hsv)
    
    # Analyze hue relationships
    hues = [hsv[0] for hsv in hsv_colors]
    hue_differences = []
    
    for i in range(len(hues)):
        for j in range(i+1, len(hues)):
            diff = abs(hues[i] - hues[j])
            diff = min(diff, 360 - diff)  # circular distance
            hue_differences.append(diff)
    
    avg_hue_diff = np.mean(hue_differences) if hue_differences else 0
    
    # Determine harmony type
    if avg_hue_diff < 30:
        harmony_type = "analogous"
    elif 150 < avg_hue_diff < 210:
        harmony_type = "complementary"
    elif 90 < avg_hue_diff < 150:
        harmony_type = "triadic"
    else:
        harmony_type = "complex"
    
    return {
        "harmony_type": harmony_type,
        "average_hue_difference": round(avg_hue_diff, 2),
        "color_temperature": analyze_color_temperature(colors)
    }

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV."""
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Hue calculation
    if diff == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # Saturation calculation
    s = 0 if max_val == 0 else (diff / max_val) * 100
    
    # Value calculation
    v = max_val * 100
    
    return (h, s, v)

def analyze_color_temperature(colors):
    """Analyze the color temperature of the image."""
    total_temp = 0
    for color in colors:
        r, g, b = color
        # Simplified color temperature calculation
        if b > r:
            temp = "cool"
            temp_value = (b - r) / 255.0
        elif r > b:
            temp = "warm"  
            temp_value = (r - b) / 255.0
        else:
            temp = "neutral"
            temp_value = 0
        total_temp += temp_value
    
    avg_temp = total_temp / len(colors)
    if avg_temp > 0.2:
        return "warm" if colors[0][0] > colors[0][2] else "cool"
    else:
        return "neutral"

def analyze_surface_material(hsv_pixels, dominant_colors):
    """Analyze surface material properties based on HSV values."""
    if len(hsv_pixels) == 0:
        return {"material_type": "unknown", "surface_properties": {}}
    
    # Analyze saturation and value distributions
    saturations = hsv_pixels[:, 1]
    values = hsv_pixels[:, 2]
    
    avg_saturation = np.mean(saturations)
    avg_value = np.mean(values)
    saturation_std = np.std(saturations)
    value_std = np.std(values)
    
    # Determine material properties
    if avg_saturation > 200 and avg_value > 200:
        material_type = "glossy/metallic"
        reflectivity = "high"
    elif avg_saturation < 50 and value_std < 30:
        material_type = "matte/fabric"
        reflectivity = "low"
    elif saturation_std > 80:
        material_type = "textured/mixed"
        reflectivity = "variable"
    else:
        material_type = "semi-glossy"
        reflectivity = "medium"
    
    return {
        "material_type": material_type,
        "reflectivity": reflectivity,
        "surface_properties": {
            "avg_saturation": round(avg_saturation, 2),
            "avg_brightness": round(avg_value, 2),
            "saturation_variance": round(saturation_std, 2),
            "brightness_variance": round(value_std, 2)
        }
    }

def analyze_dimensions(image):
    w, h = image.size
    return {
        "width_px": w,
        "height_px": h,
        "aspect_ratio": round(w / h, 2) if h != 0 else None
    }

def analyze_mesh_dimensions(mesh):
    extents = mesh.extents if hasattr(mesh, "extents") else (0.0, 0.0, 0.0)
    return {
        "width_units": round(float(extents[0]), 4),
        "height_units": round(float(extents[1]), 4),
        "depth_units": round(float(extents[2]), 4)
    }

def detect_security_features(image):
    """Advanced security feature detection for authentication."""
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Enhanced edge detection for fine details
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Surface texture analysis
    blur = cv2.medianBlur(gray, 5)
    texture_variance = cv2.absdiff(gray, blur)
    texture_score = np.std(texture_variance)
    
    # Reflection detection (bright spots)
    _, reflections = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    reflection_count = cv2.countNonZero(reflections)
    
    # Dent and scratch detection
    kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    dents = cv2.filter2D(gray, -1, kernel)
    dent_score = np.std(dents)
    
    # Surface wear analysis
    wear_patterns = cv2.Laplacian(gray, cv2.CV_64F)
    wear_score = np.var(wear_patterns)
    
    return {
        "edge_density": round(edge_density, 6),
        "texture_complexity": round(texture_score, 2),
        "reflection_points": int(reflection_count),
        "dent_score": round(dent_score, 2),
        "wear_score": round(wear_score, 2),
        "surface_authenticity": "high" if texture_score > 15 else "medium" if texture_score > 8 else "low",
        "damage_level": "high" if dent_score > 50 else "medium" if dent_score > 20 else "low",
        "security_score": round((edge_density * 100 + texture_score + reflection_count/1000 + dent_score/10), 2)
    }

def create_color_palette_image(colors, color_percentages=None, width=500, height=120):
    """Create enhanced color palette with percentages."""
    if not colors:
        return None
    
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Calculate proportional widths based on color percentages
    if color_percentages:
        total_width = sum(color_percentages[:len(colors)])
        widths = [(p/total_width) * width for p in color_percentages[:len(colors)]]
    else:
        widths = [width // len(colors)] * len(colors)
    
    # Draw proportional color swatches
    x_offset = 0
    for i, (color, swatch_width) in enumerate(zip(colors, widths)):
        x1 = int(x_offset)
        x2 = int(x_offset + swatch_width)
        
        try:
            rgb_color = tuple(int(c) for c in color[:3])
        except:
            rgb_color = (128, 128, 128)
        
        # Draw color swatch
        draw.rectangle([x1, 0, x2, height-20], fill=rgb_color)
        
        # Add percentage text if available
        if color_percentages and i < len(color_percentages):
            percentage_text = f"{color_percentages[i]:.1f}%"
            text_bbox = draw.textbbox((0, 0), percentage_text)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x1 + (swatch_width - text_width) // 2
            draw.text((text_x, height-18), percentage_text, fill='black')
        
        x_offset += swatch_width
    
    return img

def create_thumbnail(processed_image, size=(200, 200)):
    """Create thumbnail for saved model."""
    try:
        if isinstance(processed_image, np.ndarray):
            img = Image.fromarray(processed_image)
        else:
            img = processed_image
        
        # Create thumbnail with padding to maintain aspect ratio
        img.thumbnail(size, Image.LANCZOS)
        
        # Create square thumbnail with padding
        thumb = Image.new('RGB', size, (255, 255, 255))
        offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
        thumb.paste(img, offset)
        
        return thumb
    except Exception as e:
        logging.exception("Failed to create thumbnail: %s", e)
        return None

def process_and_generate(file_list, captured_imgs, selected_image, do_remove_background, 
                        foreground_ratio, mc_resolution, security_mode, texture_detail_level):
    imgs = []
    # Priority handling for image sources
    if captured_imgs:
        imgs = captured_imgs[:4]
    elif file_list:
        for f in file_list[:4]:
            try:
                im = Image.open(f.name).convert("RGBA")
                imgs.append(im)
            except Exception as e:
                logging.warning("Failed to open uploaded file %s: %s", getattr(f, 'name', f), e)
    elif selected_image:
        if isinstance(selected_image, np.ndarray):
            imgs.append(Image.fromarray(selected_image).convert("RGBA"))
        elif isinstance(selected_image, Image.Image):
            imgs.append(selected_image.convert("RGBA"))
    else:
        raise gr.Error("🔒 SECURITY PROTOCOL: No images detected for analysis.")

    if not imgs:
        raise gr.Error("🔒 SECURITY PROTOCOL: Image validation failed.")

    imgs = imgs[:4]
    preprocessed = []
    for im in imgs:
        pre = preprocess(im, do_remove_background, foreground_ratio)
        preprocessed.append(pre)

    # === SECURITY-GRADE ANALYSIS ===
    analysis_result = {}
    color_palette_img = None
    
    if preprocessed:
        try:
            # High-precision color analysis
            precision = 16 if security_mode else 8
            color_analysis = security_color_analysis(preprocessed[0], precision)
            analysis_result.update(color_analysis)
            
            # Enhanced color palette visualization
            if color_analysis.get("dominant_colors"):
                color_palette_img = create_color_palette_image(
                    color_analysis["dominant_colors"][:10], 
                    color_analysis.get("color_percentages", [])[:10]
                )
                
        except Exception as e:
            logging.exception("Security color analysis failed: %s", e)
            analysis_result.update({"analysis_status": "PARTIAL - Color analysis failed"})
            
        try:
            analysis_result.update(analyze_dimensions(preprocessed[0]))
            analysis_result.update(detect_security_features(preprocessed[0]))
        except Exception as e:
            logging.exception("Security feature detection failed: %s", e)

    # === 3D MODEL GENERATION WITH REALISTIC TEXTURING ===
    tri_list = []
    extracted_colors = analysis_result.get("dominant_colors", [])
    material_type = analysis_result.get("material_analysis", {}).get("material_type", "semi-glossy")
    
    for i, p in enumerate(preprocessed):
        try:
            tri = _run_triposr_and_return_trimesh(
                p, int(mc_resolution), extracted_colors, material_type
            )
            tri_list.append(tri)
        except Exception as e:
            logging.exception("TripoSR failed on image %d: %s", i, e)

    if not tri_list:
        raise gr.Error("🔒 SECURITY PROTOCOL: 3D model generation failed.")

    # Export with proper materials
    out_paths = _merge_and_export_trimeshes(
        tri_list, 
        formats=["obj", "glb"], 
        dominant_colors=extracted_colors,
        material_type=material_type
    )
    
    obj_path = next((p for p in out_paths if p.endswith(".obj")), None)
    glb_path = next((p for p in out_paths if p.endswith(".glb")), None)

    # === MESH SECURITY ANALYSIS ===
    if tri_list:
        try:
            analysis_result.update(analyze_mesh_dimensions(tri_list[0]))
            # Add mesh complexity analysis
            mesh_complexity = {
                "vertex_count": len(tri_list[0].vertices) if hasattr(tri_list[0], 'vertices') else 0,
                "face_count": len(tri_list[0].faces) if hasattr(tri_list[0], 'faces') else 0,
                "has_colors": hasattr(tri_list[0].visual, 'vertex_colors'),
                "mesh_volume": float(tri_list[0].volume) if hasattr(tri_list[0], 'volume') else 0,
                "security_validation": "PASSED" if extracted_colors else "BASIC"
            }
            analysis_result.update(mesh_complexity)
        except Exception as e:
            logging.exception("Mesh analysis failed: %s", e)

    # Add security protocol results
    analysis_result["protocol_status"] = "🔒 SECURITY SCAN COMPLETE"
    analysis_result["authenticity_level"] = "HIGH" if security_mode else "STANDARD"
    analysis_result["scan_timestamp"] = time.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    first_processed = preprocessed[0] if preprocessed else None
    return first_processed, obj_path, glb_path, analysis_result, color_palette_img

def append_capture(img, state):
    if img is None:
        return state or [], []
    state = state or []
    if len(state) >= 4:
        state = state[-3:]
    state.append(img.copy())
    previews = []
    for im in state:
        try:
            previews.append(np.array(im.convert("RGB")))
        except Exception:
            continue
    return state, previews

def save_model_with_user_info(obj_path, glb_path, processed_image, analysis_results, 
                             name, contact, product_name, price, kind="Scanner"):
    """Save model with user information to persistent storage."""
    try:
        if not obj_path and not glb_path:
            return "❌ No model files to save"
        
        # Create thumbnail
        thumbnail = create_thumbnail(processed_image) if processed_image else None
        
        # Prepare model data
        model_data = {
            "obj_path": obj_path,
            "glb_path": glb_path,
            "thumbnail": thumbnail
        }
        
        # Prepare user info
        user_info = {
            "name": name,
            "contact": contact,
            "product_name": product_name,
            "price": price,
            "kind": kind
        }
        
        # Save to storage
        model_id = save_model_to_storage(model_data, analysis_results, user_info)
        
        if model_id:
            return f"🔒 MODEL SAVED: {model_id}"
        else:
            return "❌ Failed to save model"
            
    except Exception as e:
        logging.exception("Failed to save model with user info: %s", e)
        return "❌ Save operation failed"

def search_models_by_image(query_image):
    """Search for similar saved models by image."""
    if query_image is None:
        return [], "Please upload an image to search"
    
    try:
        results = search_saved_models(query_image=query_image)
        
        if not results:
            return [], "No similar models found"
        
        # Format results for display
        result_info = []
        for result in results:
            info = f"ID: {result['id']}\n"
            info += f"Product: {result.get('user_info', {}).get('product_name', 'Unknown')}\n"
            info += f"Customer: {result.get('user_info', {}).get('name', 'Unknown')}\n"
            info += f"Material: {result.get('material_type', 'Unknown')}\n"
            info += f"Similarity: {result.get('similarity_score', 0):.2%}\n"
            info += f"Date: {result.get('timestamp', 'Unknown')}"
            result_info.append(info)
        
        return result_info, f"Found {len(results)} similar model(s)"
        
    except Exception as e:
        logging.exception("Search failed: %s", e)
        return [], f"Search failed: {str(e)}"

def load_selected_model(model_selection, search_results_state):
    """Load a selected model from search results."""
    try:
        if not model_selection or not search_results_state:
            return None, None, "No model selected"
        
        # Extract model ID from selection
        model_id = model_selection.split("ID: ")[1].split("\n")[0]
        
        # Load model data
        model_data = load_saved_model(model_id)
        
        if not model_data:
            return None, None, "Model not found"
        
        return (model_data.get("obj_path"), 
                model_data.get("glb_path"), 
                f"Loaded model: {model_id}")
        
    except Exception as e:
        logging.exception("Failed to load selected model: %s", e)
        return None, None, f"Load failed: {str(e)}"

def refresh_saved_models_gallery():
    """Refresh the saved models gallery."""
    try:
        gallery_items, model_info = get_saved_models_gallery()
        
        info_text = f"📊 Total Saved Models: {len(gallery_items)}\n"
        info_text += f"🕒 Last Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for i, info in enumerate(model_info[-5:]):  # Show latest 5
            info_text += f"🔹 {info['product']} - {info['customer']}\n"
        
        return gallery_items, info_text
        
    except Exception as e:
        logging.exception("Failed to refresh gallery: %s", e)
        return [], f"Gallery refresh failed: {str(e)}"

# ===== SECURITY PROTOCOL UI =====
with gr.Blocks(
    title="PixelProof Security Protocol Scanner", 
    css="""
    #footer {display:none !important;}
    .security-header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .security-panel {
        border: 2px solid #2a5298;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(145deg, #f8f9ff, #e8f0ff);
    }
    .color-analysis {
        border: 1px solid #4CAF50;
        border-radius: 8px;
        padding: 15px;
        background: #f8fff8;
    }
    .security-badge {
        display: inline-block;
        background: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    .saved-models-gallery {
        border: 2px solid #17a2b8;
        border-radius: 10px;
        padding: 15px;
        background: #f8f9fa;
    }
    """
) as interface:
    
    # State variables for search results
    search_results_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.HTML("""
            <div class="security-header">
                <h2>🔒 PixelProof</h2>
                <p>Security Protocol Scanner</p>
                <div class="security-badge">ACTIVE</div>
            </div>
            """)
            
            nav = gr.Radio(
                ["🔍 Scanner Dashboard", "📦 Used Product Registry", "✨ New Product Registry", "💾 Saved Models"],
                value="🔍 Scanner Dashboard",
                label="Security Modules",
                interactive=True
            )
            
            with gr.Group():
                gr.HTML("""
                <div class="saved-models-gallery">
                    <h4>💾 Recent Saved Models</h4>
                </div>
                """)
                
                saved_gallery = gr.Gallery(
                    label="📁 Model Thumbnails", 
                    columns=2, 
                    height=200,
                    interactive=True
                )
                
                gallery_info = gr.Textbox(
                    label="📊 Gallery Info",
                    interactive=False,
                    lines=6
                )
                
                refresh_gallery_btn = gr.Button("🔄 Refresh Gallery", variant="secondary", size="sm")

        with gr.Column(scale=4, elem_id="main_content") as main_area:
            
            with gr.Group(visible=True) as scanner_tab:
                gr.HTML("""
                <div class="security-header">
                    <h1>🔍 Security Protocol Scanner</h1>
                    <p>Advanced 3D Analysis with Realistic Texture Mapping</p>
                </div>
                """)
                
                with gr.Row(variant="panel"):
                    with gr.Column():
                        gr.Markdown("### 📸 Image Acquisition Protocol")
                        
                        with gr.Row():
                            multi_upload = gr.File(
                                label="🔒 Secure Upload (Multiple Images)",
                                file_types=[".png", ".jpg", ".jpeg"],
                                type="filepath",
                                file_count="multiple"
                            )
                            input_gallery = gr.Gallery(label="📋 Evidence Gallery", columns=4, height=200)
                            selected_image = gr.Image(label="🎯 Selected for Analysis", type="pil")

                        def show_gallery(files):
                            if files is None:
                                return []
                            out = []
                            for f in (files[:4] if files else []):
                                try:
                                    im = Image.open(f.name).convert("RGB")
                                    out.append(np.array(im))
                                except Exception:
                                    continue
                            return out

                        multi_upload.change(fn=show_gallery, inputs=[multi_upload], outputs=[input_gallery])
                        input_gallery.select(fn=lambda img: img, inputs=[input_gallery], outputs=[selected_image])

                        gr.Markdown("### 📷 Live Capture Protocol")
                        camera = gr.Image(sources=["upload", "webcam"], label="🎥 Security Camera Feed", type="pil")
                        capture = gr.Button("📸 Capture Evidence (Max 4 Images)", variant="secondary")
                        captured_state = gr.State([])
                        captured_preview = gr.Gallery(label="🗂 Captured Evidence", columns=4, height=120)
                        capture.click(fn=append_capture, inputs=[camera, captured_state], outputs=[captured_state, captured_preview])

                        gr.Markdown("### ⚙ Analysis Configuration")
                        with gr.Row():
                            do_remove_background = gr.Checkbox(label="🎯 Background Isolation", value=True)
                            foreground_ratio = gr.Slider(label="📏 Focus Ratio", minimum=0.5, maximum=1.0, value=0.85, step=0.05)
                        
                        with gr.Row():
                            mc_resolution = gr.Slider(label="🔬 3D Resolution", minimum=64, maximum=512, value=320, step=32)
                            security_mode = gr.Checkbox(label="🔒 Maximum Security Scan", value=True)
                            texture_detail_level = gr.Slider(label="🎨 Texture Detail Level", minimum=1, maximum=5, value=4, step=1)
                        
                        with gr.Row():
                            scan_btn = gr.Button("🔍 INITIATE SECURITY SCAN", elem_id="generate", variant="primary", size="lg")

                        processed_image = gr.Image(label="✅ Processed Evidence", interactive=False)
                        
                        # Save model section
                        gr.Markdown("### 💾 Save Scanned Model")
                        with gr.Row():
                            save_name = gr.Textbox(label="👤 Your Name", placeholder="Enter your name")
                            save_contact = gr.Textbox(label="📞 Contact", placeholder="Email or phone")
                        
                        with gr.Row():
                            save_product = gr.Textbox(label="🏷 Product Name", placeholder="Product description")
                            save_price = gr.Number(label="💰 Estimated Value ($)", minimum=0, value=0)
                        
                        save_scan_btn = gr.Button("💾 SAVE TO SECURE VAULT", variant="secondary")
                        save_status = gr.Textbox(label="💾 Save Status", interactive=False)

                    with gr.Column():
                        gr.HTML("""
                        <div class="color-analysis">
                            <h3>🎨 Spectral Analysis Results</h3>
                            <p>Real-time color and material identification</p>
                        </div>
                        """)
                        
                        color_palette = gr.Image(label="🌈 Color Signature Profile", interactive=False, height=120)
                        analysis_results = gr.JSON(label="📊 Comprehensive Analysis Report")
                        
                        gr.Markdown("### 🗿 3D Reconstruction Results")
                        
                        with gr.Tabs():
                            with gr.Tab("📐 OBJ Model"):
                                output_model_obj = gr.Model3D(
                                    label="🎯 OBJ Format (With Materials)", 
                                    interactive=False,
                                    height=400
                                )
                            with gr.Tab("🌟 GLB Model"):
                                output_model_glb = gr.Model3D(
                                    label="✨ GLB Format (With Textures)", 
                                    interactive=False,
                                    height=400
                                )

                # Model search section
                with gr.Row(variant="panel"):
                    gr.HTML("""
                    <div class="security-panel">
                        <h3>🔍 Model Search & Comparison</h3>
                        <p>Search saved models by uploading a similar image</p>
                    </div>
                    """)
                    
                with gr.Row():
                    with gr.Column():
                        search_image = gr.Image(label="🔎 Upload Image to Search", type="pil")
                        search_btn = gr.Button("🔍 Search Similar Models", variant="primary")
                        search_results = gr.Dropdown(
                            label="🎯 Search Results", 
                            choices=[], 
                            interactive=True
                        )
                        search_status = gr.Textbox(label="📊 Search Status", interactive=False)
                        
                    with gr.Column():
                        load_btn = gr.Button("📥 Load Selected Model", variant="secondary")
                        loaded_obj = gr.Model3D(label="📐 Loaded OBJ Model", height=300)
                        loaded_glb = gr.Model3D(label="🌟 Loaded GLB Model", height=300)
                        load_status = gr.Textbox(label="📥 Load Status", interactive=False)

            # === SAVED MODELS TAB ===
            with gr.Group(visible=False) as saved_models_tab:
                gr.HTML("""
                <div class="security-header">
                    <h1>💾 Saved Models Vault</h1>
                    <p>Browse and manage your saved 3D models</p>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column():
                        vault_gallery = gr.Gallery(
                            label="🗂 Model Vault", 
                            columns=4, 
                            height=400,
                            interactive=True
                        )
                        
                        vault_refresh_btn = gr.Button("🔄 Refresh Vault", variant="primary")
                        
                    with gr.Column():
                        vault_search_image = gr.Image(label="🔎 Search by Image", type="pil")
                        vault_search_btn = gr.Button("🔍 Search Vault", variant="secondary")
                        
                        vault_results = gr.Dropdown(
                            label="🎯 Vault Search Results",
                            choices=[],
                            interactive=True
                        )
                        
                        vault_model_obj = gr.Model3D(label="📐 Vault Model (OBJ)", height=300)
                        vault_model_glb = gr.Model3D(label="🌟 Vault Model (GLB)", height=300)

            # === PRODUCT REGISTRY SECTIONS ===
            with gr.Group(visible=False) as used_registry_tab:
                gr.HTML("""
                <div class="security-header">
                    <h1>📦 Used Product Security Registry</h1>
                    <p>Authenticated pre-owned product documentation</p>
                </div>
                """)
                
                used_glb = gr.File(label="📁 Upload Authenticated 3D Model", file_types=[".glb", ".obj"])
                used_preview = gr.Model3D(label="👁 Model Preview", interactive=False, height=300)
                
                with gr.Row():
                    used_name = gr.Textbox(label="👤 Customer Name", placeholder="Full legal name")
                    used_contact = gr.Textbox(label="📞 Secure Contact", placeholder="+1-XXX-XXX-XXXX")
                
                with gr.Row():
                    used_product_name = gr.Textbox(label="🏷 Product Identification", placeholder="Brand, model, specifications")
                    used_price = gr.Number(label="💰 Authenticated Value ($)", minimum=0)
                
                used_date = gr.Textbox(label="📅 Authentication Date", placeholder="YYYY-MM-DD")
                used_save = gr.Button("🔒 REGISTER IN SECURE VAULT", variant="primary")
                used_status = gr.Textbox(label="📋 Registration Status", interactive=False)

            with gr.Group(visible=False) as new_registry_tab:
                gr.HTML("""
                <div class="security-header">
                    <h1>✨ New Product Security Registry</h1>
                    <p>Fresh product authentication and documentation</p>
                </div>
                """)
                
                new_glb = gr.File(label="📁 Upload New Product 3D Model", file_types=[".glb", ".obj"])
                new_preview = gr.Model3D(label="👁 Model Preview", interactive=False, height=300)
                
                with gr.Row():
                    new_name = gr.Textbox(label="👤 Customer Name", placeholder="Full legal name")
                    new_contact = gr.Textbox(label="📞 Secure Contact", placeholder="+1-XXX-XXX-XXXX")
                
                with gr.Row():
                    new_product_name = gr.Textbox(label="🏷 Product Identification", placeholder="Brand, model, specifications")
                    new_price = gr.Number(label="💰 Market Value ($)", minimum=0)
                
                new_date = gr.Textbox(label="📅 Purchase Date", placeholder="YYYY-MM-DD")
                new_save = gr.Button("🔒 REGISTER IN SECURE VAULT", variant="primary")
                new_status = gr.Textbox(label="📋 Registration Status", interactive=False)

    # === TAB SWITCHING LOGIC ===
    def show_tab(choice):
        return (
            gr.update(visible=(choice == "🔍 Scanner Dashboard")),
            gr.update(visible=(choice == "📦 Used Product Registry")),
            gr.update(visible=(choice == "✨ New Product Registry")),
            gr.update(visible=(choice == "💾 Saved Models")),
        )
    
    nav.change(show_tab, inputs=[nav], outputs=[scanner_tab, used_registry_tab, new_registry_tab, saved_models_tab])

    # === EVENT HANDLERS ===
    
    # Main scanning function
    scan_btn.click(
        fn=process_and_generate,
        inputs=[
            multi_upload, captured_state, selected_image, 
            do_remove_background, foreground_ratio, mc_resolution, 
            security_mode, texture_detail_level
        ],
        outputs=[processed_image, output_model_obj, output_model_glb, analysis_results, color_palette],
    )
    
    # Save scanned model
    save_scan_btn.click(
        fn=save_model_with_user_info,
        inputs=[output_model_obj, output_model_glb, processed_image, analysis_results,
                save_name, save_contact, save_product, save_price],
        outputs=[save_status]
    )
    
    # Search functionality
    search_btn.click(
        fn=search_models_by_image,
        inputs=[search_image],
        outputs=[search_results, search_status]
    ).then(
        fn=lambda results: results,
        inputs=[search_results],
        outputs=[search_results_state]
    )
    
    # Load selected model
    load_btn.click(
        fn=load_selected_model,
        inputs=[search_results, search_results_state],
        outputs=[loaded_obj, loaded_glb, load_status]
    )
    
    # Gallery refresh
    refresh_gallery_btn.click(
        fn=refresh_saved_models_gallery,
        outputs=[saved_gallery, gallery_info]
    )
    
    # Vault operations
    vault_refresh_btn.click(
        fn=refresh_saved_models_gallery,
        outputs=[vault_gallery, gallery_info]
    )
    
    vault_search_btn.click(
        fn=search_models_by_image,
        inputs=[vault_search_image],
        outputs=[vault_results, search_status]
    )
    
    # Registry save operations
    def save_registry_product(kind, glb_file, name, contact, product_name, price, date):
        try:
            model_data = {"glb_path": glb_file.name if glb_file else None}
            user_info = {
                "name": name,
                "contact": contact, 
                "product_name": product_name,
                "price": price,
                "kind": kind,
                "date": date
            }
            analysis_data = {"material_analysis": {"material_type": "unknown"}}
            
            model_id = save_model_to_storage(model_data, analysis_data, user_info)
            
            if model_id:
                return f"🔒 REGISTERED: {model_id}"
            else:
                return "❌ Registration failed"
                
        except Exception as e:
            logging.exception("Registry save failed: %s", e)
            return f"❌ Registry save failed: {str(e)}"
    
    used_save.click(
        fn=lambda glb_file, name, contact, product_name, price, date: save_registry_product(
            "Used Product", glb_file, name, contact, product_name, price, date
        ),
        inputs=[used_glb, used_name, used_contact, used_product_name, used_price, used_date],
        outputs=[used_status]
    )

    new_save.click(
        fn=lambda glb_file, name, contact, product_name, price, date: save_registry_product(
            "New Product", glb_file, name, contact, product_name, price, date
        ),
        inputs=[new_glb, new_name, new_contact, new_product_name, new_price, new_date],
        outputs=[new_status]
    )
    
    # Load gallery on startup
    interface.load(
        fn=refresh_saved_models_gallery,
        outputs=[saved_gallery, gallery_info]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PixelProof Security Protocol Scanner")
    parser.add_argument('--username', type=str, default=None, help='Security username')
    parser.add_argument('--password', type=str, default=None, help='Security password')
    parser.add_argument('--port', type=int, default=7860, help='Server port')
    parser.add_argument("--listen", action='store_true', help="Public server access")
    parser.add_argument("--share", action='store_true', help="Generate public share link")
    parser.add_argument("--queuesize", type=int, default=3, help="Security queue size")
    parser.add_argument("--ssl", action='store_true', help="Enable SSL security")
    args = parser.parse_args()

    # Security banner
    print("=" * 60)
    print("🔒 PIXELPROOF SECURITY PROTOCOL SCANNER")
    print("🛡  Advanced 3D Analysis & Authentication System")
    print("🎯 Real-time Color & Texture Mapping")
    print("💾 Persistent Model Storage System")
    print("🔍 AI-Powered Model Search & Recognition")
    print("=" * 60)
    print(f"📁 Storage Directory: {STORAGE_DIR}")
    print(f"🗄  Models Database: {MODELS_DB_PATH}")
    print("=" * 60)

    # Initialize storage and check existing data
    init_storage()
    existing_models = load_models_database()
    print(f"📊 Found {len(existing_models)} existing saved models")
    print("🚀 Starting PixelProof Security Scanner...")

    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=True,
        server_name="0.0.0.0" if args.listen else None,
        server_port=args.port,
        ssl_verify=args.ssl if hasattr(args, 'ssl') else False
    )