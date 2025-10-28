import trimesh

def analyze_mesh(mesh_path):
    # Load mesh (.obj / .glb)
    mesh = trimesh.load(mesh_path)

    # Extract mesh details
    face_count = len(mesh.faces)
    vertex_count = len(mesh.vertices)

    return {
        "Mesh Path": mesh_path,
        "Vertices": vertex_count,
        "Faces": face_count
    }
