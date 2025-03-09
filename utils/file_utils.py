import os
from typing import Dict  # Add this import
from werkzeug.utils import secure_filename

def save_uploaded_files(request, work_dir: str) -> Dict[str, str]:
    files = {
        'audio1': request.files['audio1'],
        'text1': request.files['text1'],
        'audio2': request.files['audio2'],
        'text2': request.files['text2']
    }

    saved_paths = {}
    for key, file in files.items():
        filename = secure_filename(file.filename)
        save_path = os.path.join(work_dir, filename)
        file.save(save_path)
        saved_paths[key] = save_path

    return saved_paths
