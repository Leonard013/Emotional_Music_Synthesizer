"""
File utilities for the whistle-to-music synthesizer.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import hashlib
import json
import pickle
from datetime import datetime


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {'exists': False}
    
    stat = file_path.stat()
    
    return {
        'exists': True,
        'name': file_path.name,
        'stem': file_path.stem,
        'suffix': file_path.suffix,
        'size': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'is_file': file_path.is_file(),
        'is_dir': file_path.is_dir()
    }


def list_audio_files(directory: Union[str, Path], 
                    extensions: Optional[List[str]] = None) -> List[Path]:
    """
    List audio files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of audio file paths
    """
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff']
    
    directory = Path(directory)
    audio_files = []
    
    for ext in extensions:
        audio_files.extend(directory.glob(f'**/*{ext}'))
        audio_files.extend(directory.glob(f'**/*{ext.upper()}'))
    
    return sorted(audio_files)


def list_midi_files(directory: Union[str, Path]) -> List[Path]:
    """
    List MIDI files in directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of MIDI file paths
    """
    directory = Path(directory)
    midi_files = []
    
    for ext in ['.mid', '.midi']:
        midi_files.extend(directory.glob(f'**/*{ext}'))
        midi_files.extend(directory.glob(f'**/*{ext.upper()}'))
    
    return sorted(midi_files)


def get_file_hash(file_path: Union[str, Path], 
                 algorithm: str = 'md5') -> str:
    """
    Calculate file hash.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        File hash
    """
    file_path = Path(file_path)
    
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha1':
        hasher = hashlib.sha1()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def copy_file(src: Union[str, Path], 
              dst: Union[str, Path], 
              overwrite: bool = False) -> bool:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination file exists: {dst}")
    
    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(src, dst)
    return True


def move_file(src: Union[str, Path], 
              dst: Union[str, Path], 
              overwrite: bool = False) -> bool:
    """
    Move file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination file exists: {dst}")
    
    # Ensure destination directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.move(str(src), str(dst))
    return True


def delete_file(file_path: Union[str, Path]) -> bool:
    """
    Delete file.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if successful
    """
    file_path = Path(file_path)
    
    if file_path.exists():
        file_path.unlink()
        return True
    
    return False


def clean_directory(directory: Union[str, Path], 
                   pattern: Optional[str] = None,
                   confirm: bool = False) -> int:
    """
    Clean directory by removing files matching pattern.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match (e.g., '*.tmp')
        confirm: Whether to confirm before deletion
        
    Returns:
        Number of files deleted
    """
    directory = Path(directory)
    
    if not directory.exists():
        return 0
    
    if pattern:
        files_to_delete = list(directory.glob(pattern))
    else:
        files_to_delete = [f for f in directory.iterdir() if f.is_file()]
    
    if confirm and files_to_delete:
        response = input(f"Delete {len(files_to_delete)} files? (y/N): ")
        if response.lower() != 'y':
            return 0
    
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    return deleted_count


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to JSON file
        indent: JSON indentation
        
    Returns:
        True if successful
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    
    return True


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, file_path: Union[str, Path]) -> bool:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        file_path: Path to pickle file
        
    Returns:
        True if successful
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    
    return True


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load data from pickle file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes
    """
    directory = Path(directory)
    total_size = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size


def get_directory_info(directory: Union[str, Path]) -> Dict[str, Any]:
    """
    Get directory information.
    
    Args:
        directory: Directory path
        
    Returns:
        Dictionary with directory information
    """
    directory = Path(directory)
    
    if not directory.exists():
        return {'exists': False}
    
    files = list(directory.rglob('*'))
    file_count = len([f for f in files if f.is_file()])
    dir_count = len([f for f in files if f.is_dir()])
    total_size = get_directory_size(directory)
    
    return {
        'exists': True,
        'path': str(directory),
        'file_count': file_count,
        'dir_count': dir_count,
        'total_size': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'total_size_gb': total_size / (1024 * 1024 * 1024)
    }


def create_backup(file_path: Union[str, Path], 
                 backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create backup of file.
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backup (default: same directory)
        
    Returns:
        Path to backup file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if backup_dir is None:
        backup_dir = file_path.parent
    else:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name
    
    shutil.copy2(file_path, backup_path)
    return backup_path


def find_files(directory: Union[str, Path], 
               pattern: str,
               recursive: bool = True) -> List[Path]:
    """
    Find files matching pattern.
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))
