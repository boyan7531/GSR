from pathlib import Path
import zipfile

from huggingface_hub import snapshot_download

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

download_kwargs = {
    "repo_id": "SoccerNet/SN-GSR-2025",
    "repo_type": "dataset",
    "revision": "main",
    "local_dir": "SN-GSR-2025",
}
if tqdm is not None:
    download_kwargs["tqdm_class"] = tqdm

snapshot_download(**download_kwargs)
print("Download completed successfully.")

root_dir = Path(download_kwargs["local_dir"])
zip_paths = sorted(root_dir.rglob("*.zip"))
if zip_paths:
    for zip_path in zip_paths:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(zip_path.parent)
        zip_path.unlink()
    print("Extraction completed and zip files removed.")
else:
    print("No zip files found to extract.")
