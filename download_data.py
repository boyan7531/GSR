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
