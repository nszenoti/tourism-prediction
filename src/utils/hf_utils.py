from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

class HFHandler:
    """Helper class for HuggingFace Hub operations"""

    def __init__(self, username='nszfgfg', token=None):
        self.api = HfApi(token=token or os.getenv("HF_TOKEN"))
        self.username = username

    def upload(self, path, repo_name, repo_type="model"):
        """Upload file or folder to HF

        Args:
            path: Path to file or folder
            repo_name: Name of repository
            repo_type: Type of repository ("model", "dataset", "space")
        """
        logger.info(f"Uploading {path} to {repo_name} on HuggingFace Hub...")

        repo_id = f"{self.username}/{repo_name}"

        try:
            self.api.repo_info(repo_id=repo_id, repo_type=repo_type)
        except RepositoryNotFoundError:
            self.api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

        if os.path.isfile(path):
            file_name = os.path.basename(path)
            self.api.upload_file(
                path_or_fileobj=path,
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type=repo_type
            )
        elif os.path.isdir(path):
            self.api.upload_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type=repo_type
            )

        logger.info(f"Uploaded {path} to {repo_name} on HuggingFace Hub !!!")
