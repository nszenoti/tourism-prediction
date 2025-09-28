from src.utils.hf_utils import HFHandler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def upload_webapp():
    """Upload webapp files to HuggingFace Space"""
    logger.info("Starting webapp deployment to HuggingFace...")

    hf = HFHandler()

    try:
        # Upload entire webapp folder
        hf.upload(
            path="src/webapp",
            repo_name="tourism-prediction",
            repo_type="space",
            space_sdk="streamlit"
        )
        logger.info("Webapp successfully uploaded to HuggingFace Space")

    except Exception as e:
        logger.error(f"Error uploading webapp: {str(e)}")
        raise

if __name__ == "__main__":
    upload_webapp()
