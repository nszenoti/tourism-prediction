from src.utils.hf_utils import HFHandler

def register_dataset():
    # Initialize HF handler
    hf = HFHandler()

    # Upload dataset
    hf.upload(
        path="data/raw/tourism.csv",
        repo_name="tourism-prediction",
        repo_type="dataset"
    )

if __name__ == "__main__":
    register_dataset()
