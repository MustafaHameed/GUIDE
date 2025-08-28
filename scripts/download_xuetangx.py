"""Download XuetangX MOOC dataset from Kaggle."""

import kagglehub
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_xuetangx_dataset():
    """Download and organize XuetangX MOOC dataset."""
    
    # Create target directory
    target_dir = Path("data/xuetangx/raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading XuetangX MOOC dataset from Kaggle...")
    
    try:
        # Download latest version
        downloaded_path = kagglehub.dataset_download("anasnofal/mooc-data-xuetangx")
        logger.info(f"Downloaded to: {downloaded_path}")
        
        # Copy files to our data structure
        source_path = Path(downloaded_path)
        
        for file in source_path.glob("*"):
            if file.is_file():
                dest_file = target_dir / file.name
                shutil.copy2(file, dest_file)
                logger.info(f"Copied {file.name} to {dest_file}")
        
        logger.info(f"XuetangX dataset organized in: {target_dir}")
        
        # List downloaded files
        files = list(target_dir.glob("*"))
        logger.info(f"Downloaded {len(files)} files:")
        for file in files:
            logger.info(f"  - {file.name}")
            
        return target_dir
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_xuetangx_dataset()