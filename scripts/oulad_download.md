# OULAD Download Script Instructions

## Overview
This document provides instructions for creating a Python script to download the official Open University Learning Analytics Dataset (OULAD) from the Open University website.

## Script Requirements

Create a Python script that:

1. **Downloads the official OULAD zip file** from the Open University data repository
   - Source: https://analyse.kmi.open.ac.uk/open-dataset
   - Direct download URL: https://analyse.kmi.open.ac.uk/open-dataset/download

2. **Verifies data integrity** (if MD5 checksums are available)
   - Check file integrity after download
   - Provide warning if verification fails

3. **Extracts data to the correct location**
   - Extract all CSV files to `data/oulad/raw/`
   - Preserve original file structure and names

4. **Prints summary information**
   - List all extracted CSV files
   - Show file sizes and basic statistics
   - Display table schemas if possible

## Expected Output Files

The script should extract these core OULAD tables:
- `studentInfo.csv` - Student demographic and registration information
- `studentVle.csv` - Student interactions with Virtual Learning Environment
- `vle.csv` - Virtual Learning Environment objects and metadata
- `studentRegistration.csv` - Student module registration details
- `studentAssessment.csv` - Student assessment submission records
- `assessments.csv` - Assessment metadata and details

## Implementation Notes

- Use `requests` library for HTTP downloads
- Use `zipfile` library for extraction
- Include progress bars for large downloads
- Handle network errors gracefully
- Add retry logic for failed downloads
- Log all operations for debugging

## References

- [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open-dataset)
- [OULAD Nature Scientific Data Paper](https://www.nature.com/articles/sdata2017171)
- [OULAD PMC Documentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5704676/)