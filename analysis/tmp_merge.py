from pathlib import Path
from analysis.data_process import read_cleaned_papers, merge_cleaned_papers

incremental = read_cleaned_papers(Path('analysis/cleaned_papers_incremental.parquet'))
merge_cleaned_papers(incremental, Path('analysis/cleaned_papers.parquet'))
print('merge done', len(incremental))