# Recommendation / Ranking System Test
## Goal
For each user, when they open the web, pick top-k to show.  
This project is mainly simulate the Recommendation/Ranking System.  

Limitation
- latency: P95 < 150ms 
- feed: N requests/s
- data: implicit feedback (click / dwell / like), has bias
## Dataset
Source: Kaggle [MIND:Microsoft News Recommendation Datset](https://www.kaggle.com/datasets/arashnic/mind-news-dataset?resource=download-directory)


Raw Data files:
- news.tsv
- behaviors.tsv

## Columns
behaviors.tsv
- Impressions ID: The ID of an impression. (Unique ID)
- User ID: The anonymous ID of a user.
- Time: The impression time with format "MM?DD?YYYY HH:MM:SS AM/PM"
- History: The news click history (ID list of clicked news) of this user before this impression. The clicked news articles are ordered by time.
- Impressions: List of news displayed in this impression and user's click behaviors on them (1 for click and 0 for non-click). The orders of news in a impressions have been shuffled.

news.tsv
- News ID
- Category
- SubCategory
- Title
- Abstract
- Title Entities (entities contained in the title of this news)
- Abstract Entities (entites contained in the abstract of this news)
