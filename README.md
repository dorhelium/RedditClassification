# RedditClassification

Reddit.com is a forum and content aggregator. Our goal is to classify a reddit comment as coming from one of 20 selected subreddits (subforum communities). The training dataset is a collection of raw comments from the website with a maximum length of 10,000 characters, and may include ASCII characters, links, emojis and formatting characters, as well as which subreddit the comment came from. We are testing a variety of models and text pre-processing steps in order to maximize the classification accuracy, and also determine the effectiveness of different model types on one classification task. All models and processing algorithms except for NB were based on implementations in scikit-learn

## Dataset
The reddit comment dataset includes a comment id, the comment text and the subreddit from where the comment came. There are 20 subreddits/categories: hockey, nba, leagueoflegends, soccer, funny, movies, anime, Overwatch, trees, GlobalOffensive, nfl, AskReddit, gameofthrones, conspiracy, worldnews, wow, europe, canada, Music, baseball. The dataset is very balanced with each category having 3500 sample comments, for a total of 70000 samples.

---

[View the full report](https://github.com/dorhelium/RedditClassification/blob/master/Report.pdf)

---
