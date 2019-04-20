Design description:
Fo 9th run, we combined a query expansion technique with stopping based on the lucene baseline run. First, we used technique pseudo-relevance-feedback to expand the queries. Then we applied stopping to remove the common words from the expanded queries. And then we ran these (expanded and then stopped) queries with default lucene system.
