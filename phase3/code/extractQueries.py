import os
import csv

QUERY_FOLDER_PATH = "../data/result/BM25/"
QUERY_OUTPUT_PATH = "../data/query.csv"
queryMap = {}

# build query map<queryId, queryContents>
def buildQueryMap(folderPath):
    filenames = [name for name in os.listdir(folderPath)]
    for filename in filenames:
        filePath = folderPath + filename
        with open(filePath) as f:
            query = f.readline()
            queryId = filename.split('.')[0]
            queryContent = query.split(':')[1].strip()
            print(queryId, queryContent)
            queryMap[queryId] = queryContent


buildQueryMap(QUERY_FOLDER_PATH)
print("\nquery map: \n", queryMap)

with open(QUERY_OUTPUT_PATH, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in queryMap.items():
       writer.writerow([key, value])