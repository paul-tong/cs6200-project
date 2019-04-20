import os

RELEVANCE_FILE_PATH = "../data/cacm.rel.txt"
RESULT_FILE_PATH = "../data/result"
QUERY_HEAD = "Query"

relevanceMap = {}
precisionMap = {}
recallMap = {}
reciprocalMap = {}
mapMap = {}
mrrMap = {}


# read relevance judgement file and build relevance Map<queryId, Set<related docs>>
def buildRelevanceMap(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]

    for line in lines:
        line = line.split()
        queryId = QUERY_HEAD + line[0]
        docId = line[2]

        if queryId not in relevanceMap:
            relevanceMap[queryId] = set()
        relevanceMap[queryId].add(docId)


# build statistics map
def buildstatisticsMap(inputPath):
    print("building maps: ")
    # get names of different runs (folder names)
    runIds = [name for name in os.listdir(inputPath)]

    for runId in runIds:
        runPath = inputPath + "/" + runId
        queryIds = [name for name in os.listdir(runPath)]
        for queryId in queryIds:
            filePath = runPath + "/" + queryId
            queryId = queryId.split('.')[0]
            buildstatisticsMapEachQuery(filePath, runId, queryId)
            print(runId, queryId, filePath)



# given each query, build statistics maps for precision, recall and reciprocal
def buildstatisticsMapEachQuery(filepath, runId, queryId):
    # create a new entry for the specific run if it's not in the map previously
    if runId not in precisionMap:
        precisionMap[runId] = {}
    if runId not in recallMap:
        recallMap[runId] = {}
    if runId not in reciprocalMap:
        reciprocalMap[runId] = {}

    # exclude query that has no relevant docs
    if queryId not in relevanceMap:
        return

    # read lines from file
    with open(filepath) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]

    queryRelevantCount = len(relevanceMap[queryId]) # number of docs relevant to this query
    retrievalRelevantCount = 0 # number of relevant docs retrieval so far
    retrievalTotalCount = 0 # number of total docs retrieval so far
    precisionList = [] # list of precision[docId, score] for this query and run
    recallList = [] # list of recall[docId, score] for this query and run

    # for each result doc in the query
    for index, line in enumerate(lines):
        # escape first line
        if index == 0:
            continue

        line = line.split(',')

        docId = line[2]
        retrievalTotalCount += 1
        if docId in relevanceMap[queryId]:
            retrievalRelevantCount += 1

            # compute reciprocal
            if retrievalRelevantCount == 1:
                reciprocalMap[runId][queryId] = 1 / index
        precisionList.append([docId, retrievalRelevantCount / retrievalTotalCount])
        recallList.append([docId, retrievalRelevantCount / queryRelevantCount])

    precisionMap[runId][queryId] = precisionList
    recallMap[runId][queryId] = recallList

    '''print(runId, queryId, precisionList)
    print(runId, queryId, recallList)
    print(runId, queryId, 1 / index)'''


# compute MAP for each run, save into map<runId, MAP score>
def computeMAP():
    for runId, queries in precisionMap.items():
        averagePrecisionSum = 0 # sum of average precision score of this run
        for queryId, precisionScores in queries.items():
            # exclude query that has no relevant docs
            if queryId not in relevanceMap:
                continue

            precisionSum = 0  # sum of precision scores of retrieval relevant docs in this query
            relevantCount = 0  # number of retrieval relevant docs in this query
            for precisionScore in precisionScores:
                docId = precisionScore[0]
                score = precisionScore[1]
                if docId in relevanceMap[queryId]:
                    precisionSum += score
                    relevantCount += 1

            # compute average precision for this query and add to sum
            averagePrecision = 0
            if relevantCount != 0:
                averagePrecision = precisionSum / relevantCount
            averagePrecisionSum += averagePrecision

        # compute MAP for this run, and add to map
        mapScore = averagePrecisionSum / len(queries)
        mapMap[runId] = mapScore



# compute MRR for each run, save into map<runId, MRR score>
def computeMRR():
    for runId, queries in reciprocalMap.items():
        reciprocalScoreSum = 0 # sum of reciprocal score of this run
        for queryId, score in queries.items():
            reciprocalScoreSum += score

        mrrScore = reciprocalScoreSum / len(queries)
        mrrMap[runId] = mrrScore



# build relevance map
buildRelevanceMap(RELEVANCE_FILE_PATH)

# build statistics map
buildstatisticsMap(RESULT_FILE_PATH)

computeMAP()
print("\nMAP scores: \n", mapMap)

computeMRR()
print("\nMRR scores: \n", mrrMap)

print("\nprecision and recall socres:")
print(precisionMap["BM25"]["Query1"])
print(precisionMap["BM25"]["Query2"])
print(recallMap["BM25"]["Query1"])
print(recallMap["BM25"]["Query2"])
'''
print(precisionMap["TF"]["query1"])
print(precisionMap["TF"]["query2"])
print(recallMap["TF"]["query1"])
print(recallMap["TF"]["query2"])'''