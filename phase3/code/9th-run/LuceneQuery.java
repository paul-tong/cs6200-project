package com.ir.createIndex;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.SimpleAnalyzer;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocsEnum;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.Version;

/**
 * To create Apache Lucene index in a folder and add files into this index based
 * on the input of the user.
 */
public class LuceneQuery {
    private static Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_47);
    private static Analyzer sAnalyzer = new SimpleAnalyzer(Version.LUCENE_47);

    private IndexWriter writer;
    private ArrayList<File> queue = new ArrayList<File>();
    private String indexDir;
    Set<String> stopwordSet;

    /**
     * Constructor
     * 
     * @param indexDir
     *            the name of the folder in which the index should be created
     * @throws java.io.IOException
     *             when exception creating index.
     */
    LuceneQuery(String indexDir) throws IOException {
    	this.indexDir = indexDir;
    	
		FSDirectory dir = FSDirectory.open(new File(indexDir));
	
		IndexWriterConfig config = new IndexWriterConfig(Version.LUCENE_47,
			analyzer);
	
		writer = new IndexWriter(dir, config);
		
		stopwordSet = new HashSet<String>();
    }

    /**
     * Indexes a file or directory
     * 
     * @param fileName
     *            the name of a text file or a folder we wish to add to the
     *            index
     * @throws java.io.IOException
     *             when exception
     */
    public void indexFileOrDirectory(String fileName) throws IOException {
		// ===================================================
		// gets the list of files in a folder (if user has submitted
		// the name of a folder) or gets a single file name (is user
		// has submitted only the file name)
		// ===================================================
		addFiles(new File(fileName));
	
		int originalNumDocs = writer.numDocs();
		for (File f : queue) {
		    FileReader fr = null;
		    try {
			Document doc = new Document();
	
			// ===================================================
			// add contents of file
			// ===================================================
			fr = new FileReader(f);			
			doc.add(new TextField("contents", fr));
			doc.add(new StringField("path", f.getPath(), Field.Store.YES));
			doc.add(new StringField("filename", f.getName(),
				Field.Store.YES));
	
			writer.addDocument(doc);
			System.out.println("Added: " + f);
		    } catch (Exception e) {
			System.out.println("Could not add: " + f);
		    } finally {
			fr.close();
		    }
		}
	
		int newNumDocs = writer.numDocs();
		System.out.println("");
		System.out.println("************************");
		System.out
			.println((newNumDocs - originalNumDocs) + " documents added.");
		System.out.println("************************");
	
		queue.clear();
    }

    private void addFiles(File file) {
		if (!file.exists()) {
		    System.out.println(file + " does not exist.");
		}
		if (file.isDirectory()) {
		    for (File f : file.listFiles()) {
			addFiles(f);
		    }
		} else {
		    String filename = file.getName().toLowerCase();
		    // ===================================================
		    // Only index text files
		    // ===================================================
		    if (filename.endsWith(".htm") || filename.endsWith(".html")
			    || filename.endsWith(".xml") || filename.endsWith(".txt")) {
			queue.add(file);
		    } else {
			System.out.println("Skipped " + filename);
		    }
		}
    }

    /**
     * Close the index.
     * 
     * @throws java.io.IOException
     *             when exception closing
     */
    public void closeIndex() throws IOException {
    	writer.close();
    }
  
    /**
     * create index for docs in the given location
     * @param docsDir
     * @throws IOException
     */
    public void indexingDocs(String docsDir) throws IOException {
    	// add docs for indexing
	    try {
	    	indexFileOrDirectory(docsDir);
	    } catch (Exception e) {
	    	System.out.println("Error indexing " + docsDir + " : " + e.getMessage());
	    }
	    
		// ===================================================
		// after adding, we always have to call the
		// closeIndex, otherwise the index is not created
		// ===================================================
		closeIndex();
    }
    
    /**
     * query given term in the index, write top k results into given location
     * @param queryTerm, terms need to query
     * @param topK, number of top results
     * @param resultDir, location of saving results
     */
    public void query(String queryId, String queryTerm, int topK, String resultDir) throws IOException, ParseException {
		IndexReader reader = DirectoryReader.open(FSDirectory.open(new File(this.indexDir)));
		IndexSearcher searcher = new IndexSearcher(reader);
		TopScoreDocCollector collector = TopScoreDocCollector.create(topK, true); 
	
		System.out.println("Getting top " + topK + " results for querying [" + queryTerm + "]...");
		
		// query top results based on the default score computation
		Query q = new QueryParser(Version.LUCENE_47, "contents",analyzer).parse(queryTerm);
		searcher.search(q, collector);
		ScoreDoc[] hits = collector.topDocs().scoreDocs;
	    
		// write results to file
		StringBuilder sb = new StringBuilder();
		sb.append("Query:" + queryTerm + "\n");
		for (int i = 0; i < hits.length; ++i) {
		    int docId = hits[i].doc;
	    	
		    //System.out.println("doc id: " + docId);
		    Document d = searcher.doc(docId);
		    String docName = d.get("filename").split(".txt")[0];
		    double docScore = hits[i].score;
		    
		    sb.append(queryId + "," + "Q0," + docName.split("\\.")[0] + "," + String.valueOf(i + 1) + "," + docScore + ",Lucene" + "\n");
		    //System.out.println((i + 1) + ". " + "name=" + docName + " score=" + docScore);
		}
		
		BufferedWriter bwr = new BufferedWriter(new FileWriter(new File(resultDir)));
		bwr.write(sb.toString());
		bwr.flush();
		bwr.close();
    }
    
    
    // class that associates each term with its frequency
    private class FrequencyTerm{
    	String term;
    	int frequency;
    	
    	FrequencyTerm(String t, int f) {
    		term = t;
    		frequency = f;
    	}
    }
    
    // expand query with pesudo-relevancy-feedback 
    public void queryExpanding(String queryTerm, String docsDir, String resultDir, int topK) throws IOException, ParseException {
    	int expansionTermNum = 50; // number of terms considered for expansion
    	
    	System.out.println("\nquery espansion for: " + queryTerm);
    	
    	// get the term frequency from the most frequent documents, save it into a map
    	Map<String, Integer> termMap = new HashMap<>();
		IndexReader reader = DirectoryReader.open(FSDirectory.open(new File(this.indexDir)));
		IndexSearcher searcher = new IndexSearcher(reader);
		TopScoreDocCollector collector = TopScoreDocCollector.create(topK, true);
		
		// query top results based on the default score computation
		Query q = new QueryParser(Version.LUCENE_47, "contents",analyzer).parse(queryTerm);
		searcher.search(q, collector);
		ScoreDoc[] hits = collector.topDocs().scoreDocs;
	    
		// for each document, analyze its contents and count its term frequency
		for (int i = 0; i < hits.length; ++i) {
			//PriorityQueue<FrequencyTerm> pq = new PriorityQueue<>();
		    int docId = hits[i].doc;
	    	Document d = searcher.doc(docId);
		    String filepath = docsDir + "/" + d.get("filename");
		    System.out.println("filename: " + filepath);
		    
	        String content = "";
	        try
	        {
	            content = new String ( Files.readAllBytes( Paths.get(filepath) ) );
	        }
	        catch (IOException e)
	        {
	            e.printStackTrace();
	        }
	        
	        // add term into map
	        for (String term : content.split(" ")) {
	        	// ignore this term if its in default stopword set
	        	if(StopAnalyzer.ENGLISH_STOP_WORDS_SET.contains(term)) {
	        		continue;
	        	}
	        	
	        	termMap.put(term, termMap.getOrDefault(term, 0) + 1);
	        }
		}
		
		// get the terms with highest frequency with priority queue(min heap)
		Comparator<FrequencyTerm> termComparator = new Comparator<FrequencyTerm>() {
			public int compare(FrequencyTerm t1, FrequencyTerm t2) {
				return t1.frequency - t2.frequency;
			}
		};
		
		PriorityQueue<FrequencyTerm> pq = new PriorityQueue<>(termComparator);

		for (Map.Entry<String, Integer> e : termMap.entrySet()) {
			FrequencyTerm ft = new FrequencyTerm(e.getKey(), e.getValue());
			pq.offer(ft);
			if (pq.size() > expansionTermNum) {
				pq.poll();
			}
			//System.out.println("term: " + e.getKey() + " frequency: " + e.getValue());
		}		
		
		// write terms into file
		StringBuilder sb = new StringBuilder();
		while (!pq.isEmpty()) {
			FrequencyTerm ft = pq.poll();
			sb.insert(0, ft.term + "\n");
			//System.out.println("term: " + ft.term + " frequency: " + ft.frequency);
		}
		
		BufferedWriter bwr = new BufferedWriter(new FileWriter(new File(resultDir)));
		bwr.write(sb.toString());
		bwr.flush();
		bwr.close();
    }
   
    // apply stopping on given query term
    private String stopping(String queryTerm) {
    	String[] terms = queryTerm.split(" ");
    	StringBuilder sb = new StringBuilder();
    	
    	for (String term : terms) {
    		term = term.trim().toLowerCase();
    		
    		// escape this term if it is in stop list
    		if (stopwordSet.contains(term)) {
    			continue;
    		}
    		
    		// escape this term if it contains special character
    		for (char c : term.toCharArray()) {
    			if (!Character.isLetterOrDigit(c)) {
    				term = "";
    				break;
    			}
    		}
    		
    		if (term.length() > 0) {
    			sb.append(term + " ");
    		}
    	}
    	
    	return sb.toString();
    }
    
    public static void main(String[] args) throws IOException, ParseException {
    	int TOPK = 100;
    	String QUERY_DIR = "/home/tong/ProgramFile/eclipseProjects/createIndex/query.txt";
    	String INDEX_DIR = "/home/tong/ProgramFile/eclipseProjects/createIndex/index";
    	String DOCS_DIR = "/home/tong/ProgramFile/eclipseProjects/createIndex/docs";
    	String RESULT_DIR = "/home/tong/ProgramFile/eclipseProjects/createIndex/result"; // result without expansion
    	String EXPANSION_DIR = "/home/tong/ProgramFile/eclipseProjects/createIndex/expansion"; // expansion terms list
    	String RESULT_EXPANSION_DIR = "/home/tong/ProgramFile/eclipseProjects/createIndex/resultExpansion"; // result with query expansion
    	String STOP_LIST_DIR = "/home/tong/ProgramFile/eclipseProjects/createIndex/common_words.txt";
    	String indexDir = INDEX_DIR;
    	String docsDir = DOCS_DIR;
    	

    	// create indexer with given index location
    	LuceneQuery indexer = null;
		try {
		    indexer = new LuceneQuery(indexDir);
		} catch (Exception ex) {
		    System.out.println("Cannot create index..." + ex.getMessage());
		    System.exit(-1);
		}

		
		// indexing docs
		//indexer.indexingDocs(docsDir);
		
		// read queries from file, and save to map<queryId, queryCountent>
		Map<String, String> queryMap = new HashMap<>();
		try {
			File file = new File(QUERY_DIR);
			FileReader fileReader = new FileReader(file);
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			StringBuffer stringBuffer = new StringBuffer();
			String line;
			while ((line = bufferedReader.readLine()) != null) {
				String queryId = line.split(",")[0];
				String queryContent = line.split(",")[1].trim();
				queryMap.put(queryId, queryContent);
			}
			fileReader.close();
			System.out.println("query map:");
			System.out.println(queryMap.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// read stop list from file, and save to Set
		try {
			File file = new File(STOP_LIST_DIR);
			FileReader fileReader = new FileReader(file);
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			StringBuffer stringBuffer = new StringBuffer();
			String line;
			while ((line = bufferedReader.readLine()) != null) {
				indexer.stopwordSet.add(line.trim());
			}
			fileReader.close();
			System.out.println("stop set:");
			System.out.println(indexer.stopwordSet.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// get query expansion
		for (Map.Entry<String, String> e : queryMap.entrySet()) {
			String queryId = e.getKey();
			String queryTerm = e.getValue();
	    	String expansionDir = EXPANSION_DIR + "/" + queryId + ".txt";
			indexer.queryExpanding(queryTerm, docsDir, expansionDir, 10); // expand query terms from 10 most frequent docs	
		}
		
		// query with expansion and stopping
		for (Map.Entry<String, String> e : queryMap.entrySet()) {
			String queryId = e.getKey();
			String queryIdNumber = queryId.substring(5);
			String queryTerm = e.getValue();

	    	String expansionPath = EXPANSION_DIR + "/" + queryId + ".txt";
	    	String resultDir = RESULT_EXPANSION_DIR + "/" + queryId + ".txt";
	    	
	    	// read expansion terms line by line, append to generate new query terms
	    	File file = new File(expansionPath);
			FileReader fileReader = new FileReader(file);
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			StringBuilder sb = new StringBuilder();
			String line;
			while ((line = bufferedReader.readLine()) != null) {
				sb.append(" " + line);
			}
			fileReader.close();
			
			queryTerm += sb.toString();
	
			// apply stopping on query term
			queryTerm = indexer.stopping(queryTerm);
			
			// apply query
			indexer.query(queryIdNumber, queryTerm, TOPK, resultDir); // query top 100 results for given term	
		}
    }
}