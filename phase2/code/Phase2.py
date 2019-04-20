import os
import string
import operator
import BeautifulSoup

def get_common_words(common_words_file):
    """
    write common words into a list
    :param common_words_file: the file containing common words
    :return: a list containing all common words
    """
    lines = open(common_words_file).readlines()
    word_list = []
    for line in lines:
        line = line.strip()
        word_list.append(line)
    return word_list

def cut_content(html_string):
    """
    cut a paragraph content into sentences
    :param html_string: the given content string
    :return: a list of sentences
    """
    sentences = []
    split = html_string.split('.')
    for sentence in split:
        sentence = sentence.strip().replace('  ', ' ')
        sentences.append(sentence)
    return sentences

def count_sig_words(sentences, significant_words):
    """
    count the number of significant words in each sentence
    :param sentences: the given sentences
    :param significant_words: list of significant words
    :return: a list of significant words count for each sentence
    """
    sig_words_count = {}
    for sentence in sentences:
        counter = 0
        new_sentence = sentence
        for punc in string.punctuation:
            new_sentence = new_sentence.replace(punc, '')
        split = new_sentence.split(' ')
        for term in split:
            if term.lower() in significant_words:
                counter += 1
        sig_words_count[sentence] = counter
    return sig_words_count

def add_bold(sentence, significant_words):
    """
    Make significant words in the sentence bold
    :param sentence: the given sentence
    :param significant_words: the significant words list
    :return: the updated sentence with significant words bold
    """
    split = sentence.split(' ')
    new_sentence = ''
    for term in split:
        if term.lower() in significant_words:
            new_sentence = new_sentence + ' ' + '<B>' + term + '</B>'
        else:
            new_sentence = new_sentence + ' ' + term
    return new_sentence.strip()

def get_html_content(filename):
    """
    Read the content of the html file
    :param filename: the filename of html file
    :return: the content of html file
    """
    html = open(filename, 'r').read()
    soup = BeautifulSoup.BeautifulSoup(html)
    pre_tag = soup.find('pre')
    text = pre_tag.text
    text = text.split('CACM')[0].strip()
    text = text.replace('\n', ' ').strip()
    return str(text)

def get_significant_words(query, common_words_list):
    """
    get significant words from query
    :param query: the given query
    :param common_words_list: the common words list
    :return: list of significant words
    """
    sig_list = []
    query_split = query.strip().split(' ')
    for term in query_split:
        if term not in common_words_list:
            sig_list.append(term)
    return sig_list

def generate_snippet(html_filename, html_path, query, common_words_list):
    """
    generate snippet for the given document and the query
    :param html_filename: the given html document name
    :param html_path: file path to html document
    :param query: the given query
    :param common_words_list: common words list
    :return: the snippet string
    """
    snippet = ''
    sig_words_count = []

    # get significant words from query
    significant_words = get_significant_words(query, common_words_list)

    # get html file content
    html_string = get_html_content(html_path)

    # cut html file content into sentences
    raw_sentences = cut_content(html_string)
    sentences = []
    for sentence in raw_sentences:
        if sentence != '':
            sentences.append(sentence)

    # count the number of significant words contained in each sentence, and sort the count
    sig_words_count = count_sig_words(sentences, significant_words)
    sig_words_count = sorted(sig_words_count.items(), key=operator.itemgetter(1), reverse=True)
    # print sig_words_count

    # generate the snippet
    sentence1 = add_bold(sig_words_count[0][0], significant_words)
    if len(sig_words_count) >= 2:
        sentence2 = add_bold(sig_words_count[1][0], significant_words)
        snippet = snippet + html_filename + '\n' + sentence1 + '. ... ' + sentence2 + '.\n\n'
    else:
        snippet = snippet + html_filename + '\n' + sentence1 + '.\n\n'
    return snippet

def display(file, source_folder, target_folder, doc_folder, common_words_list):
    """
    display the given file in snippet format
    :param file: the given doc list file
    :param source_folder: source folder that contain original file
    :param target_folder: target folder to put display file to
    :param doc_folder: the folder that contains html documents
    :param common_words_list: list of common words
    :return: None
    """

    query = ''
    s = ''
    lines = open(source_folder + '/' + file).readlines()
    for line in lines:
        line = line.strip()
        if line != '':
            if line.startswith('Query:'):
                query = line.split(':')[1]
                s = s + 'Query:' + query.lower() + '\n'
            else:
                split = line.split(',')
                html_filename = split[2]
                html_path = doc_folder + '/' + html_filename + '.html'
                s = s + generate_snippet(html_filename, html_path, query, common_words_list)

    # write snippets to a new text file
    f = open(target_folder + '/' + file, 'w')
    f.write(s)
    f.close()


def main():
    """
    The main function that performs the task in Phase2, which is display query results
    :return: None
    """
    # variable definitions
    phase1_output = '../phase1/result/Task1/BM25'
    phase2_output = '../phase2/phase2_output'
    doc_folder = '../general/test-collection/cacm'
    common_words_file = '../general/test-collection/common_words'
    common_words_list = get_common_words(common_words_file)

    # read files to display into a list
    files_to_display = os.listdir(phase1_output)

    # display each file
    for file in files_to_display:
        display(file, phase1_output, phase2_output, doc_folder, common_words_list)


main()