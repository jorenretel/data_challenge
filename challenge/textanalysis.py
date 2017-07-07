
import os
import textacy
import numpy

def analyse_scripts(reference, data_dir):

    scripts = load_scripts(data_dir)

    # Transform to textacy Doc objects
    docs = {}
    for file_name, content in scripts.items():
        docs[file_name] = textacy.Doc(content, lang='en')

    # Create bag of terms for the reference file
    bot = docs[reference].to_bag_of_terms(ngrams={1, 2, 3}, as_strings=True)
    sorted_bot = sorted(bot.items(), key=lambda x: x[1], reverse=True)

    # Select all terms that appear more than twice.
    sorted_bot = [(term, freq) for term, freq in sorted_bot if freq > 2]
    print('Number N of selected terms: ', len(sorted_bot))


    terms = [term_freq[0] for term_freq in sorted_bot]
    print('Selected terms: ', terms)
    # Create a vector of the frequency of terms
    # normalized by the total amount of terms
    # found in the document
    freq_vec_script = numpy.array([term_freq[1] for term_freq in sorted_bot]) / len(bot)

    for file_name, doc in docs.items():

        print()
        print(file_name, ' vs. ', reference)
        print('First sentence: ', list(doc.sents)[0])
        bot = doc.to_bag_of_terms(ngrams={1, 2, 3}, as_strings=True)
        freq_vec = numpy.array([bot.get(term, 0) for term in terms])/len(bot)

        # Calculating the squared distance of the term-vectors
        squared_distance = sum((freq_vec_script- freq_vec)**2)
        # Multiplying by 1000 to make the numbers more readable
        print('Squared distance (*1000): ', squared_distance * 1000)


def load_scripts(data_dir):
    '''Loads all text files from the
       data directory.
    '''

    scripts = {}

    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            path_to_script = os.path.join(data_dir, file)

            with open(path_to_script, 'r') as script_file:
                scripts[file] = script_file.read()

    return scripts


if __name__ == '__main__':

    data_dir = os.path.join('..', 'data')
    analyse_scripts('script.txt', data_dir)