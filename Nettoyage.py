import re
import string
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) #mot courant en anglais

all_words = []
newStopWord = ["one", "two", "new", ".", "may", "also", "al", "use", "first", "e.g", "eg", "thus", "within", "however", "would", "likely", "three", "less", "using", "well"]
for word in newStopWord:
    stop_words.add(word) # on ajoute à la liste stop_words la liste de nouveaux mots

#fonction qui nettoie la phrase
def stemsentence(sentence, all_sentences):
    sentence = sentence.lower()             # on met la phrase en miniscule
    sentence = sentence.replace("- ", "")   # enlève les -
    sentence = sentence.replace("---", "")  # enlève les ---
    sentence = sentence.replace("...", "")  # enlève les ...
    sentence = sentence.replace(".", "")    # enlève les .
    sentence = re.sub("\d+", "", sentence)  #enlève les chiffres de la phrase
    sentence_words = nltk.word_tokenize(sentence)
    words_line = []

    for word in sentence_words:
        if word not in string.punctuation and word not in stop_words and len(word) > 2: #si le mot est ni de la ponctuation ni un stop_words et sa taille est supérieur à 2 alors
            words_line.append(word) # on ajoute le mot à la liste de mot de la phrase

    all_sentences.append(words_line) #on ajoute la phrase aux autres phrases
    return all_sentences
#fin fonction stemsentence

#ouvre le fichier qui contient les articles
file = open("all", encoding="utf-8")
# file = open("JeuxDeDonnees", encoding="utf-8")

my_lines_list = file.readlines() #récupère toutes les lignes

all_sentences = []

for line in my_lines_list:
    all_sentences = stemsentence(line, all_sentences) #on récupère les phrases nettoyées

#gestion des césures
for i in range(len(all_sentences) - 1):
    if len(all_sentences[i]) > 0 and len(all_sentences[i+1]) > 0 and "-" in all_sentences[i][-1]: # on vérifie qu'on est pas aux bornes de la liste et qu'il y ait un tiret au dernier mot
        all_sentences[i][-1] = all_sentences[i][-1].replace("-", "")    # suppression du tiret
        temp = all_sentences[i][-1] + all_sentences[i+1][0]             # on reconstitue le mot
        all_sentences[i + 1].remove(all_sentences[i+1][0])              # on supprime la fin du mot césuré au début de la phrase
        all_sentences[i + 1].insert(0, temp)                            # on insert le mot reconstitué
        all_sentences[i].remove(all_sentences[i][-1])                   # on supprime le début du mot césuré à la fin de la phrase
#fin gestion des césures

tab_all_words = []

for sentence in all_sentences:
    tab_all_words.append(" ".join(sentence)) #regroupement de toutes les phrases

all_words = " ".join(tab_all_words) #transforme le tableau de mots en une string

# stem_file = open("FinalTest", mode="a+", encoding="utf-8") #on créé un fichier s'il n'existe pas ou ajoute au fichier s'il existe déjà
stem_file = open("Final", mode="a+", encoding="utf-8") #on créé un fichier s'il n'existe pas ou ajoute au fichier s'il existe déjà
stem_file.write(all_words) #on y met le texte nettoyé