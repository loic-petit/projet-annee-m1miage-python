#import warnings
#warnings.filterwarnings(action = 'ignore')

import csv
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation; animation.writers.list()
import os as os
from scipy.spatial import distance

wordnet_lemmatizer = WordNetLemmatizer()

file = open("Final", encoding="utf-8")

print("recup lignes")

my_lines_list = file.readlines() #récupère toutes les lignes du document nettoyé

texte = " ".join(my_lines_list)

all_sentences = nltk.sent_tokenize(texte) # on récupère toutes les phrases
all_words = [nltk.word_tokenize(sent) for sent in all_sentences] #ou récupère tous les mots

model = Word2Vec(all_words, min_count=10) # on récupère les mots qui apparaissent au moins 10 fois

vocabulary = model.wv.vocab

print("vocab")

#récup les 100 mots les plus fréquents
w2c = dict()
for item in vocabulary:
   w2c[item]=vocabulary[item].count
w2cSorted=dict(sorted(w2c.items(), key=lambda x: x[1],reverse=True))
w2cSortedList = list(w2cSorted.keys()) #on trie le tableau en fonction du nombre d'occurence des mots
mostCommonWord = w2cSortedList[:100] #on prend les 100 mots les plus fréquents

print("reduction dimension")

ar = []
for word in mostCommonWord:
   ar.append(model.wv[word])

X = np.array(ar)
# X_embedded = TSNE(n_components=2) # 2D
X_embedded3 = TSNE(n_components=3) # 3D
# tsne_obj = X_embedded.fit_transform(X)
tsne_obj3 = X_embedded3.fit_transform(X)

# def coordonées des points 2D
# x = tsne_obj[:,0]
# y = tsne_obj[:,1]
# fin def coordonées des points 2D

# def coordonées des points 3D
x3 = tsne_obj3[:,0]
y3 = tsne_obj3[:,1]
z3 = tsne_obj3[:,2]
# fin def coordonées des points 3D

print("graph")

# fig = plt.figure(figsize=(8, 8)) # pour image
fig3 = plt.figure(figsize=(15, 15)) # pour animation 3D

# parametre graph 2D
# plt.plot(x,y,"ob") # ob = type de points "o" ronds, "b" bleus
# plt.ylabel('y')
# plt.xlabel('x')
# fin parametre graph 2D

# parametre graph 3D
ax = fig3.gca(projection='3d')
ax.scatter(x3,y3,z3, zdir='z',s=40,depthshade=True)
ax.set_xlim3d([-150.0, 150.0]); ax.set_xlabel('X')
ax.set_ylim3d([-150.0, 150.0]); ax.set_ylabel('Y')
ax.set_zlim3d([-150.0, 150.0]); ax.set_zlabel('Z')
# fin parametre graph 3D

# annotation 2D
# for i, label in enumerate(mostCommonWord):
#     plt.annotate(label, (x[i], y[i]))
# fin annotation 2D

print("annotate")
# annotation 3D
for i in range(len(mostCommonWord)):
  x = x3[i]
  y = y3[i]
  z = z3[i]
  label = mostCommonWord[i]
  ax.scatter(x, y, z, color='b')
  ax.text(x, y, z, '%s' % (label), size=10, zorder=1, color='k')
# fin annotation 3D

print("Faire une animation")
# animation gif
def rotate(angle):
    ax.view_init(azim=angle)

rot_animation = animation.FuncAnimation(fig3, rotate, frames=np.arange(0, 362, 2), interval=100)
os.chdir("..\Gif")
rot_animation.save('rotation2.gif', dpi=80, writer='pillow')
# fin animation gif

# plt.show() # Obligatoire pour voir le graphique

# lancer uniquement le graphe 2D ou le graphe 3D, pas les deux en même temps, cela créer un problème au niveau des axes

# création du fichier csv (idWord)
entetesIdWord = [u'ID', u'Label']

valeursIdWord = []
for i in range(len(mostCommonWord)):
    valeursIdWord.append([mostCommonWord[i], mostCommonWord[i]])

os.chdir('../CSV')
f = open('idWord.csv', 'w')
ligneEntete = ";".join(entetesIdWord) + "\n"
f.write(ligneEntete)
for valeur in valeursIdWord:
     ligne = ";".join(valeur) + "\n"
     f.write(ligne)

f.close()
# fin création du fichier csv (idWord)

# création du fichier csv (weightWord)
entetesWeightWord = [u'Source', u'Target', 'Weight']

valeurWeightWord = []

for i in range(len(mostCommonWord)):
    mostCommonWordReduit = mostCommonWord[i:100] # sous tableau afin de faire toute les combinaisons de mots
    for j in range(len(mostCommonWordReduit)):
        j += 1                                  # pour ne pas avoir mot1 et mot2 égaux
        if (j != len(mostCommonWordReduit)):    # pour que le dernier array ne soit pas 2 fois le même mot
            word1 = mostCommonWord[i]
            word2 = mostCommonWordReduit[j]
            strWeight = str(model.wv.similarity(word1, word2))
            valeurWeightWord.append([word1, word2, strWeight])

os.chdir('../CSV')
f = open('weightWord.csv', 'w')
ligneEntete = ";".join(entetesWeightWord) + "\n"
f.write(ligneEntete)
for valeur in valeurWeightWord:
     ligne = ";".join(valeur) + "\n"
     f.write(ligne)

f.close()
# fin création du fichier csv (weightWord)