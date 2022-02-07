# team6_spam

Objectif du projet:
Concevoir un classifieur de détection automatique de SPAM.

La collection SMS Spam est un ensemble de messages SMS marqués qui ont été collectés pour la recherche sur les SMS Spam. Elle contient un ensemble de messages SMS en anglais de 5 574 messages, étiquetés selon qu'ils sont ham (légitimes) ou spam.
Ce corpus a été collecté à partir de sources libres ou gratuites pour la recherche sur Internet :

https://archive.ics.uci.edu/ml/datasets/sms+spam+collection#

Attention : il s'agit d'un jeu de données qui ne doit pas être travaillé en NLP (on fera du NLP un peu plus tard).
En terme de préparation des données, il s'agit d'extraire une multitude d'infos à partir du message texte, afin d'obtenir un ensemble de descripteurs (features) comme :

    - nombre et proportion des caractères "!, ?, €, #" etc
    - longueur du message
    - nombre et proportion de majuscules
    - présence d'URL
    - et d'autres descripteurs

Critères de performance

    - compréhension du jeux de données
    - capacité à préparer les données
    - performance des modèles de prédiction
    - capacité à apporter une solution dans le temps imparti
    - rédaction du rapport technique
    - qualité du synthèse du travail

Livrables

    - créer un/des notebook reproductible, commenté, expliqué
    - créer un repo git et un espace sur github/gitlab pour le projet (code refactorisé)
    - présenter un planning de travail
    - présenter un document technique qui explique l'outil
    - présenter la procédure suivie pour préparer les données et le preprocessing
    - présenter la procédure suivie pour trouver un modèle adapté
    - créer un modèle d'IA entraîné et adapté au problème
    - faire une présentation qui explique votre démarche et les résultats obtenus

