# Choix techniques et pistes d'amélioration

## Choix techniques

- **FastAPI** : Framework web moderne et rapide, idéal pour créer des APIs REST. Il permet de structurer le code en modules clairs (routes, services, schémas).
- **Routers** : Les endpoints sont gérés dans des fichiers dédiés, ce qui facilite la maintenance et la clarté du code.
- **Schemas (Pydantic)** : Les modèles de données sont validés automatiquement, ce qui évite les erreurs et documente l’API.
- **Service ReviewRecommender** : 
  - Utilise FAISS pour la recherche sémantique (trouver des critiques similaires en sens).
  - Utilise TF-IDF pour la recherche lexicale (recherche par mots).
  - Fusionne les scores et peut appliquer un reranking par sentiment (optionnel, via le payload).
- **Stockage des indices** : Les données nécessaires à la recherche (embeddings, TF-IDF, métadonnées) sont stockées sur le disque pour des recherches rapides.
- **Modèle de sentiment HuggingFace** : Permet d’analyser le sentiment des critiques en français ou anglais, activé seulement si demandé.
- **Configuration par payload** : Toutes les options (mode, reranking, etc.) sont passées dans la requête JSON, ce qui simplifie l’utilisation et le déploiement.

## Avantages

- **Simplicité** : Le système est facile à comprendre et à utiliser, la configuration est minimale.
- **Performance** : Les recherches sont rapides et montrent des résultats satisfaisants pour une première version avec FAISS et TF-IDF.
- **Adapté au français** : Le modèle de sentiment est multilingue, ce qui permet de traiter des critiques en français.

## Pistes d’amélioration (scalabilité, robustesse…)

- **Scalabilité** : 
  - Pour gérer plus d’utilisateurs ou de films, il serait utile de stocker les indices sur un stockage partagé (cloud, NAS) et de répliquer l’API sur plusieurs serveurs.
  - Utiliser un cache mémoire (Redis, Memcached) pour accélérer les accès aux indices les plus demandés.
  - Utiliser une meilleure base vectorielle (Qdrant par exemple avec ses indexs adaptés à un gros volume de données)
- **Extensibilité** : 
  - Ajouter d’autres modèles de recherche ou de sentiment selon les besoins (ex : modèles plus légers ou spécialisés).
  - Permettre l’ajout de nouveaux modes de fusion des scores.
- **Robustesse** : 
  - Ajouter des tests automatiques et une gestion plus fine des erreurs.
  - Surveiller les performances et prévoir des alertes en cas de lenteur ou d’erreur.
- **Déploiement** : 
  - Dockeriser le service pour faciliter le déploiement sur le cloud ou en entreprise.
  - Prévoir une documentation utilisateur et développeur plus détaillée.

## Fusion des scores et formule utilisée

Pour chaque critique candidate, le score final est calculé en combinant :
- Le score sémantique (similarité de l’embedding, FAISS)
- Le score lexical (similarité TF-IDF, si mode "hybrid")
- Un bonus lié au nombre d’upvotes
- (Optionnel) Un bonus si le sentiment de la critique est aligné avec celui de la requête

La formule principale est :

$$
\text{score final} = \text{sem} \times (1 + \lambda_{up} \times up) + w \times \text{lex}
$$

- $\text{sem}$ : score sémantique, normalisé entre 0 et 1
- $up$ : bonus upvotes, normalisé
- $\lambda_{up}$ : pondération des upvotes (ex : 0.1)
- $w$ : pondération du score lexical (ex : 0.3)
- $\text{lex}$ : score lexical, normalisé entre 0 et 1

Si le reranking par sentiment est activé et que le sentiment de la requête et de la critique sont alignés, un bonus de $+0.05$ est ajouté au score final.

Cette formule permet de prendre en compte à la fois la proximité sémantique, la pertinence lexicale, la popularité (upvotes) et, si demandé, l’alignement du sentiment.

---

Ce système est une première version pour ce service de recommandation rapide, mais il peut évoluer facilement selon les besoins (volume, langues, personnalisation).
