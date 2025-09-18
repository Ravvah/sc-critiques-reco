# Évaluation Non Supervisée du Système de Recommandation

## Vue d'ensemble

Ce système d'évaluation utilise des métriques standard de la littérature pour évaluer la qualité des recommandations de critiques de films sans nécessiter d'annotations manuelles.

## Métriques Implémentées

### 1. Diversité Intra-Liste (ILD)
- **Définition** : Mesure la variété des recommandations dans une liste
- **Calcul** : Moyenne des dissimilarités TF-IDF entre toutes les paires de recommandations
- **Formule** : `ILD = 1/n(n-1) * Σᵢⱼ (1 - cos_similarity(i,j))`
- **Interprétation** : Plus élevé = recommandations plus diverses
- **Source** : Ziegler et al. (2005) "Improving Recommendation Lists Through Topic Diversification"

### 2. Nouveauté
- **Définition** : Tendance du système à recommander des items moins populaires
- **Calcul** : `-log₂(popularité)` où popularité = proportion des upvotes
- **Formule** : `Nouveauté = -log₂((upvotes + 1) / (total_upvotes + N))`
- **Interprétation** : Plus élevé = recommande des critiques moins connues
- **Source** : Vargas & Castells (2011) "Rank and Relevance in Novelty and Diversity Metrics for Recommender Systems"

### 3. Cohérence
- **Définition** : Pertinence des recommandations par rapport à la requête
- **Calcul** : Similarité TF-IDF moyenne entre requête et recommandations
- **Formule** : `Cohérence = 1/k * Σᵢ cos_similarity(requête, rec_i)`
- **Interprétation** : Plus élevé = recommandations plus pertinentes
- **Source** : Baeza-Yates & Ribeiro-Neto (2011) "Modern Information Retrieval"

### 4. Qualité du Ranking
- **Variance des scores** : Discrimination entre les recommandations
- **Range des scores** : Étendue des scores de confiance
- **Ordre monotone** : % de listes avec scores décroissants
- **Source** : Manning et al. (2008) "Introduction to Information Retrieval"

### 5. Couverture du Catalogue
- **Item Coverage** : Pourcentage d'items du catalogue recommandés
- **Formule** : `Coverage = |Items recommandés| / |Catalogue total|`
- **Interprétation** : Plus élevé = meilleure exploitation du catalogue
- **Source** : Herlocker et al. (2004) "Evaluating Collaborative Filtering Recommender Systems"

## Processus d'Évaluation

### Étape 1 : Échantillonnage
- Sélection aléatoire de 15 critiques par film comme requêtes test
- Utilisation du contenu de chaque critique comme query textuelle

### Étape 2 : Génération des Recommandations
- Appel API pour chaque requête avec k=10 et k=20
- Test en mode "hybrid" par défaut
- Récupération des recommandations avec scores de confiance

### Étape 3 : Calcul des Métriques
- **Diversité** : Analyse TF-IDF des contenus recommandés
- **Nouveauté** : Calcul basé sur la popularité (upvotes)
- **Cohérence** : Similarité requête-recommandations
- **Ranking** : Analyse des scores de confiance API

### Étape 4 : Agrégation
- Moyennage des métriques sur toutes les requêtes
- Calcul séparé pour Top-10 et Top-20
- Métriques globales de couverture

## Interprétation des Résultats

### Exemple de Résultats et Analyse
```
FIGHT_CLUB:
  Top-10:
    Diversité:     0.692  # Modérément diverse
    Nouveauté:     10.344 # Items relativement peu populaires
    Cohérence:     0.279  # Pertinence modérée  
    Ordre monotone: 100%  # Ranking cohérent
  
  Couverture:
    Items couverts: 206/1000 (20.6%)  # Exploration limitée du catalogue
```

**Analyse qualitative** :
- **Système performant** en nouveauté (recommande des critiques moins connues)
- **Trade-off visible** entre cohérence et diversité (typique)  
- **Ranking de qualité** (scores décroissants)
- **Couverture modérée** (amélioration possible)

### Interprétation des Métriques

**Important** : Les métriques doivent être interprétées de manière relative et comparative plutôt qu'avec des seuils absolus. Les valeurs "bonnes" dépendent du domaine, du dataset, et du contexte d'application.

#### Principes d'Interprétation
- **Diversité** : Plus élevé = recommandations plus variées. Comparer entre différents modes.
- **Nouveauté** : Plus élevé = recommande des items moins populaires. Dépend de la distribution de popularité du dataset.
- **Cohérence** : Plus élevé = recommandations plus pertinentes. Attention au sur-ajustement lexical.
- **Couverture** : Plus élevé = meilleure exploration du catalogue. Trade-off avec la précision.

#### Utilisation Comparative
Au lieu de seuils absolus, utilisez ces métriques pour :
1. **Comparer différentes configurations** (dense vs hybrid)
2. **Évaluer l'impact de paramètres** (ef_search, lambda_up)
3. **Analyser les trade-offs** (pertinence vs diversité)
4. **Suivre l'évolution** des performances dans le temps


## Avantages de cette Approche

1. **Automatique** : Pas d'annotation manuelle requise
2. **Standard** : Métriques reconnues dans la littérature
3. **Multi-dimensionnelle** : Évalue diversité, pertinence, nouveauté
4. **Comparative** : Permet de comparer différentes configurations
5. **Scalable** : Fonctionne sur de gros datasets

## Limitations

1. **Pas de vérité terrain humaine** : Basé sur des heuristiques
2. **Biais TF-IDF** : Favorise la similarité lexicale
3. **Définition popularité** : Basée uniquement sur upvotes
4. **Échantillonnage** : Résultats peuvent varier selon l'échantillon
5. **Absence de seuils universels** : Pas de benchmarks absolus établis dans la littérature - l'interprétation doit être relative et contextuelle

## Utilisation

```bash
# Lancer l'évaluation complète
python evaluation/evaluation.py

# Lancer via le script simplifié  
python evaluation/run_evaluation.py
```

## Fichiers de Sortie

- `unsupervised_evaluation_results.json` : Résultats détaillés
- Logs : Progression et métriques intermédiaires
- Résumé console : Moyennes par film et globales

## Extensions Possibles

1. **Métriques additionnelles** : Coverage utilisateur, serendipité
2. **Comparaison modes** : Dense vs Hybrid vs Sentiment

## Ressources Additionnelles

- **Survey on Evaluation Metrics** : Shani & Gunawardana (2011) "Evaluating Recommendation Systems"
- **MovieLens Dataset Papers** : https://grouplens.org/datasets/movielens/