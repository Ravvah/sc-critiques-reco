# Système de recommandation de critiques de films

API FastAPI pour la recommandation de critiques similaires à une critique de film.

## Objectif
À partir d'une critique (texte + film_id), l'API retourne les critiques du même film les plus proches sémantiquement.

## Fonctionnement
- **Indexation batch** :
	- Nettoyage des données (HTML, espaces)
	- Embedding (MiniLM multilingue)
	- Construction d'index FAISS HNSW
	- Construction TF-IDF pour le mode hybride
- **Service API** :
	- `POST /v1/recommendations` : retourne les critiques similaires
	- `GET /v1/readyz` : index disponible


## Exemple d'appel
```bash
curl -X POST localhost:8000/v1/recommendations \
	-H 'Content-Type: application/json' \
	-d '{
		"film_id": "fight_club",
		"text": "je n'ai pas aimé Fight Club",
		"k": 5,
		"mode": "dense",
		"ef_search": 48,
		"lambda_up": 0.1
	}'
```

## Payload d'entrée (JSON)

| Champ         | Type      | Description                                                        |
|--------------|-----------|--------------------------------------------------------------------|
| film_id       | str       | Identifiant du film (ex: "fight_club")                            |
| text          | str       | Texte de la critique à comparer                                    |
| review_id     | str/int   | (optionnel) ID d'une critique existante à utiliser comme requête   |
| k             | int       | Nombre de recommandations à retourner (1-50)                       |
| mode          | str       | "dense" (sémantique uniquement) ou "hybrid" (sémantique + lexical)|
| ef_search     | int       | Paramètre technique HNSW (48 par défaut)                           |
| lambda_up     | float     | Pondération des upvotes (0.0 à 1.0, 0.1 par défaut)                |
| use_sentiment | bool      | (optionnel) Active le reranking par sentiment (false par défaut)   |


## Structure du projet
```
app/
	api.py            # FastAPI app
	routers/          # Endpoints
	schemas/          # Modèles Pydantic
	services/         # Logique métier (recommendation)
	settings.py       # Configuration
	utils.py          # Nettoyage texte
```

## Fonctionnement du reranking par sentiment

Si le champ `use_sentiment` est à `true` dans le payload, l'API utilise un modèle HuggingFace multilingue pour analyser le sentiment de la requête et des critiques candidates. Les critiques dont le sentiment est aligné avec la requête reçoivent un léger bonus dans le score final.

Ce reranking est optionnel et désactivé par défaut. Il est utile pour mieux faire ressortir les critiques négatives ou positives selon le contexte, mais peut être ignoré si les avis sont neutres ou ambigus.

Exemple de payload avec reranking sentiment :
```json
{
	"film_id": "fight_club",
	"text": "je n'ai pas aimé Fight Club, il y'a beaucoup de combat à mains nues",
	"k": 5,
	"mode": "dense",
	"use_sentiment": true
}
```

## Logs importants

- `[REQUEST]` : chaque requête reçue
- `[SUCCESS]` : recommandations retournées
- `[ERROR]` : erreur lors du traitement

## Installation
```bash
uv venv
uv pip install -e .[dev]
DATA_DIR=./data OUT_DIR=./indices uv run -m app.services.indexer
uv run python -m uvicorn app.api:app --reload --port 8000
```

## Auteur
Rabah ACHOUR
