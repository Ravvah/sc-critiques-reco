#!/usr/bin/env python3
"""
Évaluation non supervisée du système de recommandation selon certaines métriques exploratoires.
Il s'agit d'une évaluation hypothétique. Elle n'est pas très fiable.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import math

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnsupervisedRecommendationEvaluator:
    
    def __init__(self, api_url: str = "http://localhost:8000/v1"):
        self.api_url = api_url
        self.data_dir = Path(__file__).parent.parent / "data"
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        datasets = {}
        film_mapping = {
            "fightclub_critiques.csv": "fight_club",
            "interstellar_critiques.csv": "interstellar"
        }
        
        for film_data, film_id in film_mapping.items():
            file_path = self.data_dir / film_data
            if file_path.exists():
                datasets[film_id] = pd.read_csv(file_path)
                logger.info(f"Chargé {len(datasets[film_id])} critiques pour {film_id}")
        return datasets
    
    def get_recommendations(self, film_id: str, text: str, k: int = 20, 
                          mode: str = "hybrid", use_sentiment: bool = False) -> List[Dict]:
        """Appelle l'API de recommandation."""
        try:
            payload = {
                "film_id": film_id,
                "text": text,
                "k": k,
                "mode": mode,
                "use_sentiment": use_sentiment
            }
            
            response = requests.post(f"{self.api_url}/recommendations", 
                                   json=payload, timeout=30)
            response.raise_for_status()
            
            return response.json().get('items', [])
            
        except Exception as e:
            logger.error(f"Erreur API: {e}")
            return []

    def calculate_intra_list_diversity(self, recommendations: List[Dict]) -> float:
        if len(recommendations) < 2:
            return 0.0
            
        texts = [r.get('text', '') for r in recommendations]
        if not any(texts):
            return 0.0
            
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            n = len(recommendations)
            total_dissimilarity = 0.0
            pairs = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    dissimilarity = 1 - similarity_matrix[i][j]
                    total_dissimilarity += dissimilarity
                    pairs += 1
                    
            return total_dissimilarity / pairs if pairs > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Erreur calcul diversité: {e}")
            return 0.0

    def calculate_novelty(self, recommendations: List[Dict], df: pd.DataFrame) -> float:
        """
        Calcule la nouveauté moyenne des recommandations.
        Items populaires = moins de nouveauté. Basé sur -log(popularité).
        """
        if not recommendations:
            return 0.0
            
        # Calculer la popularité de chaque critique (basée sur upvotes/hits)
        popularity_scores = {}
        total_upvotes = df['gen_review_like_count'].sum() if 'gen_review_like_count' in df.columns else 1
        
        for _, row in df.iterrows():
            review_id = str(row['id'])
            upvotes = row.get('gen_review_like_count', 0)
            # Popularité = proportion des upvotes totaux + lissage
            popularity = (upvotes + 1) / (total_upvotes + len(df))
            popularity_scores[review_id] = popularity
            
        # Calculer la nouveauté pour chaque recommandation
        novelty_scores = []
        for rec in recommendations:
            review_id = str(rec.get('review_id', ''))
            popularity = popularity_scores.get(review_id, 1 / len(df))  # Default si pas trouvé
            novelty = -math.log2(popularity)
            novelty_scores.append(novelty)
            
        return np.mean(novelty_scores) if novelty_scores else 0.0

    def calculate_coverage(self, all_recommendations: List[List[Dict]], df: pd.DataFrame) -> Dict:
        """
        Calcule la couverture du catalogue.
        - Item Coverage: % d'items du catalogue recommandés
        - User Coverage: % d'utilisateurs pour qui on peut recommander
        """
        recommended_ids = set()
        for recs in all_recommendations:
            for rec in recs:
                review_id = str(rec.get('review_id', ''))
                if review_id:
                    recommended_ids.add(review_id)
        
        total_items = set(df['id'].astype(str))
        
        item_coverage = len(recommended_ids.intersection(total_items)) / len(total_items)
        
        # Coverage des utilisateurs (tous peuvent avoir des recommandations)
        user_coverage = 1.0  # Dans notre cas, on peut recommander pour toute requête
        
        return {
            'item_coverage': item_coverage,
            'user_coverage': user_coverage,
            'items_recommended': len(recommended_ids),
            'total_items': len(total_items)
        }

    def calculate_coherence_score(self, query_text: str, recommendations: List[Dict]) -> float:
        """
        Calcule la cohérence entre la requête et les recommandations.
        Utilise la similarité TF-IDF moyenne entre requête et recommandations.
        """
        if not recommendations:
            return 0.0
            
        rec_texts = [r.get('text', '') for r in recommendations]
        if not any(rec_texts):
            return 0.0
            
        try:
            all_texts = [query_text] + rec_texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculer similarité entre requête (index 0) et chaque recommandation
            query_vector = tfidf_matrix[0]
            similarities = []
            
            for i in range(1, len(all_texts)):
                rec_vector = tfidf_matrix[i]
                similarity = cosine_similarity(query_vector, rec_vector)[0][0]
                similarities.append(similarity)
                
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Erreur calcul cohérence: {e}")
            return 0.0

    def calculate_ranking_quality(self, recommendations: List[Dict]) -> Dict:
        """
        Évalue la qualité du ranking basé sur les scores de confiance de l'API.
        """
        if not recommendations:
            return {'score_variance': 0.0, 'score_range': 0.0, 'monotonic_decrease': False}
            
        scores = []
        for rec in recommendations:
            score = rec.get('final', rec.get('cos', 0.0))
            scores.append(float(score) if score is not None else 0.0)
        
        score_variance = np.var(scores) if len(scores) > 1 else 0.0
        
        score_range = max(scores) - min(scores) if scores else 0.0
        
        monotonic_decrease = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        return {
            'score_variance': score_variance,
            'score_range': score_range,
            'monotonic_decrease': monotonic_decrease,
            'mean_score': np.mean(scores) if scores else 0.0
        }

    def evaluate_film_unsupervised(self, film_id: str, df: pd.DataFrame, sample_size: int = 20) -> Dict:
        logger.info(f"Évaluation non supervisée pour {film_id} avec {sample_size} échantillons")
        
        sample_reviews = df.sample(min(sample_size, len(df))).to_dict('records')
        
        all_recommendations = []
        metrics_per_query = []
        successful_queries_set = set()
        
        for i, review in enumerate(sample_reviews):
            text = review['review_content'][:1000]
            query_successful = False
            
            for k in [10, 20]:
                recommendations = self.get_recommendations(film_id, text, k=k)
                
                if not recommendations:
                    continue
                    
                all_recommendations.append(recommendations)
                query_successful = True
                
                query_metrics = {
                    'k': k,
                    'query_id': review['id'],
                    'diversity': self.calculate_intra_list_diversity(recommendations),
                    'novelty': self.calculate_novelty(recommendations, df),
                    'coherence': self.calculate_coherence_score(text, recommendations),
                    'ranking_quality': self.calculate_ranking_quality(recommendations)
                }
                
                metrics_per_query.append(query_metrics)
            
            if query_successful:
                successful_queries_set.add(review['id'])
                
            if (i + 1) % 5 == 0:
                logger.info(f"Traité {i+1}/{len(sample_reviews)} requêtes")
        
        coverage_metrics = self.calculate_coverage(all_recommendations, df)
        
        results = {}
        for k in [10, 20]:
            k_metrics = [m for m in metrics_per_query if m['k'] == k]
            if k_metrics:
                results[f'k_{k}'] = {
                    'diversity_avg': np.mean([m['diversity'] for m in k_metrics]),
                    'novelty_avg': np.mean([m['novelty'] for m in k_metrics]),
                    'coherence_avg': np.mean([m['coherence'] for m in k_metrics]),
                    'score_variance_avg': np.mean([m['ranking_quality']['score_variance'] for m in k_metrics]),
                    'score_range_avg': np.mean([m['ranking_quality']['score_range'] for m in k_metrics]),
                    'monotonic_ratio': np.mean([m['ranking_quality']['monotonic_decrease'] for m in k_metrics]),
                    'successful_queries': len(k_metrics)
                }
        
        results['coverage'] = coverage_metrics
        results['total_queries'] = len(sample_reviews)
        results['successful_queries'] = len(successful_queries_set)
        
        return results

    def run_unsupervised_evaluation(self) -> Dict:
        """Lance l'évaluation non supervisée complète."""
        logger.info("Démarrage de l'évaluation non supervisée du système de recommandation")
        
        try:
            response = requests.get(f"{self.api_url}/readyz", timeout=10)
            if not response.json().get('ready', False):
                logger.error("L'API n'est pas prête (index non disponible)")
                return {}
        except Exception as e:
            logger.error(f"Impossible de contacter l'API: {e}")
            return {}
        
        datasets = self.load_datasets()
        
        if not datasets:
            logger.error("Aucun dataset trouvé")
            return {}
        
        results = {}
        for film_id, df in datasets.items():
            logger.info(f"Évaluation non supervisée de {film_id}")
            film_results = self.evaluate_film_unsupervised(film_id, df, sample_size=15)
            results[film_id] = film_results
            
            logger.info(f"Résultats pour {film_id.upper()}:")
            for k_name, metrics in film_results.items():
                if k_name.startswith('k_'):
                    k_val = k_name.split('_')[1]
                    logger.info(f"  Top-{k_val}:")
                    logger.info(f"    Diversité:     {metrics['diversity_avg']:.3f}")
                    logger.info(f"    Nouveauté:     {metrics['novelty_avg']:.3f}")
                    logger.info(f"    Cohérence:     {metrics['coherence_avg']:.3f}")
                    logger.info(f"    Var. scores:   {metrics['score_variance_avg']:.3f}")
                    logger.info(f"    Range scores:  {metrics['score_range_avg']:.3f}")
                    logger.info(f"    Ordre monotone: {metrics['monotonic_ratio']:.1%}")
            
            if 'coverage' in film_results:
                cov = film_results['coverage']
                logger.info(f"  Couverture:")
                logger.info(f"    Items couverts: {cov['items_recommended']}/{cov['total_items']} ({cov['item_coverage']:.1%})")
            
            logger.info(f"  Requêtes réussies: {film_results.get('successful_queries', 0)}/{film_results.get('total_queries', 0)}")
        
        return results
        
    def create_ground_truth(self, df: pd.DataFrame, sample_size: int = 50) -> List[Dict]:
        """
        Crée la vérité terrain basée sur les ratings et la similarité des critiques.
        Pour chaque critique, on considère comme pertinentes les critiques avec:
        - Rating similaire (±1 point)
        - Contenu non vide
        """
        sample_reviews = df.sample(min(sample_size, len(df))).to_dict('records')
        ground_truth = []
        
        for review in sample_reviews:
            rating = review['rating']
            review_id = review['id']
            
            # Critiques pertinentes: rating similaire (±1) et contenu valide
            relevant_mask = (
                (df['rating'] >= rating - 1) & 
                (df['rating'] <= rating + 1) &
                (df['review_content'].str.len() > 100) &  
                (df['id'] != review_id)  
            )
            
            relevant_ids = df[relevant_mask]['id'].tolist()
            
            ground_truth.append({
                'query_review': review,
                'relevant_ids': set(relevant_ids),
                'total_relevant': len(relevant_ids)
            })
            
        return ground_truth
    
    def get_recommendations(self, film_id: str, text: str, k: int = 20, 
                          mode: str = "hybrid", use_sentiment: bool = False) -> List[Dict]:
        """Appelle l'API de recommandation."""
        try:
            payload = {
                "film_id": film_id,
                "text": text,
                "k": k,
                "mode": mode,
                "use_sentiment": use_sentiment
            }
            
            response = requests.post(f"{self.api_url}/recommendations", 
                                   json=payload, timeout=30)
            response.raise_for_status()
            
            return response.json().get('items', [])
            
        except Exception as e:
            logger.error(f"Erreur API: {e}")
            return []
    
    def calculate_metrics(self, relevant_ids: set, recommended_ids: list, k: int) -> Dict:
        """Calcule précision, rappel et F1 score."""
        # Prendre seulement les k premières recommandations
        top_k_recommended = set(recommended_ids[:k])
        
        # Intersection entre recommandé et pertinent
        true_positives = len(relevant_ids.intersection(top_k_recommended))
        
        # Métriques
        precision = true_positives / len(top_k_recommended) if top_k_recommended else 0
        recall = true_positives / len(relevant_ids) if relevant_ids else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'total_recommended': len(top_k_recommended),
            'total_relevant': len(relevant_ids)
        }
    
    def evaluate_film(self, film_id: str, df: pd.DataFrame, sample_size: int = 20) -> Dict:
        """Évalue les recommandations pour un film donné."""
        logger.info(f"Évaluation pour {film_id} avec {sample_size} échantillons")
        
        # Créer la vérité terrain
        ground_truth = self.create_ground_truth(df, sample_size)
        
        all_metrics = []
        successful_queries = 0
        
        for i, gt in enumerate(ground_truth):
            review = gt['query_review']
            relevant_ids = gt['relevant_ids']
            
            text = review['review_content'][:1000]  # Limiter la taille
            
            recommendations = self.get_recommendations(film_id, text, k=20)
            
            if not recommendations:
                logger.warning(f"Aucune recommandation pour la requête {i+1}")
                continue
            
            if i == 0:
                logger.info(f"Structure de la première réponse: {recommendations[0] if recommendations else 'vide'}")
                
                
            recommended_ids = []
            for r in recommendations:
                review_id = r.get('review_id') or r.get('id')
                if review_id:
                    try:
                        recommended_ids.append(int(review_id))
                    except (ValueError, TypeError):
                        continue
            
            if not recommended_ids:
                logger.warning(f"IDs de recommandations manquants pour la requête {i+1}")
                logger.debug(f"Structure de la réponse: {recommendations[:2] if recommendations else 'vide'}")
                continue
            
            # Calculer les métriques pour différentes valeurs de k
            for k in [5, 10, 20]:
                metrics = self.calculate_metrics(relevant_ids, recommended_ids, k)
                metrics['k'] = k
                metrics['query_id'] = review['id']
                all_metrics.append(metrics)
            
            successful_queries += 1
            
            if (i + 1) % 5 == 0:
                logger.info(f"Traité {i+1}/{len(ground_truth)} requêtes")
        
        # Calculer les moyennes par k
        results = {}
        for k in [5, 10, 20]:
            k_metrics = [m for m in all_metrics if m['k'] == k]
            if k_metrics:
                results[f'k_{k}'] = {
                    'precision_avg': sum(m['precision'] for m in k_metrics) / len(k_metrics),
                    'recall_avg': sum(m['recall'] for m in k_metrics) / len(k_metrics),
                    'f1_score_avg': sum(m['f1_score'] for m in k_metrics) / len(k_metrics),
                    'successful_queries': len(k_metrics)
                }
        
        results['total_queries'] = len(ground_truth)
        results['successful_queries'] = successful_queries
        
        return results
    
    def run_evaluation(self) -> Dict:
        """Lance l'évaluation complète."""
        logger.info("Démarrage de l'évaluation du système de recommandation")
        
        try:
            response = requests.get(f"{self.api_url}/readyz", timeout=10)
            if not response.json().get('ready', False):
                logger.error("L'API n'est pas prête (index non disponible)")
                return {}
        except Exception as e:
            logger.error(f"Impossible de contacter l'API: {e}")
            return {}
        
        datasets = self.load_datasets()
        
        if not datasets:
            logger.error("Aucun dataset trouvé")
            return {}

def main():
    evaluator = UnsupervisedRecommendationEvaluator()
    results = evaluator.run_unsupervised_evaluation()
    
    if results:
        results_file = Path(__file__).parent / "unsupervised_evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Résultats sauvegardés dans {results_file}")
        
        logger.info("RÉSUMÉ GLOBAL - ÉVALUATION NON SUPERVISÉE")
        
        for k in [10, 20]:
            k_key = f'k_{k}'
            diversities = []
            novelties = []
            coherences = []
            
            for film_results in results.values():
                if k_key in film_results:
                    diversities.append(film_results[k_key]['diversity_avg'])
                    novelties.append(film_results[k_key]['novelty_avg'])
                    coherences.append(film_results[k_key]['coherence_avg'])
            
            if diversities:
                logger.info(f"Top-{k} (moyenne sur tous les films):")
                logger.info(f"  Diversité:  {np.mean(diversities):.3f}")
                logger.info(f"  Nouveauté:  {np.mean(novelties):.3f}")
                logger.info(f"  Cohérence:  {np.mean(coherences):.3f}")
        
        total_items_covered = 0
        total_items = 0
        for film_results in results.values():
            if 'coverage' in film_results:
                cov = film_results['coverage']
                total_items_covered += cov['items_recommended']
                total_items += cov['total_items']
        
        if total_items > 0:
            global_coverage = total_items_covered / total_items
            logger.info(f"Couverture globale: {global_coverage:.1%}")
        
    else:
        logger.error("Évaluation échouée")


if __name__ == "__main__":
    main()