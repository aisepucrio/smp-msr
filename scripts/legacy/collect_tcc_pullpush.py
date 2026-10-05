import requests
import pandas as pd
import time
import os
from datetime import datetime

SUBREDDITS = ["cscareerquestions", "ExperiencedDevs", "womenintech", "softwareengineering", "devops"]
QUERIES = {
    "mental_health": "burnout|exhausted|overwhelmed|anxiety|stress|depression|therapy|toxic|hostile|RTO",
    "metrics_productivity": "metric|measurement|OKR|KPI|velocity|throughput|productivity|benchmark"
}

AFTER = 1609459200  # 2021-01-01
BEFORE = 1735689599 # 2024-12-31

def fetch_data():
    all_posts = []
    
    for sub in SUBREDDITS:
        for q_name, q_terms_str in QUERIES.items():
            print(f"Buscando {q_name} em r/{sub}...")
            terms = q_terms_str.split('|')
            for term in terms:
                term_posts = 0
                current_before = BEFORE
                
                while True:
                    url = f"https://api.pullpush.io/reddit/search/submission/?subreddit={sub}&q={term}&after={AFTER}&before={current_before}&size=100"
                    
                    try:
                        response = requests.get(url)
                        response.raise_for_status()
                        data = response.json().get('data', [])
                        
                        if not data:
                            break  # Sem mais posts para esse termo
                            
                        for post in data:
                            created_utc = int(post.get('created_utc', 0))
                            all_posts.append({
                                "id": post.get('id', ''),
                                "subreddit": sub,
                                "query_group": q_name,
                                "title": post.get('title', ''),
                                "author": post.get('author', ''),
                                "score": post.get('score', 0),
                                "num_comments": post.get('num_comments', 0),
                                "created_utc": datetime.fromtimestamp(created_utc).strftime('%Y-%m-%d %H:%M'),
                                "selftext": str(post.get('selftext', ''))[:500],
                                "permalink": f"https://reddit.com{post.get('permalink', '')}"
                            })
                        
                        term_posts += len(data)
                        
                        # O PullPush retorna os posts ordenados do mais recente para o mais antigo.
                        # Pegamos o 'created_utc' do último post (o mais antigo do lote) e subtraímos 1
                        last_post_utc = int(data[-1].get('created_utc', 0))
                        current_before = last_post_utc - 1
                        
                        time.sleep(1.5)  # Respeita o rate limit
                        
                    except Exception as e:
                        print(f"Erro na API para r/{sub} termo {term}: {e}")
                        break
                
                print(f" -> {term_posts} posts totais para '{term}'.")

    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(all_posts)
    # Remove duplicatas caso algum post cruze os termos
    df = df.drop_duplicates(subset=['id'])
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"data/{stamp}_posts_unique.csv"
    df.to_csv(path, index=False)
    print(f"\n✅ SUCESSO! {len(df)} posts REAIS extraídos e salvos em {path}")

if __name__ == "__main__":
    fetch_data()
