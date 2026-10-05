import pandas as pd
import glob
import os

# Pega o arquivo mais recente
csv_files = glob.glob('data/*_posts_unique.csv')
latest_csv = max(csv_files, key=os.path.getctime)
df = pd.read_csv(latest_csv)

# Dicionário de métricas específicas
METRICAS = {
    'Velocity / Story Points': 'velocity|story point|sprint point',
    'KPIs': 'kpi',
    'OKRs': 'okr',
    'Linhas de Código (LOC)': 'lines of code|loc',
    'Throughput / DORA': 'throughput|dora',
    'Commits / PRs': 'number of commit|pull request count|pr count'
}

print("=== CONTABILIDADE DAS MÉTRICAS ===")
resultados = []

for nome_metrica, regex in METRICAS.items():
    # Filtra posts que mencionam a métrica no título ou texto
    mask = df['title'].astype(str).str.contains(regex, case=False, na=False) | \
           df['selftext'].astype(str).str.contains(regex, case=False, na=False)
    
    posts_metrica = df[mask].copy()
    contagem = len(posts_metrica)
    print(f"{nome_metrica}: {contagem} posts encontrados")
    
    if contagem > 0:
        # Pega o melhor exemplo baseado em engajamento (score + comments)
        posts_metrica['engajamento'] = posts_metrica['score'] + posts_metrica['num_comments']
        melhor = posts_metrica.nlargest(1, 'engajamento').iloc[0]
        
        resultados.append(f"\nMétrica: {nome_metrica} ({contagem} menções)\n"
                          f"Melhor Exemplo (r/{melhor['subreddit']} - Ano: {melhor['created_utc'][:4]}):\n"
                          f"Título: {melhor['title']}\n"
                          f"Trecho: {str(melhor['selftext'])[:250]}...\n"
                          f"Engajamento: Score {melhor['score']}, Comentários {melhor['num_comments']}")

print("\n\n=== MELHORES EXEMPLOS POR MÉTRICA ===")
for r in resultados:
    print(r)
