import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():
    # 1. Carregar os dados reais mais recentes
    csv_files = glob.glob('data/*_posts_unique.csv')
    latest_csv = max(csv_files, key=os.path.getctime)
    df = pd.read_csv(latest_csv)
    
    analyzer = SentimentIntensityAnalyzer()
    
    METRICAS = {
        'LOC': 'lines of code|loc',
        'Velocity / Story Points': 'velocity|story point|sprint point',
        'Throughput / DORA': 'throughput|dora',
        'KPIs': 'kpi',
        'OKRs': 'okr',
        'Commits / PRs': 'number of commit|pull request count|pr count'
    }
    
    resultados_formais = []
    
    for nome, regex in METRICAS.items():
        mask = df['title'].astype(str).str.contains(regex, case=False, na=False) | \
               df['selftext'].astype(str).str.contains(regex, case=False, na=False)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            continue
            
        # 2. Calcular scores de sentimento
        # Combina titulo e texto para analise
        subset['text_to_analyze'] = subset['title'].fillna('') + ' ' + subset['selftext'].fillna('')
        subset['compound'] = subset['text_to_analyze'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        
        # Classificacao rigorosa: >0.05 é positivo, <-0.05 é negativo, resto é neutro
        subset['sentiment_label'] = pd.cut(subset['compound'], 
                                           bins=[-1, -0.05, 0.05, 1], 
                                           labels=['Negativo', 'Neutro', 'Positivo'])
        
        mean_sentiment = subset['compound'].mean()
        std_sentiment = subset['compound'].std()
        negativity_ratio = (subset['sentiment_label'] == 'Negativo').mean()
        
        resultados_formais.append({
            'Métrica': nome,
            'Volume (N)': len(subset),
            'Média de Sentimento (VADER)': round(mean_sentiment, 3),
            'Desvio Padrão': round(std_sentiment, 3),
            'Proporção de Negatividade (%)': round(negativity_ratio * 100, 1)
        })
        
        # Adiciona flag ao dataframe para o boxplot
        subset['Metric_Category'] = nome
        if 'plot_df' not in locals():
            plot_df = subset
        else:
            plot_df = pd.concat([plot_df, subset])
            
    # 3. Gerar Tabela Acadêmica
    df_results = pd.DataFrame(resultados_formais)
    df_results.to_csv('data/metricas_analise_sentimento.csv', index=False)
    print("=== RESULTADOS DA ANÁLISE FORMAL DE SENTIMENTO ===")
    print(df_results.to_string(index=False))
    
    # 4. Plot Formal (Boxplot de sentimento por métrica)
    os.makedirs('docs/figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x='Metric_Category', y='compound', palette='Set2')
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Distribuição Formal do Sentimento por Métrica (VADER Compound)')
    plt.xlabel('Métrica')
    plt.ylabel('Score de Sentimento (-1 a 1)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('docs/figures/analise_sentimento_metricas.pdf')
    plt.close()
    
    print("\n[+] Tabela salva em: data/metricas_analise_sentimento.csv")
    print("[+] Gráfico salvo em: docs/figures/analise_sentimento_metricas.pdf")

if __name__ == "__main__":
    main()
