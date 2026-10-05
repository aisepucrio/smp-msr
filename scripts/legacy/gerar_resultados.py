import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Ensure output directory exists
os.makedirs('docs/figures', exist_ok=True)

# Find the latest unique posts CSV in the data folder
csv_files = glob.glob('data/*_posts_unique.csv')
if not csv_files:
    print("Nenhum CSV de posts únicos encontrado na pasta data/")
    exit(1)
latest_csv = max(csv_files, key=os.path.getctime)
print(f"Lendo dados de: {latest_csv}")

df = pd.read_csv(latest_csv)
df['created_utc'] = pd.to_datetime(df['created_utc'])

# Define colors for themes
theme_colors = {'mental_health': '#e74c3c', 'metrics_productivity': '#3498db'}

# 1. Engagement by Theme (Score)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='query_group', y='score', palette=theme_colors)
plt.yscale('log') # Log scale because reddit scores are heavily skewed
plt.title('Distribuição de Engajamento (Score) por Tema')
plt.xlabel('Tema (Query Group)')
plt.ylabel('Score (Escala Log)')
plt.tight_layout()
plt.savefig('docs/figures/engajamento_score.pdf')
plt.close()

# 2. Distribution of Themes by Subreddit
plt.figure(figsize=(10, 6))
theme_counts = df.groupby(['subreddit', 'query_group']).size().unstack(fill_value=0)
theme_counts.plot(kind='bar', stacked=True, color=[theme_colors.get(c, '#333') for c in theme_counts.columns], figsize=(10, 6))
plt.title('Distribuição de Temas por Subreddit')
plt.xlabel('Subreddit')
plt.ylabel('Número de Posts')
plt.xticks(rotation=45)
plt.legend(title='Tema')
plt.tight_layout()
plt.savefig('docs/figures/temas_por_subreddit.pdf')
plt.close()

# 3. Temporal Trend (Volume of Posts over time)
df['year_month'] = df['created_utc'].dt.to_period('M')
temporal_counts = df.groupby(['year_month', 'query_group']).size().unstack(fill_value=0)
temporal_counts.index = temporal_counts.index.astype(str)

plt.figure(figsize=(12, 6))
for col in temporal_counts.columns:
    plt.plot(temporal_counts.index, temporal_counts[col], marker='o', label=col, color=theme_colors.get(col, '#333'), linewidth=2)
plt.title('Evolução Temporal do Volume de Posts (2021-2024)')
plt.xlabel('Mês/Ano')
plt.ylabel('Número de Posts')
plt.xticks(rotation=45)
plt.legend(title='Tema')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('docs/figures/tendencia_temporal.pdf')
plt.close()

print("Gráficos gerados com sucesso na pasta docs/figures/")
