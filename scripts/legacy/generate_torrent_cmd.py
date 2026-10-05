import requests
import bencode
import os

TORRENT_URL = "https://academictorrents.com/download/1614740ac8c94505e4ecb9d88be8bed7b6afddd4.torrent"
SUBREDDITS = ["cscareerquestions", "experienceddevs", "womenintech", "softwareengineering", "devops"]

print("Baixando arquivo .torrent...")
headers = {'User-Agent': 'Mozilla/5.0'}
r = requests.get(TORRENT_URL, headers=headers, allow_redirects=True)
try:
    torrent_data = bencode.bdecode(r.content)
except Exception as e:
    print(f"Erro ao ler bencode. Resposta HTTP: {r.status_code}")
    print(r.text[:200])
    exit(1)

info = torrent_data['info']
files = info['files']

indices = []
print("Procurando subreddits no torrent...")
for i, file_info in enumerate(files):
    path = file_info['path']
    # path is a list of strings
    filename = path[-1].lower()
    for sub in SUBREDDITS:
        if filename.startswith(f"{sub}_submissions") or filename.startswith(f"{sub}_comments"):
            indices.append(str(i + 1)) # aria2c indices are 1-based
            print(f"Encontrado: {filename} (índice {i+1})")

if not indices:
    print("Nenhum subreddit encontrado!")
else:
    idx_str = ",".join(indices)
    print("\n===============================")
    print("Execute o comando abaixo no seu terminal para baixar APENAS os arquivos necessários:")
    print("===============================\n")
    print(f'aria2c "{TORRENT_URL}" --select-file={idx_str} --dir=data/reddit/subreddits24/')
    print("\n(Requer ter o aria2 instalado: sudo apt install aria2)")
