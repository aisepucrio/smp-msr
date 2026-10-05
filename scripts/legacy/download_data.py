import os
import requests

SUBREDDITS = [
    "cscareerquestions",
    "ExperiencedDevs",
    "womenintech",
    "softwareengineering",
    "devops",
]

# The-Eye (RedArcs) é um mirror HTTP confiável dos dumps do Reddit (Pushshift)
BASE_URL = "https://the-eye.eu/redarcs/files"
OUTPUT_DIR = "data/reddit/subreddits24"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(url, output_path):
    print(f"Baixando: {url}")
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        with requests.get(url, stream=True, verify=False) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Sucesso: {output_path}")
    except Exception as e:
        print(f"Erro ao baixar {url}: {e}")

if __name__ == "__main__":
    for sub in SUBREDDITS:
        # A API e os mirrors costumam manter o casing exato ou minúsculo. 
        # RedArcs usa lowercase para os nomes dos arquivos.
        sub_lower = sub.lower()
        
        sub_file = f"{sub_lower}_submissions.zst"
        com_file = f"{sub_lower}_comments.zst"
        
        # Download submissions
        download_file(f"{BASE_URL}/{sub_file}", os.path.join(OUTPUT_DIR, sub_file))
        
        # Download comments
        download_file(f"{BASE_URL}/{com_file}", os.path.join(OUTPUT_DIR, com_file))
        
    print("\nDownload concluído! Agora você pode rodar o extrator offline.")
