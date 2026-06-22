import ast
import re
import base64
import mimetypes

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set, List, Tuple
from bs4 import BeautifulSoup

from config import logger, LOCAL_SCREENSHOTS_DIR


def parse_supplementary_info(supp_info: str) -> Tuple[List[str], str]:
    """
    Analisa a coluna supplementary_information.
    Retorna uma tupla: (Lista de caminhos de imagens válidos, Texto suplementar).
    """
    images = []
    text_context = ""
    
    # Tratamento de valores nulos do Pandas (NaN)
    if pd.isna(supp_info) or not str(supp_info).strip():
        return images, text_context

    supp_str = str(supp_info)

    # Identifica se o campo contém caminhos de arquivos de imagem
    if ".png" in supp_str or ".jpg" in supp_str:
        raw_paths = [p.strip() for p in supp_str.split(",")]
        for rp in raw_paths:
            if rp.endswith((".png", ".jpg")):
                # Extrai apenas o nome do arquivo do path absoluto original do CSV
                filename = Path(rp).name 
                local_path = LOCAL_SCREENSHOTS_DIR / filename
                
                if local_path.exists():
                    images.append(str(local_path))
                else:
                    logger.warning("missing_image_file", expected_path=str(local_path))
    else:
        # Se não há extensão de imagem, assumimos que é contexto textual (ex: link-name)
        text_context = supp_str.strip()

    return images, text_context


def extract_predicted_wcag(llm_output: str) -> Set[str]:
    """
    Extrai códigos numéricos WCAG da resposta livre (texto natural) do LLM.
    Não utiliza ast.literal_eval para garantir tolerância a alucinações de formato.
    """
    try:
        if not llm_output:
            return set()
            
        # Busca direta em qualquer parte do texto retornando o padrão X.Y.Z
        pattern = r"\b[1-4]\.\d+\.\d+\b"
        matches = re.findall(pattern, str(llm_output))
        return set(matches)
        
    except Exception as e:
        logger.warning("failed_to_extract_prediction", output=str(llm_output), error=str(e))
        return set()

def extract_wcag_codes(wcag_string: str) -> Set[str]:
    """Extrai os códigos WCAG (ex: 1.1.1) da string bruta."""
    try:
        wcag_list = ast.literal_eval(wcag_string)
        codes = set()
        for item in wcag_list:
            match = re.search(r"\b[1-4]\.\d+\.\d+\b", item)
            if match:
                codes.add(match.group())
        return codes
    except Exception as e:
        logger.warning("failed_to_parse_wcag", wcag_string=wcag_string, error=str(e))
        return set()


def sanitize_html_for_llm(html_content: str) -> str:
    """
    Remove payloads pesados (Base64, Geometria SVG, Scripts) do HTML 
    para economizar tokens, mantendo a integridade semântica para validação WCAG.
    """
    if not html_content or not isinstance(html_content, str):
        return ""

    
    # Busca o padrão data:image/... e substitui o payload gigante por um placeholder.
    b64_pattern = re.compile(r'(src|href)=["\']data:image\/[^;]+;base64,[a-zA-Z0-9+/=]+["\']')
    html_content = b64_pattern.sub(r'\1="data:image/[REMOVIDO_PARA_ECONOMIA_DE_TOKENS]"', html_content)

    try:
        soup = BeautifulSoup(html_content, "html.parser")

        
        for svg in soup.find_all('svg'):
            for geom in svg.find_all(['path', 'polygon', 'polyline', 'g', 'rect', 'circle', 'defs']):
                geom.decompose()
            
            if not svg.string:
                svg.append(soup.new_string(" "))

        for tag in soup.find_all(['script', 'style', 'noscript']):
            tag.decompose()

        sanitized_html = str(soup)

    except Exception as e:
        logger.error("html_sanitization_failed", error=str(e))
        # Fallback de segurança: retorna o HTML (com o Base64 removido pela regex)
        sanitized_html = html_content
    
    return sanitized_html


def calculate_metrics():
    df = pd.read_csv("./experiment_results/metrics_output_final.csv")

    error_mask = df['error'].astype(str).str.startswith("Error code: 400", na=False)
    
    # Extrai a lista de IDs únicos onde o limite de tokens foi excedido
    failed_item_ids = df[error_mask]['item_id'].unique()
    
    # Exportação para a Seção de Limitações do Artigo
    if len(failed_item_ids) > 0:
        dropped_output_path = Path('./experiment_results/dropped_items_limitations.csv')
        dropped_df = pd.DataFrame({'failed_item_id': failed_item_ids})
        dropped_df.to_csv(dropped_output_path, index=False)

    # Mantém no cálculo apenas os registros que não fazem parte da lista de falhas
    clean_df = df[~df['item_id'].isin(failed_item_ids)].copy()

    # 4. Agrupamento e soma dos valores brutos sobre o dataset estabilizado
    global_metrics = clean_df.groupby(['model', 'strategy'])[['tp', 'fp', 'fn']].sum().reset_index()

    # 5. Cálculo de Métricas com Segurança (Evita Divisão por Zero)
    global_metrics['precision'] = np.where(
        (global_metrics['tp'] + global_metrics['fp']) == 0, 0.0, 
        global_metrics['tp'] / (global_metrics['tp'] + global_metrics['fp'])
    )
    
    global_metrics['recall'] = np.where(
        (global_metrics['tp'] + global_metrics['fn']) == 0, 0.0, 
        global_metrics['tp'] / (global_metrics['tp'] + global_metrics['fn'])
    )
    
    global_metrics['f1_score'] = np.where(
        (global_metrics['precision'] + global_metrics['recall']) == 0, 0.0, 
        (2 * global_metrics['precision'] * global_metrics['recall']) / (global_metrics['precision'] + global_metrics['recall'])
    )

    # Arredondamento para facilitar a leitura nos relatórios e artigos
    cols_to_round = ['precision', 'recall', 'f1_score']
    global_metrics[cols_to_round] = global_metrics[cols_to_round].round(4)

    # 6. Exportação das Métricas Finais
    final_metrics_path = Path('./experiment_results/final_metrics_fair_baseline.csv')
    global_metrics.to_csv(final_metrics_path, index=False)


def encode_image_for_opeanai(image_path: str) -> str:
    """
    Lê uma imagem local e converte para o formato Data URI do padrão OpenAI.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
        
    return f"data:{mime_type};base64,{encoded_string}"
