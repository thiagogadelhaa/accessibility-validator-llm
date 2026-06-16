import csv
from pathlib import Path
from typing import Dict, Set, List
import time

import ollama
import pandas as pd

from procecss import extract_wcag_codes, parse_supplementary_info, extract_predicted_wcag, calculate_metrics
from config import logger, CSV_PATH

def calculate_advanced_metrics(ground_truth: Set[str], predictions: Set[str]) -> Dict[str, float]:
    """
    Calcula TP, FP, FN e as métricas derivadas (Precision, Recall, F1).
    Inclui proteção contra divisão por zero.
    """
    tp = len(predictions.intersection(ground_truth))
    fp = len(predictions - ground_truth)
    fn = len(ground_truth - predictions)
    
    # Prevenção de divisão por zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4)
    }

def init_csv_file(filepath: Path):
    """
    Cria o arquivo CSV e escreve o cabeçalho caso ele ainda não exista.
    """
    headers = [
        "item_id", "model", "strategy", "duration_ms", 
        "tp", "fp", "fn", "precision", "recall", "f1_score", 
        "ground_truth", "predictions", "error"
    ]
    
    if not filepath.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_to_csv(filepath: Path, record: Dict):
    """
    Adiciona uma única linha ao CSV. Abertura em modo 'a' (append) garante
    resiliência: se o script falhar, os dados processados até o momento estão salvos.
    """
    headers = [
        "item_id", "model", "strategy", "duration_ms", 
        "tp", "fp", "fn", "precision", "recall", "f1_score", 
        "ground_truth", "predictions", "error"
    ]
    
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(record)

RESULTS_CSV_PATH = Path("./experiment_results/metrics_output.csv")

def build_prompt_with_strategy(strategy: str, base_context: str) -> str:
    # (Mantém a mesma implementação fornecida anteriormente)
    base_instruction = "Liste todas as violações de acessibilidade (WCAG) encontradas. Retorne APENAS os códigos das diretrizes (ex: 1.1.1, 1.3.1, 1.4.6)."
    
    if strategy == "zero-shot":
        return f"{base_instruction}\n\n{base_context}"
    elif strategy == "few-shot":
        examples = "Exemplos de Saída:\n- <img src='logo.png'> -> 1.1.1\n- <div aria-hidden='true'>... -> 1.3.1\n"
        return f"{base_instruction}\n{examples}\n\n{base_context}"
    elif strategy == "chain-of-thought":
        cot_instruction = "Analise o contexto passo a passo. 1) Identifique os elementos estruturais e visuais. 2) Avalie o contraste e atributos ARIA. 3) Determine a regra WCAG violada. 4) Por fim, extraia apenas os códigos numéricos."
        return f"{cot_instruction}\n{base_instruction}\n\n{base_context}"
    
    return f"{base_instruction}\n\n{base_context}"


def run_evaluation(
    client: ollama.Client,
    item_id: str,
    ground_truth: Set[str],
    text_prompt: str,
    images: List[str],
    models: List[str],
    strategies: List[str]
):
    """
    Executa a inferência, grava logs estruturados e persiste os resultados no CSV.
    """
    # Inicializa o CSV garantindo a presença do cabeçalho
    init_csv_file(RESULTS_CSV_PATH)
    
    for model in models:
        for strategy in strategies:
            final_prompt = build_prompt_with_strategy(strategy, text_prompt)
            start_time = time.perf_counter()
            
            logger.info("inference_started", item_id=item_id, model=model, strategy=strategy)
            
            # Estrutura base do registro para o CSV
            record = {
                "item_id": item_id,
                "model": model,
                "strategy": strategy,
                "duration_ms": 0,
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                "ground_truth": "|".join(ground_truth), # Salva como string delimitada para não quebrar o CSV
                "predictions": "",
                "error": ""
            }
            
            try:
                response = client.generate(
                    model=model,
                    prompt=final_prompt,
                    images=images,
                    stream=False
                )
                
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                raw_output = response.get('response', '')
                
                predicted_codes = extract_predicted_wcag(raw_output) 
                metrics = calculate_advanced_metrics(ground_truth, predicted_codes)
                
                # Atualiza o registro com sucesso
                record.update({
                    "duration_ms": duration_ms,
                    "predictions": "|".join(predicted_codes),
                    **metrics
                })
                
                logger.info("inference_success", item_id=item_id, model=model, strategy=strategy, metrics=metrics)
                
            except Exception as e:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                error_msg = str(e)
                
                # Atualiza o registro refletindo a falha
                record.update({
                    "duration_ms": duration_ms,
                    "error": error_msg
                })
                
                logger.error("inference_failed", item_id=item_id, model=model, strategy=strategy, error=error_msg)
            
            finally:
                # O bloco finally garante que a linha será salva no CSV, independentemente
                # de sucesso (try) ou falha de rede/OOM (except).
                append_to_csv(RESULTS_CSV_PATH, record)


def process_dataset(client: ollama.Client, models: List[str], strategies: List[str]):
    """
    Lê o dataset, prepara o payload de inferência e aciona o runner.
    Itera linha a linha para manter footprint de memória baixo.
    """
    if not Path(CSV_PATH).exists():
        logger.error("dataset_not_found", path=str(CSV_PATH))
        return

    logger.info("dataset_ingestion_started", path=str(CSV_PATH))
    
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        logger.error("dataset_load_failed", error=str(e))
        return

    total_rows = len(df)
    processed_count = 0

    for index, row in df.iterrows():
        item_id = str(row['id'])
        wcag_raw = row.get('wcag_reference', '')
        affected_elements = row.get('affected_html_elements', '')
        supp_info_raw = row.get('supplementary_information', '')
        
        ground_truth_codes = extract_wcag_codes(wcag_raw)
        
        if not ground_truth_codes:
            logger.debug("skipping_row_no_ground_truth", item_id=item_id)
            continue
            
        image_paths, extra_text_context = parse_supplementary_info(supp_info_raw)
        
        prompt_payload = f"Abaixo estão os elementos HTML afetados por potenciais violações:\n"
        prompt_payload += f"```html\n{affected_elements}\n```\n"
        
        if extra_text_context:
            prompt_payload += f"\nContexto Suplementar:\n{extra_text_context}\n"

        logger.info(
            "item_ready_for_inference", 
            item_id=item_id, 
            ground_truth=list(ground_truth_codes),
            image_count=len(image_paths),
            progress=f"{processed_count + 1}/{total_rows}"
        )
        
        run_evaluation(
            client=client,
            item_id=item_id,
            ground_truth=ground_truth_codes,
            text_prompt=prompt_payload,
            images=image_paths,
            models=models,
            strategies=strategies
        )
        
        processed_count += 1

    logger.info("dataset_ingestion_completed", total_processed=processed_count)

if __name__ == "__main__":
    logger.info("Iniciando o experimento...\n")
    
    ollama_client = ollama.Client(host="http://localhost:11434")
    
    MODELS_TO_TEST = ["gemma4:e2b"]
    STRATEGIES_TO_TEST = ["zero-shot", "few-shot", "chain-of-thought"]
    
    process_dataset(
        client=ollama_client,
        models=MODELS_TO_TEST,
        strategies=STRATEGIES_TO_TEST
    )

    calculate_metrics()