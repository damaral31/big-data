import os
import requests
from google.cloud import bigquery
from google.cloud import storage

# 1. Autenticação
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"./gcp_key.json"

GCP_PROJECT_ID = "big-data-497416"
MIMIC_DATASET = "mimic_prod"
# COLOCA AQUI O NOME DO TEU BUCKET (o que criaste no início do projeto)
BUCKET_NAME = "big-data-visty-up" 

bq_client = bigquery.Client(project=GCP_PROJECT_ID)
storage_client = storage.Client(project=GCP_PROJECT_ID)

def transferir_para_gcs_e_bq(url_origem, nome_tabela, esquema):
    nome_ficheiro = f"{nome_tabela.upper()}.csv.gz"
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(nome_ficheiro)
    
    # Passo A: Baixar da UP e enviar para o Cloud Storage
    print(f"📦 A transferir {nome_ficheiro} da UP para o Google Cloud Storage...")
    response = requests.get(url_origem, stream=True)
    if response.status_code == 200:
        # Envia o fluxo de dados diretamente para o teu bucket sem encher o teu disco local
        blob.upload_from_file(response.raw, content_type="application/gzip")
        print(f"  ✓ Guardado em gs://{BUCKET_NAME}/{nome_ficheiro}")
    else:
        raise Exception(f"Erro ao aceder ao link da UP: Status {response.status_code}")

    # Passo B: Criar a tabela no BigQuery apontando para o GCS
    print(f"🏛️ A indexar '{nome_tabela}' no BigQuery...")
    table_id = f"{GCP_PROJECT_ID}.{MIMIC_DATASET}.{nome_tabela}"
    gcs_uri = f"gs://{BUCKET_NAME}/{nome_ficheiro}"
    
    job_config = bigquery.LoadJobConfig(
        schema=esquema,
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
    )
    
    load_job = bq_client.load_table_from_uri(gcs_uri, table_id, job_config=job_config)
    load_job.result()  # Aguarda a conclusão do BigQuery
    print(f"  ✓ Tabela '{nome_tabela}' pronta a usar!\n")

# ==============================================================================
# ESQUEMAS FORMAIS DO MIMIC-III
# ==============================================================================
esquema_admissions = [
    bigquery.SchemaField("ROW_ID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("SUBJECT_ID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("HADM_ID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("ADMITTIME", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("DISCHTIME", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("DEATHTIME", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("ADMISSION_TYPE", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("ADMISSION_LOCATION", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("DISCHARGE_LOCATION", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("INSURANCE", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("LANGUAGE", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("RELIGION", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("MARITAL_STATUS", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("ETHNICITY", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("EDREGTIME", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("EDOUTTIME", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("DIAGNOSIS", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("HOSPITAL_EXPIRE_FLAG", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("HAS_CHARTEVENTS_DATA", "INTEGER", mode="REQUIRED"),
]

esquema_icustays = [
    bigquery.SchemaField("ROW_ID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("SUBJECT_ID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("HADM_ID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("ICUSTAY_ID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("DBSOURCE", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("FIRST_CAREUNIT", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("LAST_CAREUNIT", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("FIRST_WARDID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("LAST_WARDID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("INTIME", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("OUTTIME", "TIMESTAMP", mode="NULLABLE"),  # <- Corrigido para NULLABLE
    bigquery.SchemaField("LOS", "FLOAT", mode="NULLABLE"),          # <- Garantido como NULLABLE
]

# ==============================================================================
# EXECUÇÃO DO FLUXO
# ==============================================================================
try:
    transferir_para_gcs_e_bq(
        "https://www.dcc.fc.up.pt/~ines/MIMIC-III/ADMISSIONS.csv.gz", 
        "admissions", 
        esquema_admissions
    )
    
    transferir_para_gcs_e_bq(
        "https://www.dcc.fc.up.pt/~ines/MIMIC-III/ICUSTAYS.csv.gz", 
        "icustays", 
        esquema_icustays
    )
    
    print("🚀 Sucesso total! Dados reais injetados no BigQuery.")

except Exception as e:
    print(f"❌ Erro crítico: {e}")