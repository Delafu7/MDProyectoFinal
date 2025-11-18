import random
import pandas as pd
import json
from pathlib import Path
from transformers import RobertaTokenizer, RobertaModel
import torch
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score ,precision_recall_curve, average_precision_score,accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import seaborn as sns
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ast
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from scipy.spatial.distance import cosine
import joblib
from sklearn.metrics import classification_report, confusion_matrix ,roc_curve, auc


PALABRAS_RUIDO = [
    "perro", "nube", "cosa", "elemento", "random", "xD", "jeje", "árbol", "silla", "objeto", "rumor", "pixel", "caja", "supercalifragilisticoespialidoso"
]



def baseline_robertuito(input_csv, output_dir="output_robertuito", seed=42):
    """
    Entrena un baseline con el modelo 'pysentimiento/robertuito-sentiment-analysis'
    usando embeddings + distancias coseno + probabilidades de sentimiento.
    Guarda métricas, parámetros y genera gráficas.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ROBERTA_MODEL = "pysentimiento/robertuito-sentiment-analysis"

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    model_cls = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL).to(device)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    predef_msg = "esto es lo peor que me ha pasado, tengo depresion, y no puedo dejar de comer comida poco sana"
    predef_emb = embedder.encode(predef_msg, convert_to_numpy=True)

    print(" Modelos cargados en", device)

    # Leer CSV
    df = pd.read_csv(input_csv)
    if "mensajes" in df.columns:
        # Si ya es texto plano (después del ruido), simplemente úsalo tal cual
        texts = df["mensajes"].astype(str).tolist()
    elif "text" in df.columns:
        texts = df["text"].astype(str).tolist()
    else:
        raise ValueError("El CSV debe contener columna 'mensajes' o 'text'")

    if "label" in df.columns:
        y = df["label"].astype(int).to_numpy()
    elif "sufre" in df.columns:
        y = df["sufre"].astype(int).to_numpy()
    else:
        raise ValueError("No se encontró ninguna columna de etiquetas ('label' o 'sufre') en el CSV.")

    # --- Funciones auxiliares ---
    def get_sentiment_probs(texts, batch_size=16):
        model_cls.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                out = model_cls(**tokens)
                logits = out.logits
                batch_probs = F.softmax(logits, dim=-1).cpu().numpy()
                probs.append(batch_probs)
        return np.vstack(probs)

    def get_embeddings_and_dists(texts):
        embs = embedder.encode(texts, convert_to_numpy=True)
        dists = np.array([cosine(e, predef_emb) for e in embs]).reshape(-1, 1)
        return embs, dists

    # --- Generar features ---
    print("🔹 Generando features...")
    probs = get_sentiment_probs(texts)
    embs, dists = get_embeddings_and_dists(texts)
    X = np.hstack([embs, probs, dists])

    print(" Features generadas:", X.shape)

    # --- Split train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # --- Pipeline + búsqueda ---
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(random_state=seed, max_iter=500))
    ])

    param_grid = {
        "mlp__hidden_layer_sizes": [(64,), (128,), (128, 64)],
        "mlp__alpha": [1e-4, 1e-3],
        "mlp__learning_rate_init": [1e-3, 5e-4]
    }

    print(" Entrenando MLP (GridSearchCV)...")
    gs = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(" Entrenamiento completado. Mejores parámetros:", gs.best_params_)

    # --- Evaluación ---
    y_pred = gs.predict(X_test)
    y_prob = gs.predict_proba(X_test)[:, 1] if hasattr(gs, "predict_proba") else None

    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")

    print(f"\n🔹 F1 macro: {f1:.4f}")
    print("\n Reporte de clasificación:\n", classification_report(y_test, y_pred))
    print(" Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    # Guardar modelo
    model_path = os.path.join(output_dir, f"robertuito_mlp_{Path(input_csv).stem}.joblib")
    joblib.dump(gs.best_estimator_, model_path)
    print(f" Modelo guardado en: {model_path}")

    # === Gráficas ===

    base_name = Path(input_csv).stem

    # Curva Precision–Recall
    """
    if y_prob is not None:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure()
        plt.plot(recall, precision, label=f"F1 = {f1:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Curva Precision–Recall ({base_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/PR_curve_{base_name}.png", dpi=300)
        plt.close()

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Curva ROC ({base_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/ROC_curve_{base_name}.png", dpi=300)
        plt.close()
    """
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusión ({base_name})")
    plt.tight_layout()
    plt.savefig(f"plots/confusion_matrix_{base_name}.png", dpi=300)
    plt.close()

    # === Guardar resultados en CSV ===
    from datetime import datetime
    results_file = "resultados_baseline.csv"

    registro = pd.DataFrame([{
        "condicion": base_name,
        "modelo": "MLPClassifier",
        "parametros": str(gs.best_params_),
        "f1_macro": round(f1, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo_path": model_path
    }])

    if os.path.exists(results_file):
        registro.to_csv(results_file, mode='a', header=False, index=False)
    else:
        registro.to_csv(results_file, index=False)

    print(f" Resultados y gráficas guardados en: {results_file} / plots/")

    return f1


def procesar_mentalrisk_arrays(root_folder, output_folder):
    """Genera 4 CSVs: original + versiones con ruido (1%, 5%, 15%)."""
    root_folder = Path(root_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cond = root_folder / "Depresión"

    if not cond.exists() or not cond.is_dir():
        print("La carpeta 'Depresión' no existe dentro del root.")
        return

    print(f"\nProcesando condición: {cond.name}")

    df_train = leer_subcarpeta(cond / 'Train')
    df_test = leer_subcarpeta(cond / 'Test')
    df_final = pd.concat([df_train, df_test], ignore_index=True)

    if df_final.empty:
        print("Aviso: no hay datos para la condición Depresión.")
        return

    df_final['mensajes'] = df_final['mensajes'].apply(str)

    # --- 1) CSV ORIGINAL ---
    output_file = output_folder / "Depresión.csv"
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    print(f" Archivo creado: {output_file}")

    # --- Función interna para replicar ruido ---
    def generar_csv_ruido(df, porcentaje, nombre):
        df_ruido = df.copy()
        df_ruido['mensajes'] = df_ruido['mensajes'].apply(
            lambda x: introducir_ruido_combinado(x, porcentaje)
        )
        output = output_folder / nombre
        df_ruido.to_csv(output, index=False, encoding='utf-8')
        print(f" Archivo creado: {output}")

    # --- 2) Ruido 5% ---
    generar_csv_ruido(df_final, 0.05, "Depresión_ruido_5.csv")
    # --- 3) Ruido 15% ---
    generar_csv_ruido(df_final, 0.15, "Depresión_ruido_15.csv")
    # --- 4) Ruido 30% ---
    generar_csv_ruido(df_final, 0.30, "Depresión_ruido_30.csv")
    # --- 3) Ruido 50% ---
    generar_csv_ruido(df_final, 0.50, "Depresión_ruido_50.csv")

def leer_subcarpeta(subfolder: Path) -> pd.DataFrame:
    """Lee una subcarpeta (Train/Test) del dataset MentalRisk y devuelve un DataFrame con sujeto, mensajes y etiqueta."""
    gold_file = subfolder / 'Gold' / 'gold_a.txt'
    if not gold_file.exists():
        print(f"Aviso: no existe gold_a.txt en {subfolder}, se ignorará")
        return pd.DataFrame(columns=['sujeto', 'mensajes', 'sufre'])

    gold = pd.read_csv(gold_file, header=None, names=['subject', 'label'])

    data = []
    for json_file in subfolder.glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            mensajes_json = json.load(f)
        mensajes_list = [m['message'] for m in mensajes_json]
        data.append({'sujeto': json_file.stem, 'mensajes': mensajes_list})

    df = pd.DataFrame(data)
    if df.empty:
        return df

    df = pd.merge(df, gold, left_on='sujeto', right_on='subject', how='left')
    df.rename(columns={'label': 'sufre'}, inplace=True)
    return df[['sujeto', 'mensajes', 'sufre']]

def introducir_ruido_en_texto(texto, porcentaje):
    """
    Modifica un porcentaje de palabras del texto.
    Ruido simple: altera una letra aleatoria dentro de la palabra.
    """
    palabras = texto.split()
    n_palabras = len(palabras)
    if n_palabras == 0:
        return texto

    # Cantidad de palabras a alterar
    n_ruido = max(1, int(n_palabras * porcentaje))

    indices = random.sample(range(n_palabras), n_ruido)

    for i in indices:
        palabra = palabras[i]
        if len(palabra) > 1:
            # Alteramos una letra al azar
            pos = random.randint(0, len(palabra) - 1)
            letra_random = chr(random.randint(97, 122))  # letra a–z
            palabra_ruido = palabra[:pos] + letra_random + palabra[pos+1:]
            palabras[i] = palabra_ruido

    return " ".join(palabras)

def alterar_palabra(palabra):
    """Cambia una letra aleatoria de la palabra."""
    if len(palabra) < 2:
        return palabra
    pos = random.randint(0, len(palabra) - 1)
    letra_random = chr(random.randint(97, 122))  # letra a-z
    return palabra[:pos] + letra_random + palabra[pos+1:]

def introducir_ruido_combinado(texto, porcentaje):
    """
    Introduce dos tipos de ruido:
    1) Cambia un porcentaje de palabras del texto
    2) Inserta palabras irrelevantes (mismo porcentaje)
    """
    palabras = texto.split()
    n_palabras = len(palabras)
    if n_palabras == 0:
        return texto

    # Cantidad de palabras a modificar y agregar
    n_cambios = max(1, int(n_palabras * porcentaje))
    n_insertar = max(1, int(n_palabras * porcentaje))

    # --- 1) Alterar palabras existentes ---
    indices_cambiar = random.sample(range(n_palabras), min(n_cambios, n_palabras))
    for i in indices_cambiar:
        palabras[i] = alterar_palabra(palabras[i])

    # --- 2) Insertar palabras irrelevantes ---
    for _ in range(n_insertar):
        palabra_ruido = random.choice(PALABRAS_RUIDO)
        pos = random.randint(0, len(palabras))
        palabras.insert(pos, palabra_ruido)

    return " ".join(palabras)

if __name__ == "__main__":
    hacer_embeddings = True
    # Paso 1: procesar archivos JSON → CSVs con mensajes y etiquetas
    procesar_mentalrisk_arrays("MentalRiskEs2023", "output_ruido")
    baseline_robertuito("output_ruido/Depresión.csv")
    baseline_robertuito("output_ruido/Depresión_ruido_5.csv")
    baseline_robertuito("output_ruido/Depresión_ruido_15.csv")
    baseline_robertuito("output_ruido/Depresión_ruido_30.csv")
    baseline_robertuito("output_ruido/Depresión_ruido_50.csv")
