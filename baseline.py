import pandas as pd
import json
from pathlib import Path
from transformers import RobertaTokenizer, RobertaModel
import torch
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score ,precision_recall_curve, average_precision_score
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

# -----------------------------------------------------
# FRASES SEMILLA
# -----------------------------------------------------
SEMILLAS = {
    "depresion": [
        "Me siento sin energía ni motivación.",
        "No tengo ganas de hacer nada.",
        "Me siento triste todo el tiempo."
    ],
    "ansiedad": [
        "No puedo dejar de preocuparme.",
        "Siento miedo constante y no sé por qué.",
        "Mi corazón late muy rápido, me cuesta respirar."
    ],
    "tca": [
        "Me siento culpable cuando como.",
        "No me gusta mi cuerpo.",
        "Trato de comer lo menos posible."
    ]
}

def cosine_similarity(A, B):
    """Calcula la similitud coseno entre dos conjuntos de vectores A y B."""
    A_norm = A / A.norm(dim=1)[:, None]
    B_norm = B / B.norm(dim=1)[:, None]
    return torch.mm(A_norm, B_norm.t()).numpy()

def generar_similitudes_semanticas(input_folder, output_file, model_name="roberta-base", pooling="cls"):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()

    input_folder = Path(input_folder)
    resultados = []

    # Pre-calcular embeddings de las frases semilla
    semillas_embeddings = {}
    for categoria, frases in SEMILLAS.items():
        tokens = tokenizer(frases, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
            if pooling == "cls":
                emb = outputs.last_hidden_state[:, 0, :]
            else:
                emb = outputs.last_hidden_state.mean(dim=1)
        semillas_embeddings[categoria] = emb

    # Calcular número total de filas a procesar (para progreso global)
    total_filas = sum(len(pd.read_csv(f)) for f in input_folder.glob("*.csv"))

    # Crear barra de progreso global
    with tqdm(total=total_filas, desc="Procesando archivos", unit="fila") as pbar:
        for csv_file in input_folder.glob('*.csv'):
            print(f"\n🔹 Procesando {csv_file.name}")
            df = pd.read_csv(csv_file, converters={'mensajes': ast.literal_eval})

            for _, row in df.iterrows():
                usuario = row['sujeto']
                mensajes = row['mensajes']
                label = row['sufre']

                tokens = tokenizer(mensajes, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**tokens)
                    if pooling == "cls":
                        emb_msgs = outputs.last_hidden_state[:, 0, :]
                    else:
                        emb_msgs = outputs.last_hidden_state.mean(dim=1)
                #emb_usuario = emb_msgs.mean(dim=0, keepdim=True)
                sim_prom = {}
                for cat, emb_cat in semillas_embeddings.items():
                    sim = cosine_similarity(emb_msgs.cpu(), emb_cat.cpu())
                    sim_prom[cat] = np.mean(sim)

                resultados.append({
                    "usuario": usuario,
                    **sim_prom,
                    "label": label
                })

                # Actualizar barra de progreso
                pbar.update(1)
    
    # Guardar resultados
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out = pd.DataFrame(resultados)
    df_out.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Archivo de similitudes guardado en: {output_file}")



# -----------------------------------------------------
#  FUNCIONES AUXILIARES
# -----------------------------------------------------

def leer_subcarpeta(subfolder: Path) -> pd.DataFrame:
    """Lee una subcarpeta (Train/Test) del dataset MentalRisk y devuelve un DataFrame con sujeto, mensajes y etiqueta."""
    gold_file = subfolder / 'Gold' / 'gold_a.txt'
    if not gold_file.exists():
        print(f"Aviso: no existe gold_a.txt en {subfolder}, se ignorará")
        return pd.DataFrame(columns=['sujeto', 'mensajes', 'sufre'])

    # Leer etiquetas (no tiene cabecera)
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


# -----------------------------------------------------
#  PROCESAMIENTO INICIAL DEL DATASET
# -----------------------------------------------------

def procesar_mentalrisk_arrays(root_folder, output_folder):
    """Lee todas las condiciones de MentalRisk (TCA, Depresion, Ansiedad) y genera CSVs combinados con mensajes y labels."""
    root_folder = Path(root_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    condiciones = [d for d in root_folder.iterdir() if d.is_dir()]

    for cond in condiciones:
        print(f"\nProcesando condición: {cond.name}")

        df_train = leer_subcarpeta(cond / 'Train')
        df_test = leer_subcarpeta(cond / 'Test')
        df_final = pd.concat([df_train, df_test], ignore_index=True)

        if df_final.empty:
            print(f"Aviso: no hay datos para la condición {cond.name}, se ignora.")
            continue

        output_file = output_folder / f"{cond.name}.csv"
        # Guardamos los mensajes como string (para poder recuperarlos después)
        df_final['mensajes'] = df_final['mensajes'].apply(lambda x: str(x))
        df_final.to_csv(output_file, index=False, encoding='utf-8')
        print(f" Archivo creado: {output_file}")


# -----------------------------------------------------
#  GENERACIÓN DE EMBEDDINGS
# -----------------------------------------------------

def generar_embeddings_csv(input_folder, output_folder, model_name="roberta-base", pooling="cls"):
    """
    Genera embeddings con RoBERTa y guarda un CSV con usuario, embedding y label.
    pooling puede ser:
        - 'cls': usa el token [CLS]
        - 'mean': promedio de tokens
    """
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.eval()

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for csv_file in input_folder.glob('*.csv'):
        print(f"\n🔹 Generando embeddings para: {csv_file.name}")
        df = pd.read_csv(csv_file, converters={'mensajes': ast.literal_eval})

        usuarios, embeddings_list, labels = [], [], []

        for _, row in df.iterrows():
            usuario = row['sujeto']
            mensajes = row['mensajes']
            label = row['sufre']

            # Tokenizar en batch
            tokens = tokenizer(
                mensajes, padding=True, truncation=True, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**tokens)
                last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

                if pooling == "cls":
                    emb = last_hidden[:, 0, :]  # (batch, hidden)
                elif pooling == "mean":
                    emb = last_hidden.mean(dim=1)  # Promedio de tokens
                else:
                    raise ValueError("pooling debe ser 'cls' o 'mean'")

            emb_usuario = emb.mean(dim=0)  # (hidden)
            usuarios.append(usuario)
            embeddings_list.append(emb_usuario.detach().cpu().numpy().tolist())
            labels.append(label)

        df_out = pd.DataFrame({
            'usuario': usuarios,
            'embedding': embeddings_list,
            'label': labels
        })

        output_file = output_folder / csv_file.name
        df_out.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Embeddings guardados en: {output_file}")

# -----------------------------------------------------
#  EVALUACIÓN DEL BASELINE
# -----------------------------------------------------

def evaluar_baseline(embeddings_folder, modo="embeddings", output_plots="plots"):
    """
    Evalúa un baseline sobre embeddings o similitudes semánticas.
    Guarda automáticamente las gráficas generadas.
    Puede recibir una carpeta o un único archivo CSV.
    """
    embeddings_folder = Path(embeddings_folder)
    output_plots = Path(output_plots)
    output_plots.mkdir(parents=True, exist_ok=True)  # Crear carpeta si no existe

    # 🔹 Detectar si es archivo único o carpeta
    if embeddings_folder.is_file():
        csv_files = [embeddings_folder]
    else:
        csv_files = list(embeddings_folder.glob("*.csv"))

    if not csv_files:
        print(f"⚠️ No se encontraron archivos CSV en {embeddings_folder.resolve()}")
        return pd.DataFrame()

    resultados = []

    for csv_file in csv_files:
        print(f"\n📄 Evaluando baseline para: {csv_file.name}")
        df = pd.read_csv(csv_file)

        # --- Detectar modo ---
        if modo == "auto":
            modo = "similitudes" if {"depresion", "ansiedad", "tca"}.issubset(df.columns) else "embeddings"

        # --- Preparar datos ---
        if modo == "embeddings":
            df["embedding"] = df["embedding"].apply(ast.literal_eval)
            X = np.vstack(df["embedding"].values)
            y = df["label"].values
        elif modo == "similitudes":
            X = df[["depresion", "ansiedad", "tca"]].values
            y = df["label"].values
        else:
            raise ValueError("modo debe ser 'embeddings' o 'similitudes'")

        # --- Validación de datos ---
        if len(np.unique(y)) < 2:
            print(f"⚠️ Saltando {csv_file.name}: solo hay una clase en los datos.")
            continue

        if len(y) < 10:
            print(f"⚠️ Saltando {csv_file.name}: muy pocos ejemplos ({len(y)} filas).")
            continue

        classifiers = {
            "MLP": MLPClassifier(hidden_layer_sizes=(128,), max_iter=200)
            ,"LogReg": LogisticRegression(max_iter=200)
            ,"RandomForest": RandomForestClassifier(n_estimators=100)
            ,"SVM": SVC(probability=True)
        }

        for name, clf in classifiers.items():
            f1_scores, auc_scores = [], []
            for seed in range(5):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, stratify=y, random_state=seed
                )
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

                f1 = f1_score(y_test, preds, average='macro')
                auc = roc_auc_score(y_test, probs) if probs is not None and len(np.unique(y)) == 2 else np.nan
                if probs is not None and len(np.unique(y)) == 2:
                    precision, recall, _ = precision_recall_curve(y_test, probs)
                    ap = average_precision_score(y_test, probs)
                else:
                    precision, recall, ap = None, None, np.nan

                f1_scores.append(f1)
                auc_scores.append(auc)

            resultados.append({
                'dataset': csv_file.stem,
                'modelo': name,
                'modo': modo,
                'F1_mean': np.mean(f1_scores),
                'F1_std': np.std(f1_scores),
                'AUC_mean': np.nanmean(auc_scores),
                'AUC_std': np.nanstd(auc_scores),
                'AP_mean': np.nanmean([ap])
            })

            # --- Curva Precision–Recall ---
            if name == "LogReg" and precision is not None:
                plt.figure(figsize=(6, 4))
                plt.plot(recall, precision, label=f'AP = {ap:.3f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Curva PR: {csv_file.stem} ({name}, {modo})')
                plt.legend()
                plt.tight_layout()

                plot_path = output_plots / f"PR_{csv_file.stem}_{name}_{modo}.png"
                plt.savefig(plot_path, dpi=300)
                print(f"📊 Guardada: {plot_path}")
                plt.close()

    # --- Resultados promedio ---
    df_res = pd.DataFrame(resultados)
    print("\n=== Resultados promedio ===")
    print(df_res)

    if df_res.empty:
        print("⚠️ No se generaron resultados válidos.")
        return df_res

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_res, x='modelo', y='F1_mean', hue='dataset')
    plt.title(f"F1-score promedio por modelo ({modo})")
    plt.ylabel("F1 promedio (5 runs)")
    plt.tight_layout()

    summary_path = output_plots / f"F1_resumen_{modo}.png"
    plt.savefig(summary_path, dpi=300)
    print(f"📊 Guardada: {summary_path}")
    plt.close()

    return df_res


# -----------------------------------------------------
#  EJECUCIÓN PRINCIPAL
# -----------------------------------------------------

if __name__ == "__main__":
    hacer_embeddings = True
    # Paso 1: procesar archivos JSON → CSVs con mensajes y etiquetas
    procesar_mentalrisk_arrays("MentalRiskEs2023", "output")
    if hacer_embeddings:
    # Paso 2: generar embeddings de los CSVs creados
    #generar_embeddings_csv("output", "output_tokenized", model_name="roberta-base", pooling="cls")
        generar_similitudes_semanticas("output", "output_tokenized/similitudes_baseline.csv", model_name="roberta-base", pooling="cls")
    
    # Paso 3: evaluar baseline
    resultados = evaluar_baseline("output_tokenized/similitudes_baseline.csv", modo="similitudes", output_plots="plots")
