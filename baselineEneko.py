
import pandas as pd
import json
from pathlib import Path
import torch
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score ,precision_recall_curve,accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 
from tqdm import tqdm
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from scipy.spatial.distance import cosine
import os
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")

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
# -----------------------------------------------------
#  EXPERIMENTO: ANÁLISIS DE TRANSFERIBILIDAD (CROSS-CONDITION)
# -----------------------------------------------------



def cross_condition_experiment(train_csv, test_csv, output_dir="output_cross", seed=42, representation="robertuito"):
    """
    Entrena y evalúa un modelo cross-condition entre dos datasets (ej. Depresión→Ansiedad).
    Puede usar representaciones "roberta" o "tfidf".
    Muestra barras de progreso y guarda resultados + gráficas.
    """

    os.makedirs(output_dir, exist_ok=True)

    train_name = Path(train_csv).stem
    test_name = Path(test_csv).stem

    print(f"\n=== CROSS-CONDITION ({representation.upper()}) ===\nEntrenando en {train_name} → Evaluando en {test_name}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Leer datasets ---
    def load_dataset(path):
        df = pd.read_csv(path)
        if "mensajes" in df.columns:
            df["mensajes"] = df["mensajes"].apply(ast.literal_eval)
            texts = [" ".join(msgs) for msgs in df["mensajes"]]
        elif "text" in df.columns:
            texts = df["text"].astype(str).tolist()
        else:
            raise ValueError("El CSV debe contener 'mensajes' o 'text'")
        if "sufre" in df.columns:
            y = df["sufre"].astype(int).to_numpy()
        elif "label" in df.columns:
            y = df["label"].astype(int).to_numpy()
        else:
            raise ValueError("No se encontró ninguna columna de etiquetas.")
        return texts, y

    X_train_texts, y_train = load_dataset(train_csv)
    X_test_texts, y_test = load_dataset(test_csv)

    print(f"Train: {len(X_train_texts)} ejemplos | Test: {len(X_test_texts)} ejemplos\n")

    # --- Representaciones ---
    if representation == "robertuito":

        ROBERTA_MODEL = "pysentimiento/robertuito-sentiment-analysis"
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
        model_cls = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL).to(device)
        embedder = SentenceTransformer(ROBERTA_MODEL, device=device)

        predef_msg = "esto es lo peor que me ha pasado, tengo depresion, y no puedo dejar de comer comida poco sana"
        predef_emb = embedder.encode(predef_msg, convert_to_numpy=True)

        def get_sentiment_probs(texts, batch_size=16):
            model_cls.eval()
            probs = []
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size), desc="Probabilidades", ncols=80):
                    batch = texts[i:i+batch_size]
                    tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                    out = model_cls(**tokens)
                    logits = out.logits
                    batch_probs = F.softmax(logits, dim=-1).cpu().numpy()
                    probs.append(batch_probs)
            return np.vstack(probs)

        def get_embeddings_and_dists(texts, batch_size=32):
            all_embs = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings", ncols=80):
                batch = texts[i:i+batch_size]
                embs = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                all_embs.append(embs)
            embs = np.vstack(all_embs)
            dists = np.array([cosine(e, predef_emb) for e in embs]).reshape(-1, 1)
            return embs, dists

        print("(-) Generando features RoBERTuito...")
        t0 = time()
        X_train_probs = get_sentiment_probs(X_train_texts)
        X_train_embs, X_train_dists = get_embeddings_and_dists(X_train_texts)
        X_train = np.hstack([X_train_embs, X_train_probs, X_train_dists])

        X_test_probs = get_sentiment_probs(X_test_texts)
        X_test_embs, X_test_dists = get_embeddings_and_dists(X_test_texts)
        X_test = np.hstack([X_test_embs, X_test_probs, X_test_dists])
        print(f"Features RoBERTuito generadas en {round((time()-t0)/60,2)} min")

    elif representation == "tfidf":
        print("(-) Generando features TF-IDF...")
        t0 = time()
        vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), stop_words=spanish_stopwords
        )
        X_train = vectorizer.fit_transform(X_train_texts)
        X_test = vectorizer.transform(X_test_texts)
        print(f"Features TF-IDF generadas en {round((time()-t0)/60,2)} min")
    else:
        raise ValueError("representation debe ser 'roberta' o 'tfidf'.")

    # ============================================================
    # === MULTI-CLASIFICADORES + MULTI-RUN                     ===
    # ============================================================

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier

    N_RUNS = 5

    classifiers = {
        "MLP": MLPClassifier(
            random_state=seed,
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            learning_rate_init=0.001
        ),
        "LogReg": LogisticRegression(max_iter=2000),
        "SVM": SVC(probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=seed)
    }

    results = []
    best_clf_name = None
    best_f1 = -1
    best_model = None

    for clf_name, clf in classifiers.items():
        print(f"\n=== Entrenando {clf_name} ({N_RUNS} runs) ===")

        metrics_runs = {"f1": [], "acc": [], "prec": [], "rec": []}

        for run in range(N_RUNS):
            print(f"  → Run {run+1}/{N_RUNS}")

            X_train_i, X_val, y_train_i, y_val = train_test_split(
                X_train, y_train,
                test_size=0.20,
                random_state=seed + run,
                stratify=y_train
            )

            pipeline = Pipeline([
                ("scaler", StandardScaler(with_mean=False) if representation=="tfidf" else StandardScaler()),
                ("clf", clf)
            ])

            pipeline.fit(X_train_i, y_train_i)

            y_pred = pipeline.predict(X_test)

            f1 = f1_score(y_test, y_pred, average="macro")
            metrics_runs["f1"].append(f1)
            metrics_runs["acc"].append(accuracy_score(y_test, y_pred))
            metrics_runs["prec"].append(precision_score(y_test, y_pred, average="macro"))
            metrics_runs["rec"].append(recall_score(y_test, y_pred, average="macro"))

            # Guardar mejor modelo → para la gráfica
            if f1 > best_f1:
                best_f1 = f1
                best_clf_name = clf_name
                best_model = pipeline
                best_pred = y_pred

        results.append({
            "train_cond": train_name,
            "test_cond": test_name,
            "representation": representation,
            "classifier": clf_name,
            "f1_mean": np.mean(metrics_runs["f1"]),
            "f1_std": np.std(metrics_runs["f1"]),
            "acc_mean": np.mean(metrics_runs["acc"]),
            "acc_std": np.std(metrics_runs["acc"]),
            "prec_mean": np.mean(metrics_runs["prec"]),
            "prec_std": np.std(metrics_runs["prec"]),
            "rec_mean": np.mean(metrics_runs["rec"]),
            "rec_std": np.std(metrics_runs["rec"])
        })

    # ============================================================
    # === BASELINE
    # ============================================================

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    results.append({
        "train_cond": train_name,
        "test_cond": test_name,
        "representation": representation,
        "classifier": "Dummy",
        "f1_mean": f1_score(y_test, dummy_pred, average="macro"),
        "f1_std": 0,
        "acc_mean": accuracy_score(y_test, dummy_pred),
        "acc_std": 0,
        "prec_mean": precision_score(y_test, dummy_pred, average="macro"),
        "prec_std": 0,
        "rec_mean": recall_score(y_test, dummy_pred, average="macro"),
        "rec_std": 0,
    })

    # ============================================================
    # === MATRIZ DE CONFUSIÓN (del mejor clasificador)         ===
    # ============================================================

    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"{best_clf_name} ({representation.upper()})\n{train_name}→{test_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"conf_matrix_{representation}_{train_name}_to_{test_name}.png"), dpi=300)
    plt.close()

    # ============================================================
    # === GUARDAR RESULTADOS
    # ============================================================

    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, "cross_condition_results.csv")

    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, index=False)

    return results_df



# -----------------------------------------------------
#  EJEMPLO DE USO
# -----------------------------------------------------

if __name__ == "__main__":
    pairs = [
        ("output/Depresión.csv", "output/Ansiedad.csv"),
        ("output/Depresión.csv", "output/TCA.csv"),
        ("output/Ansiedad.csv", "output/Depresión.csv"),
        ("output/TCA.csv", "output/Depresión.csv"),
        ("output/Ansiedad.csv", "output/TCA.csv"),
        ("output/TCA.csv", "output/Ansiedad.csv")
    ]

    resultados = []
    for rep in ["robertuito", "tfidf"]:
        for train_csv, test_csv in pairs:
            res = cross_condition_experiment(train_csv, test_csv, representation=rep)
            resultados.append(res)

    df_all = pd.concat(resultados, ignore_index=True)
    print("\n=== RESULTADOS CROSS-CONDITION ===")
    print(df_all)

    # ---------------------------
    # PROMEDIOS CROSS-CONDITION
    # ---------------------------

    promedios = []

    for rep in ["robertuito", "tfidf"]:
        df_rep = df_all[df_all["representation"] == rep]

        # Filtrar por cada clasificador
        df_mlp    = df_rep[df_rep["classifier"] == "MLP"]
        df_logreg = df_rep[df_rep["classifier"] == "LogReg"]
        df_svm    = df_rep[df_rep["classifier"] == "SVM"]
        df_rf     = df_rep[df_rep["classifier"] == "RandomForest"]
        df_dummy  = df_rep[df_rep["classifier"] == "Dummy"]

        promedios.append({
            "representation": rep,

            # MLP
            "mlp_f1_macro": df_mlp["f1_mean"].mean(),
            "mlp_accuracy": df_mlp["acc_mean"].mean(),

            # Logistic Regression
            "logreg_f1_macro": df_logreg["f1_mean"].mean(),
            "logreg_accuracy": df_logreg["acc_mean"].mean(),

            # SVM
            "svm_f1_macro": df_svm["f1_mean"].mean(),
            "svm_accuracy": df_svm["acc_mean"].mean(),

            # Random Forest
            "rf_f1_macro": df_rf["f1_mean"].mean(),
            "rf_accuracy": df_rf["acc_mean"].mean(),

            # Dummy baseline
            "dummy_f1_macro": df_dummy["f1_mean"].mean(),
            "dummy_accuracy": df_dummy["acc_mean"].mean()
        })

    df_promedios = pd.DataFrame(promedios)
    print("\n=== PROMEDIOS CROSS-CONDITION POR CLASIFICADOR ===")
    print(df_promedios)

    df_promedios.to_csv("output_cross/cross_condition_promedios.csv", index=False)
    
    # --- Gráfica comparativa ---
    plt.figure(figsize=(6,4))
    sns.barplot(data=df_all, x="train_cond", y="f1_mean", hue="representation", errorbar=None)
    plt.title("Comparación TF-IDF vs RoBERTuito (Cross-condition)")
    plt.ylabel("F1 Macro")
    plt.xlabel("Condición de entrenamiento")
    plt.legend(title="Representación")
    plt.tight_layout()
    plt.savefig("output_cross/f1_comparison_tfidf_vs_robertuito.png", dpi=300)
    plt.close()


    # Heatmap F1 para RoBERTuito
# Agrupar para dejar una sola fila por combinación
    classifiers = df_all["classifier"].unique()

    for clf in classifiers:
        for rep in ["robertuito", "tfidf"]:

            df_sub = df_all[
                (df_all["representation"] == rep) &
                (df_all["classifier"] == clf)
            ].groupby(["train_cond", "test_cond"])["f1_mean"].mean().reset_index()

            pivot = df_sub.pivot(index="train_cond", columns="test_cond", values="f1_mean")

            plt.figure(figsize=(5,4))
            sns.heatmap(pivot, annot=True, cmap="mako" if rep=="robertuito" else "crest", fmt=".3f")
            plt.title(f"Transferibilidad {rep.upper()} – {clf} (F1 macro)")
            plt.tight_layout()
            plt.savefig(f"output_cross/heatmap_{rep}_{clf}.png", dpi=300)
            plt.close()

    # === Gráfica TRIPLE: RoBERTuito vs TF-IDF vs Dummy ===
    plot_rows = []

    for _, row in df_all.iterrows():
        # Representación normal (RoBERTuito / TFIDF)
        plot_rows.append({
            "train_cond": row["train_cond"],
            "test_cond": row["test_cond"],
            "representation": row["representation"],
            "f1": row["f1_mean"]
        })
        # Dummy, en su propia categoría
        if row["classifier"] == "Dummy":
            plot_rows.append({
                "train_cond": row["train_cond"],
                "test_cond": row["test_cond"],
                "representation": "Dummy",
                "f1": row["f1_mean"]
            })

    df_plot = pd.DataFrame(plot_rows)

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df_plot,
        x="train_cond",
        y="f1",
        hue="representation",
        errorbar=None
    )
    plt.title("Comparación RoBERTuito vs TF-IDF vs Baseline (Cross-condition)")
    plt.ylabel("F1 Macro")
    plt.xlabel("Condición de entrenamiento")
    plt.tight_layout()
    plt.savefig("output_cross/f1_comparison_triple.png", dpi=300)
    plt.close()


    # Gráfica
    plot_rows = []

    for _, row in df_all.iterrows():
        model_name = f"{row['representation'].upper()} + {row['classifier']}"
        plot_rows.append({
            "train_cond": row["train_cond"],
            "test_cond": row["test_cond"],
            "model": model_name,
            "f1": row["f1_mean"]
        })

    df_plot = pd.DataFrame(plot_rows)

    plt.figure(figsize=(12,6))
    sns.barplot(
        data=df_plot,
        x="train_cond",
        y="f1",
        hue="model",
        palette="Set2", 
        errorbar=None
    )
    plt.title("Comparativa Cross-Condition: Todos los modelos")
    plt.ylabel("F1 Macro")
    plt.xlabel("Condición de entrenamiento")
    plt.ylim(0,1)
    plt.legend(title="Modelo", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("output_cross/f1_comparison_all_models.png", dpi=300)
    plt.close()

    # === NUEVA GRÁFICA: Todos los modelos ===
    plot_rows = []

    for _, row in df_all.iterrows():
        model_name = f"{row['representation'].upper()} + {row['classifier']}"
        plot_rows.append({
            "train_cond": row["train_cond"],
            "test_cond": row["test_cond"],
            "model": model_name,
            "f1": row["f1_mean"]
        })

    df_plot = pd.DataFrame(plot_rows)

    plt.figure(figsize=(12,6))
    sns.barplot(
        data=df_plot,
        x="train_cond",
        y="f1",
        hue="model",
        palette="Set2",
        errorbar=None
    )
    plt.title("Comparativa Cross-Condition: Todos los modelos")
    plt.ylabel("F1 Macro")
    plt.xlabel("Condición de entrenamiento")
    plt.ylim(0,1)
    plt.legend(title="Modelo", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("output_cross/f1_comparison_all_models.png", dpi=300)
    plt.close()
