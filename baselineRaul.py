import pandas as pd
import json
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score, precision_score, recall_score, roc_curve, \
    auc, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
import seaborn as sns
import matplotlib
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from scipy.spatial.distance import cosine
import os
import joblib
from sklearn.model_selection import GridSearchCV
from collections import Counter
import random
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

nltk.download("wordnet")
nltk.download("stopwords")
spanish_stopwords = stopwords.words("spanish")


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
        df["mensajes"] = df["mensajes"].apply(ast.literal_eval)
        texts = [" ".join(msgs) for msgs in df["mensajes"]]
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
#  FUNCION DE AUGMENTACION
# -----------------------------------------------------

def synonym_replacement(text, n=100):
    words = text.split()
    new_words = words.copy()
    random_idx = list(range(len(words)))
    random.shuffle(random_idx)
    num_replaced = 0

    for idx in random_idx:
        word = words[idx]
        if word.lower() in spanish_stopwords:
            continue
        synonyms = set()
        for syn in wordnet.synsets(word, lang='spa'):
            for lemma in syn.lemmas(lang='spa'):
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        if synonyms:
            new_words[idx] = random.choice(list(synonyms))
            num_replaced += 1
        if num_replaced >= n:
            break
    return " ".join(new_words)

# -----------------------------------------------------
#  EXPERIMENTO BALANCEO
# -----------------------------------------------------

def experimento_balanceo(train_csv, test_csv=None, output_dir="output_balanceo", seed=42):
    os.makedirs(output_dir, exist_ok=True)
    train_name = Path(train_csv).stem
    print(f"\n=== EXPERIMENTO BALANCEO ===\nEntrenando y Evaluando {train_name}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ROBERTA_MODEL = "pysentimiento/robertuito-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    model_cls = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL).to(device)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    predef_msg = "esto es lo peor que me ha pasado, tengo depresion, y no puedo dejar de comer comida poco sana"
    predef_emb = embedder.encode(predef_msg, convert_to_numpy=True)

    # --- Leer dataset ---
    df = pd.read_csv(train_csv)
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

    # --- Funciones de features ---
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

    print("Generando features RoBERTuito...")
    X_probs = get_sentiment_probs(texts)
    X_embs, X_dists = get_embeddings_and_dists(texts)
    X = np.hstack([X_embs, X_probs, X_dists])

    # --- Split train/test interno incluyendo los textos ---
    X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
        X, y, texts, test_size=0.2, stratify=y, random_state=seed
    )
    print(f"Train: {len(X_train)} ejemplos | Test: {len(X_test)} ejemplos\n")

    # --- Métodos de balanceo ---
    balance_methods = {
        "ManualAugmentation": None,  # Aquí se hará augmentación manual
        "SMOTE": SMOTE(random_state=seed),
        "RandomUnderSampler": RandomUnderSampler(random_state=seed)
    }

    results = []

    modelos = {
        "MLP": MLPClassifier(random_state=seed, max_iter=500, hidden_layer_sizes=(128, 64)),
        "LogisticRegression": LogisticRegression(max_iter=2000)
    }

    for method_name, sampler in balance_methods.items():
        print(f"\n=== Balanceo: {method_name} ===")

        if method_name == "ManualAugmentation":
            # Detectar clase minoritaria
            counts = Counter(y_train)
            minor_class = min(counts, key=counts.get)
            major_class = max(counts, key=counts.get)

            # Obtener textos de la clase minoritaria
            minor_texts = [t for t, label in zip(texts_train, y_train) if label == minor_class]

            synthetic_texts = []
            max_synthetic = counts[major_class] - counts[minor_class]  # generar solo los necesarios

            # Generar frases sintéticas hasta balancear
            while len(synthetic_texts) < max_synthetic:
                for t in minor_texts:
                    if len(synthetic_texts) >= max_synthetic:
                        break
                    # augmentación con sinónimos
                    new_text = synonym_replacement(t, n=100)
                    if new_text not in synthetic_texts:
                        synthetic_texts.append(new_text)

            # Combinar textos originales + sintéticos
            X_train_aug_texts = texts_train + synthetic_texts
            y_train_aug = np.concatenate([y_train, np.array([minor_class] * len(synthetic_texts))])

            # Recalcular features
            if synthetic_texts:
                X_probs_aug = get_sentiment_probs(X_train_aug_texts)
                X_embs_aug, X_dists_aug = get_embeddings_and_dists(X_train_aug_texts)
                X_bal = np.hstack([X_embs_aug, X_probs_aug, X_dists_aug])
                y_bal = y_train_aug
            else:
                # No hay frases nuevas: usar directamente los datos originales
                X_bal = X_train
                y_bal = y_train
        else:
            X_bal, y_bal = sampler.fit_resample(X_train, y_train)

        for model_name, clf in modelos.items():
            print(f"\nEntrenando modelo: {model_name} con {method_name}")

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf)
            ])

            pipeline.fit(X_bal, y_bal)
            y_pred = pipeline.predict(X_test)

            f1 = f1_score(y_test, y_pred, average="macro")
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro")
            rec = recall_score(y_test, y_pred, average="macro")

            print(f"[{model_name}] F1={f1:.4f} | Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f}")

            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicho")
            plt.ylabel("Real")
            plt.title(f"{method_name} - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"conf_matrix_{train_name}_{method_name}_{model_name}.png"), dpi=300)
            plt.close()

            results.append({
                "train_cond": train_name,
                "balanceo": method_name,
                "modelo": model_name,
                "f1": f1,
                "accuracy": acc,
                "precision": prec,
                "recall": rec
            })

    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, "resultados_balanceo.csv")
    if os.path.exists(results_file):
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_file, index=False)

    return results_df



if __name__ == "__main__":

    procesar_mentalrisk_arrays("MentalRiskEs2023", "output")

    baseline_robertuito("output/Depresión.csv", output_dir="output_robertuito")
    baseline_robertuito("output/Ansiedad.csv", output_dir="output_robertuito")
    baseline_robertuito("output/TCA.csv", output_dir="output_robertuito")

    experimento_balanceo("output/Depresión.csv", None, output_dir="output_balanceo")
    experimento_balanceo("output/Ansiedad.csv", None, output_dir="output_balanceo")
    experimento_balanceo("output/TCA.csv", None, output_dir="output_balanceo")

    df_base = pd.read_csv("resultados_baseline.csv")
    df_balance = pd.read_csv("output_balanceo/resultados_balanceo.csv")

    # Renombramos f1_macro → f1
    df_base = df_base.rename(columns={"f1_macro": "f1"})

    # Filtro: baseline solo tiene MLPClassifier
    df_base = df_base[df_base["modelo"] == "MLPClassifier"]
    df_base["modelo"] = "MLP"  # normalizamos nombre

    df_base["balanceo"] = "SinBalanceo"
    df_base = df_base[["condicion", "balanceo", "modelo", "f1"]]

    # ================================
    # FILTRAR BALANCEO SOLO MLP
    # ================================

    df_balance = df_balance.rename(columns={"train_cond": "condicion"})
    df_balance_mlp = df_balance[df_balance["modelo"] == "MLP"]

    df_balance_mlp = df_balance_mlp[["condicion", "balanceo", "modelo", "f1"]]

    # ================================
    # UNIR BASELINE + BALANCEO MLP
    # ================================

    df_all = pd.concat([df_base, df_balance_mlp], ignore_index=True)

    # ================================
    # GRÁFICO 1: SOLO MLP
    # ================================

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=df_all,
        x="condicion",
        y="f1",
        hue="balanceo",
        errorbar=None,
        palette="Set2"
    )

    plt.title("Comparación de F1: SinBalanceo vs ManualAug vs SMOTE vs UnderSampling (Solo MLP)")
    plt.ylabel("F1 Macro")
    plt.xlabel("Condición")

    plt.tight_layout()
    output_plot = os.path.join("output_balanceo", "comparacion_balanceo_solo_mlp.png")
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"\n📊 Gráfico creado: {output_plot}")

    # ================================
    # GRÁFICO 2 — MLP vs LOGISTIC (con ManualAug + SMOTE + UnderSampling)
    # ================================

    df_plot = df_balance[["condicion", "balanceo", "modelo", "f1"]]

    g = sns.catplot(
        data=df_plot,
        kind="bar",
        x="condicion",
        y="f1",
        hue="modelo",
        col="balanceo",
        col_order=["ManualAugmentation", "SMOTE", "RandomUnderSampler"],
        palette="Set2",
        height=5,
        aspect=1,
        errorbar=None
    )

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Comparación MLP vs Logistic para Cada Método de Balanceo")

    out_grid = os.path.join("output_balanceo", "comparacion_modelos_por_balanceo.png")
    g.savefig(out_grid, dpi=300)
    plt.close()

    print(f"\n📊 Gráfico creado: {out_grid}")