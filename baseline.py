import pandas as pd
import json
from pathlib import Path

def procesar_mentalrisk_arrays(root_folder, output_folder):
    root_folder = Path(root_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Listar condiciones (TCA, Depresion, Ansiedad)
    condiciones = [d for d in root_folder.iterdir() if d.is_dir()]

    for cond in condiciones:
        print(f"Procesando condición: {cond.name}")
        def leer_subcarpeta(subfolder):
            subfolder = Path(subfolder)
            gold_file = subfolder / 'Gold' / 'gold_a.txt'

            if not gold_file.exists():
                print(f"Aviso: no existe gold_a.txt en {subfolder}, se ignorará")
                return pd.DataFrame(columns=['sujeto', 'mensajes', 'sufre'])

            # Indicamos que no hay cabecera y asignamos nombres
            gold = pd.read_csv(gold_file, header=None, names=['subject', 'label'])

            data = []
            if not subfolder.exists():
                return pd.DataFrame(columns=['sujeto', 'mensajes', 'sufre'])

            for json_file in subfolder.glob('*.json'):
                with open(json_file, 'r', encoding='utf-8') as f:
                    mensajes_json = json.load(f)
                mensajes_list = [m['message'] for m in mensajes_json]
                data.append({'sujeto': json_file.stem, 'mensajes': mensajes_list})

            df = pd.DataFrame(data)
            df = pd.merge(df, gold, left_on='sujeto', right_on='subject', how='left')
            df.rename(columns={'label': 'sufre'}, inplace=True)
            return df[['sujeto', 'mensajes', 'sufre']]

        # Leer Train y Test
        df_train = leer_subcarpeta(cond / 'Train')
        df_test = leer_subcarpeta(cond / 'Test')

        df_final = pd.concat([df_train, df_test], ignore_index=True)
        if df_final.empty:
            print(f"Aviso: no hay datos para la condición {cond.name}, se ignorará")
            continue

        output_file = output_folder / f"{cond.name}.csv"
        # Convertir listas a string para guardarlas en CSV
        df_final['mensajes'] = df_final['mensajes'].apply(lambda x: str(x))
        df_final.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Archivo creado: {output_file}")

# Uso:
procesar_mentalrisk_arrays("MentalRiskEs2023", "output")
