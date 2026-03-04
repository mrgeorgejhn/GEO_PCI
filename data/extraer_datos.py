import pandas as pd
import os

def procesar_pci():
    # 1. Definir la ruta de destino según tu solicitud
    folder_data = r'C:\Users\jhnav\OneDrive\Documentos\GEO_PCI\data'
    
    # Crear la carpeta si no existe
    if not os.path.exists(folder_data):
        os.makedirs(folder_data)
        print(f"✅ Carpeta creada: {folder_data}")

    # 2. Archivos a procesar
    archivos = {
        "VD FLEXIBLE.xlsx": "vd_flexible",
        "VD RIGIDO.xlsx": "vd_rigido",
        "CORRECCION DV.xlsx": "correccion"
    }

    for excel_name, prefix in archivos.items():
        if not os.path.exists(excel_name):
            print(f"❌ No se encontró: {excel_name}")
            continue

        print(f"📖 Leyendo {excel_name}...")
        xl = pd.ExcelFile(excel_name)
        
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            
            # Normalizar nombre de la hoja para que el buscador del app.py lo encuentre
            # Ejemplo: "Piel de Cocodrilo" -> "piel_cocodrilo"
            sheet_norm = sheet.lower().strip().replace(" ", "_")
            
            # Nombre final: vd_flexible_piel_cocodrilo.csv
            csv_name = f"{prefix}_{sheet_norm}.csv"
            ruta_csv = os.path.join(folder_data, csv_name)
            
            # Guardar
            df.to_csv(ruta_csv, index=False)
            print(f"   -> Generado: {csv_name}")

    print("\n🚀 ¡Listo! Todos los archivos están en la carpeta /data.")
    print("Ahora sube esa carpeta a tu repositorio de GitHub.")

if __name__ == "__main__":
    procesar_pci()