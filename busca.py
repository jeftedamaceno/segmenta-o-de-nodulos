import os

def listar_arquivos_e_pastas(caminho):
    for raiz, pastas, arquivos in os.walk(caminho):
        print(f"\n📂 Pasta atual: {raiz}")

        # Lista as subpastas
        for pasta in pastas:
            print(f"   📁 Pasta: {pasta}")

        # Lista os arquivos
        for arquivo in arquivos:
            print(f"   📄 Arquivo: {arquivo}")

# Caminho da pasta que deseja analisar
caminho_pasta = r"C:\LIDC\pacientes\pacientes_estado_avançado"

listar_arquivos_e_pastas(caminho_pasta)