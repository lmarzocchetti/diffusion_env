from pathlib import Path

def main():
    input_dir = "/home/rhohen/Workspace/diffusion-env/data_precomputed_npy_pq"
    folder_path = Path(input_dir)
    files = [f.name for f in folder_path.iterdir() if f.is_file()]

    f = open(f"{folder_path}/metadata.jsonl", "w")

    for file in files:
        f.write("{\"file_name\": \""+f"{file}\", \"additional_feature\": \"Outdoor Environment Map.\""+"}\n")

    f.close()


if __name__ == "__main__":
    main()