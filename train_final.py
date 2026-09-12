from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "data.yaml"
RESULTS_PATH = BASE_DIR / "results"


def main():
    print("=" * 60)
    print("TESLA PLATE AI - TREINAMENTO FINAL")
    print("=" * 60)

    print(f"Dataset encontrado em: {DATASET_PATH}")
    print("Carregando modelo YOLO11n pré-treinado...")

    model = YOLO("yolo11n.pt")

    print("Modelo carregado.")
    print("Iniciando treinamento...")

    model.train(
        data=str(DATASET_PATH),
        epochs=30,
        imgsz=640,
        batch=4,
        device="cpu",
        project=str(RESULTS_PATH),
        name="tesla_plate_final",
        patience=10,
        workers=2,
        save=True,
        plots=True,
        verbose=True,
        seed=42,
    )

    print()
    print("=" * 60)
    print("TREINAMENTO FINAL CONCLUÍDO")
    print("=" * 60)


if __name__ == "__main__":
    main()