from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET", "dummy-bucket")
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

# Cho phep chay cuc bo test
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
if not os.path.exists(MODEL_PATH):
    if os.path.exists("models/model.pkl"):
        import shutil
        shutil.copy("models/model.pkl", MODEL_PATH)

def download_model():
    """
    Tai file model.pkl tu AWS S3 ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import. Su dung
    AWS_ACCESS_KEY_ID va AWS_SECRET_ACCESS_KEY de xac thuc (duoc dat trong systemd service).
    """
    if "AWS_ACCESS_KEY_ID" not in os.environ:
        print("Bo qua download_model do khong co AWS credentials.")
        return

    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(AWS_S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
        print("Model da duoc tai xuong tu S3.")
    except Exception as e:
        print(f"Loi tai model tu S3: {e}")

download_model()
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Chua the load model: {e}")
    model = None


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    pred = model.predict([req.features])[0]

    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    label = label_map.get(int(pred), "unknown")

    return {"prediction": int(pred), "label": label}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
