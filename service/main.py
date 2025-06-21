from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from service.sentiment_service import analyze_document

app = FastAPI()


class DocumentRequest(BaseModel):
    text: str


@app.post("/analyze")
def analyze(req: DocumentRequest):
    try:
        result = analyze_document(req.text)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Sentiment API is running. Use /analyze or /docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service.main:app", host="127.0.0.1", port=8080, reload=True)
