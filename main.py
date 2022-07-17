from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from static.utils import Model, decode_image, encode_image_to_base64

VERSION = "1.0.0"

class Image(BaseModel):
    imageData: str


STATIC_PATH = "static"

origins = [
    "http://localhost:6601",
]

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return JSONResponse({
        "statusText" : "Root Endpoint of YOLOv3 Inference API",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.get("/version")
async def get_version():
    return JSONResponse({
        "statusText" : "Version Fetch Successful",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.get("/infer")
async def get_version():
    return JSONResponse({
        "statusText" : "Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.get("/infer/tiny-yolo-v3")
async def get_tiny_yolo_v3_infer():
    return JSONResponse({
        "statusText" : "Tiny Yolo V3 Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.post("/infer/tiny-yolo-v3")
async def post_tiny_yolo_v3_infer(image: Image):
    _, image = decode_image(image.imageData)

    model = Model(model_name="tiny-yolo-v3")
    model.setup()
    label, score, box = model.infer(image=image)

    if label is not None:
        return JSONResponse({
            "statusText" : "Tiny Yolo V3 Inference Inference Complete",
            "statusCode" : 200,
            "label" : label,
            "score" : str(score),
            "box" : box,
        })
    else:
        return JSONResponse({
            "statusText" : "Inference Failed",
            "statusCode" : 500,
        })
