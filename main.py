from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from static.utils import YoloV3, YoloV6, decode_image, encode_image_to_base64

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
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/version")
async def get_version():
    return JSONResponse({
        "statusText" : "Version Fetch Successful",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/infer")
async def get_version():
    return JSONResponse({
        "statusText" : "Inference Endpoint",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.get("/infer/v{number}/tiny")
async def get_tiny_yolo_infer(number: int):
    return JSONResponse({
        "statusText" : f"Tiny Yolo V{number} Inference Endpoint; V3 is the only supported model type at present",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.post("/infer/v{number}/tiny")
async def post_tiny_yolo_infer(number: int, image: Image):
    _, image = decode_image(image.imageData)

    if number == 3:
        model = YoloV3(model_type="tiny")
    elif number == 6:
        model = YoloV6(model_type="tiny")
    else:
        return JSONResponse({
            "statusText" : "Tiny Yolo V3 and V6 is the only supported model type at present",
            "statusCode" : status.HTTP_500_INTERNAL_SERVER_ERROR,
        })
    
    model.setup()
    label, score, box = model.infer(image=image)

    if label is not None:
        return JSONResponse({
            "statusText" : f"Tiny Yolo V{number} Inference Inference Complete",
            "statusCode" : status.HTTP_200_OK,
            "label" : label,
            "score" : str(score),
            "box" : box,
        })
    else:
        return JSONResponse({
            "statusText" : "Inference Failed",
            "statusCode" : status.HTTP_500_INTERNAL_SERVER_ERROR,
        })
    

@app.get("/infer/v{number}/small")
async def get_small_yolo_infer(number: int):
    return JSONResponse({
        "statusText" : f"Small Yolo V{number} Inference Endpoint; V6 is the only supported model type at present",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.post("/infer/v{number}/small")
async def post_small_yolo_infer(number: int, image: Image):
    _, image = decode_image(image.imageData)

    if number == 6:
        model = YoloV6(model_type="small")
    else:
        return JSONResponse({
            "statusText" : "Small Yolo V6 is the only supported model type at present",
            "statusCode" : status.HTTP_500_INTERNAL_SERVER_ERROR,
        })
    
    model.setup()
    label, score, box = model.infer(image=image)

    if label is not None:
        return JSONResponse({
            "statusText" : f"Small Yolo V{number} Inference Inference Complete",
            "statusCode" : status.HTTP_200_OK,
            "label" : label,
            "score" : str(score),
            "box" : box,
        })
    else:
        return JSONResponse({
            "statusText" : "Inference Failed",
            "statusCode" : status.HTTP_500_INTERNAL_SERVER_ERROR,
        })


@app.get("/infer/v{number}/nano")
async def get_nano_yolo_infer(number: int):
    return JSONResponse({
        "statusText" : f"Nano Yolo V{number} Inference Endpoint; V6 is the only supported model type at present",
        "statusCode" : status.HTTP_200_OK,
        "version" : VERSION,
    })


@app.post("/infer/v{number}/nano")
async def post_nano_yolo_infer(number: int, image: Image):
    _, image = decode_image(image.imageData)

    if number == 6:
        model = YoloV6(model_type="nano")
    else:
        return JSONResponse({
            "statusText" : "Nano Yolo V6 is the only supported model type at present",
            "statusCode" : status.HTTP_500_INTERNAL_SERVER_ERROR,
        })
    
    model.setup()
    label, score, box = model.infer(image=image)

    if label is not None:
        return JSONResponse({
            "statusText" : f"Nano Yolo V{number} Inference Inference Complete",
            "statusCode" : status.HTTP_200_OK,
            "label" : label,
            "score" : str(score),
            "box" : box,
        })
    else:
        return JSONResponse({
            "statusText" : "Inference Failed",
            "statusCode" : status.HTTP_500_INTERNAL_SERVER_ERROR,
        })

