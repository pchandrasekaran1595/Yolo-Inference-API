from fastapi import FastAPI, status
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, FileResponse

from static.utils import YoloV3, YoloV6, YoloV7, YoloV8, decode_image

VERSION: str = "1.0.0"
STATIC_PATH: str = "static"


class Image(BaseModel):
    imageData: str


origins = [
    "http://localhost:5051",
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(f"{STATIC_PATH}/media/favicon.ico")


@app.get("/")
async def root():
    return JSONResponse(
        {
            "statusText": "Root Endpoint of YOLO Inference API",
            "statusCode": status.HTTP_200_OK,
            "version": VERSION,
        }
    )


@app.get("/version")
async def get_version():
    return JSONResponse(
        {
            "statusText": "Version Fetch Successful",
            "statusCode": status.HTTP_200_OK,
            "version": VERSION,
        }
    )


@app.get("/infer")
async def get_version():
    return JSONResponse(
        {
            "statusText": "Base Inference Endpoint",
            "statusCode": status.HTTP_200_OK,
            "version": VERSION,
        }
    )


@app.get("/infer/v{number}/small")
async def get_small_yolo_infer(number: int):
    return JSONResponse(
        {
            "statusText": f"Small Yolo V{number} Inference Endpoint; Supported Models [V6, V8]",
            "statusCode": status.HTTP_200_OK,
            "version": VERSION,
        }
    )


@app.post("/infer/v{number}/small")
async def post_small_yolo_infer(number: int, image: Image):
    _, image = decode_image(image.imageData)

    if number == 6:
        model = YoloV6(model_type="small")
    elif number == 8:
        model = YoloV8(model_type="small")
    else:
        return JSONResponse(
            {
                "statusText": "Supported Models [V6, V8]",
                "statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR,
            }
        )

    label, score, box = model.infer(image=image)

    if label is not None:
        return JSONResponse(
            {
                "statusText": f"Small Yolo V{number} Inference Inference Complete",
                "statusCode": status.HTTP_200_OK,
                "label": label,
                "score": str(score),
                "box": box,
            }
        )
    else:
        return JSONResponse(
            {
                "statusText": "Inference Failed",
                "statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR,
            }
        )


@app.get("/infer/v{number}/tiny")
async def get_tiny_yolo_infer(number: int):
    return JSONResponse(
        {
            "statusText": f"Tiny Yolo V{number} Inference Endpoint; Supported Models [V3, V6, V7]",
            "statusCode": status.HTTP_200_OK,
            "version": VERSION,
        }
    )


@app.post("/infer/v{number}/tiny")
async def post_tiny_yolo_infer(number: int, image: Image):
    _, image = decode_image(image.imageData)

    if number == 3:
        model = YoloV3(model_type="tiny")
    elif number == 6:
        model = YoloV6(model_type="tiny")
    elif number == 7:
        model = YoloV7(model_type="tiny")
    else:
        return JSONResponse(
            {
                "statusText": "Supported Models [V3, V6, V7]",
                "statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR,
            }
        )

    label, score, box = model.infer(image=image)

    if label is not None:
        return JSONResponse(
            {
                "statusText": f"Tiny Yolo V{number} Inference Inference Complete",
                "statusCode": status.HTTP_200_OK,
                "label": label,
                "score": str(score),
                "box": box,
            }
        )
    else:
        return JSONResponse(
            {
                "statusText": "Inference Failed",
                "statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR,
            }
        )


@app.get("/infer/v{number}/nano")
async def get_nano_yolo_infer(number: int):
    return JSONResponse(
        {
            "statusText": f"Nano Yolo V{number} Inference Endpoint; Supported Models [V6, V8]",
            "statusCode": status.HTTP_200_OK,
            "version": VERSION,
        }
    )


@app.post("/infer/v{number}/nano")
async def post_nano_yolo_infer(number: int, image: Image):
    _, image = decode_image(image.imageData)

    if number == 6:
        model = YoloV6(model_type="nano")
    elif number == 8:
        model = YoloV8(model_type="nano")
    else:
        return JSONResponse(
            {
                "statusText": "Supported Models [V6, V8]",
                "statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR,
            }
        )

    label, score, box = model.infer(image=image)

    if label is not None:
        return JSONResponse(
            {
                "statusText": f"Nano Yolo V{number} Inference Inference Complete",
                "statusCode": status.HTTP_200_OK,
                "label": label,
                "score": str(score),
                "box": box,
            }
        )
    else:
        return JSONResponse(
            {
                "statusText": "Inference Failed",
                "statusCode": status.HTTP_500_INTERNAL_SERVER_ERROR,
            }
        )
