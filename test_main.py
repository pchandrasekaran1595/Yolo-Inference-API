import pytest

from fastapi.testclient import TestClient
from main import app, VERSION


client = TestClient(app=app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "statusText": "Root Endpoint of YOLO Inference API",
        "statusCode": 200,
        "version": VERSION,
    }


def test_get_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {
        "statusText": "Version Fetch Successful",
        "statusCode": 200,
        "version": VERSION,
    }


def test_get_base_infer():
    response = client.get("/infer")
    assert response.status_code == 200
    assert response.json() == {
        "statusText": "Base Inference Endpoint",
        "statusCode": 200,
        "version": VERSION,
    }


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_get_small_yolo_infer(number):
    response = client.get(f"/infer/v{number}/small")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {
        "statusText": f"Small Yolo V{number} Inference Endpoint; Supported Models [V6, V8]",
        "statusCode": 200,
        "version": VERSION,
    }


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_get_tiny_yolo_infer(number):
    response = client.get(f"/infer/v{number}/tiny")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {
        "statusText": f"Tiny Yolo V{number} Inference Endpoint; Supported Models [V3, V6, V7]",
        "statusCode": 200,
        "version": VERSION,
    }


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_get_nano_yolo_infer(number):
    response = client.get(f"/infer/v{number}/nano")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {
        "statusText": f"Nano Yolo V{number} Inference Endpoint; Supported Models [V6, V8]",
        "statusCode": 200,
        "version": VERSION,
    }
