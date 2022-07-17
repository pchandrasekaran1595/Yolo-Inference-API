import pytest

from fastapi.testclient import TestClient
from main import app, VERSION


client = TestClient(app=app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Root Endpoint of YOLOv3 Inference API",
        "statusCode" : 200,
        "version" : VERSION,
    }
 

def test_get_version():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Version Fetch Successful",
        "statusCode" : 200,
        "version" : VERSION,
    }
    
      
def test_get_infer():
    response = client.get("/infer")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }


@pytest.mark.parametrize(
    "number", 
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
)
def test_get_tiny_yolo_infer(number):
    response = client.get(f"/infer/v{number}/tiny")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : f"Tiny Yolo V{number} Inference Endpoint; V3 is the only supported model type at present",
        "statusCode" : 200,
        "version" : VERSION,
    }


@pytest.mark.parametrize(
    "number", 
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
)
def test_get_nano_yolo_infer(number):
    response = client.get(f"/infer/v{number}/nano")
    print(response.json())
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : f"Nano Yolo V{number} Inference Endpoint; V6 is the only supported model type at present",
        "statusCode" : 200,
        "version" : VERSION,
    }
