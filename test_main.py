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


def test_get_tiny_yolo_v3_infer():
    response = client.get("/infer/tiny-yolo-v3")
    assert response.status_code == 200
    assert response.json() == {
        "statusText" : "Tiny Yolo V3 Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    }
