### **YOLO Inference API**

<br>

1. Install Python
2. Run `pip install virtualenv`
3. Run `make-env.bat` or `make-env-3.9.bat`
4. Run `start-api-server.bat` (or setup `.vscode` or use `docker`).
5. The API will now be served at `http://127.0.0.1:6600`


<br>

**Endpoints**

`/infer/tiny-yolo-v3` - returns label, score and bounding box coordinate of the highest confidence box

<br>

*Notes: Implement for other YOLO Architectures as well*
