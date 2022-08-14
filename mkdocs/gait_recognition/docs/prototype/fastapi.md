# Backend Server
The prototype requires 2 servers to be run at the same time. The backend server is run on Uvicorn and it uses FastAPI to run the model for prediction.

## How To Run

1. clone this repository
    ```bash
    git clone https://github.com/KLASS-gait-recognitionn/GaitSearch
    ```
2. change directory to the backend folder 
    ```bash
    cd backend
    ```

3. Install relevant dependencies
    ```bash
    pip install -r requirements.txt
    ```

4. run the server  
    ```bash
    uvicorn main:app 
    ```

## Endpoints 

#### Special requests

1. `preflight_handler` handles preflight requests sent by the browser before the actual request. The preflight gives the server a chance to examine what the actual request will look like before it's made.
    ```python
    @app.options('/{rest_of_path:path}')
    async def preflight_handler
    ```

2. `add_CORS_header` enables CORS (Cross-Origin Resource Sharing) to occur between the backend server and the frontend server in the browser
    ```python
    @app.middleware("http")
    async def add_CORS_header(request: Request, call_next)
    ```

#### POST request
1. `upload` endpoint posts a file added through the React server to the Backend server. File is converted to .mp4 regardless of filetype.
    ```python
    @app.post("/upload", response_description="", response_model = "")
    async def upload(file:UploadFile = File(...)):
    ```

#### GET requests
1. `loading` endpoint runs the silhouette extractor in the background when this endpoint is called. The silhoeutte extraction occurs on the video that was uploaded. At the same time, the video is split into a set of images to feed into the model.
    ```python
    @app.get("/loading")
    async def loading():
    ```
2. `video` endpoint streams the original video with a bounding box attached to the subject to focus on.
    ```python
    @app.get("/video")
    async def video_endpoint(range: str = Header(None)):
    ```
3. `extract` endpoint streams the silhouete extracted video with no bounding box to show that the subject has been properly extracted.
    ```python
    @app.get("/extract")
    async def extractor_endpoint(range: str = Header(None)):
    ```
4. `opengait` endpoint runs the model on the background on the set of images to create an embedding vector of the subject in the video. The embedding vector is compared with other embedding vectors in the database using the smallest Euclidean distance to give a subject-id of another video that matches with the embedding vector of the original video.
    ```python
    @app.get("/opengait")
    async def opengait_embeddings():
    ```
5. `embed1` endpoint streams the original video
    ```python
    @app.get("/embed1", response_class=StreamingResponse)
    async def video_endpoint(range: str = Header(None)):
    ```
6. `embed2` endpoint streams the video that has the most similar embedding vector to the subject in the original video
    ```python
    @app.get("/embed2", response_class=StreamingResponse)
    async def video_endpoint(range: str = Header(None)):
    ```