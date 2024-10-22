from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {'message':'안녕'}

# @app.post("/squat")
# async def squat():
#     return {'':''}

# @app.post("/trackball")
# async def trackball():
#     return {'':''}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7900)
    #http://metaai2.iptime.org:7900/