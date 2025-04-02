import os, sys
from fastapi import FastAPI,  File, UploadFile, Depends, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import argparse
import uvicorn
from typing import List, Optional
from uuid import uuid4
import base64, requests, io, datetime
from pathlib import Path
import numpy as np
import json
from PIL import Image
import imageio
from video_api.services.wan_i2v_service import (
    WanI2V_CkptConfig,
    WanI2V_Input,
    WanI2V_Service,
)

device = "cuda"
OUTPUT_PATH = Path("outputs")

class VideoGenInput(BaseModel):
    prompt: str = Field(default=Query("High quality video", description="영상 생성용 프롬프트"))
    num_frames: int = Field(default=Query(25, description="생성 영상 프레임 수"))
    seed: Optional[int] = Field(default=Query(-1, description="시드"))

class VideoGenOutput(BaseModel):
    videos: List = None

app = FastAPI(
    title="비디오 생성모델 서빙용 API",
    description="""
    작성자 : 권용재
    버전 : 0.1.0
    """,
    summary="FastAPI repository to service the latest video generative models.",
    prefix="/wan_i2v",
)

ckpt_config = json.loads(Path("assets/configs/wan_i2v_ckpt_config.json").read_bytes())
config = WanI2V_CkptConfig(**ckpt_config, device=device)
service = WanI2V_Service(config)
app.mount("/static", StaticFiles(directory=str(OUTPUT_PATH)), name="static")

@app.post("/run", status_code=201, description="비디오 모델")
async def run(
    item: VideoGenInput = Depends(),
    image: UploadFile = File(
        ...,
        description="입력 이미지, base64 로 인코딩 되어있어야 합니다. 가급적 가로/세로변이 64의 배수로 맞춰져있도록 하는게 좋습니다.",
    ),
):
    max_size = 720
    batch_size = 1
    image_inp = await image.read()
    image = Image.open(io.BytesIO(image_inp)).convert("RGB")

    o_width, o_height = image.size
    width, height = o_width, o_height
    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)
        width, height = int(width * ratio), int(height * ratio)
    width, height = int(width // 16 * 16), int(height // 16 * 16)
    seed = np.random.randint(2**31) if item.seed == -1 else item.seed

    input_config = {
        "prompt": item.prompt,
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, bad quality video, sfw, bad anatomy, mutation, flashing, blurry",
        "image": image,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "length": item.num_frames,
        "sampler_name" : "euler_ancestral_RF", 
        "steps": 20,
        "step_multiplier": 0.75,
        "shift": 5.0,
        "cfg_1": 6.0,
        "cfg_2": 1.0,
        "seed": seed,
    }
    inp = WanI2V_Input(**input_config)
    outputs = service(inp)

    tz = datetime.timezone(datetime.timedelta(hours=9))
    session_time = datetime.datetime.now(tz).strftime("%Y%m%d_%H%M%S")

    output_name = f"{session_time}_{str(uuid4())}"
    (OUTPUT_PATH / output_name).mkdir(parents=True, exist_ok=True)
    
    for i, frames in enumerate(outputs.images):
        video_file = str(OUTPUT_PATH / f"{output_name}/result{i}.mp4")
        with imageio.get_writer(video_file, fps=16, codec='libx264', quality=8) as writer:
            for frame in frames:
                frame = frame.resize((o_width, o_height), Image.Resampling.LANCZOS)
                writer.append_data(np.asarray(frame))

    return JSONResponse({
        "url": [
            f"http://0.0.0.0:{PORT}/static/{output_name}/result{i}.mp4"
            for i in range(batch_size)
        ]
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI with custom args")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=31055, help="Port to run the server on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    PORT = args.port
    uvicorn.run("router:app", host=args.host, port=args.port, reload=args.reload)