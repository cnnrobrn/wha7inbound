from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Union, Dict, Any
from openai import OpenAI
import logging
from enum import Enum
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import base64


def setup_logging():
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger

logger = setup_logging()

app = FastAPI()
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Review(BaseModel):
    Review: str
    Giving: str 
    Evaluation_style: int
    Evaluation_fit: int
    Evaluation_color: int
    Evaluation_matching: int
    Evaluation_trendiness: int
    Evaluation_overall_look: int

class ImageAnalysisRequest(BaseModel):
    base64_image: Optional[str] = None
    text_input: Optional[str] = None

prompt = """You are the world's premier fashion consultant. Millions are seeking out your knowledge on what items look good on them.

You give good but fair feedback on what items look good on people and why certain items may not look good on them. When something doesn't quite go, you suggest alternative sthat would better suit the person.

You make comments on the clothing, fitting, matching, style, colors, and overall look of the outfit.

You do not comment on whether the person is overweight, underweight, or any other physical attributes. You only comment on the clothing and how it fits the person.

EVALUATIONS:
You also provide evaluations out of 100 for the following attributes:
- Style
- Fit
- Color
- Matching
- Trendiness
- Overall Look

GIVING:
You also provide a culturally relevant reference to what the outfit represents:
[existing styles list...]

REVIEW:
Please also provide a text response that compliments the indivudal and gives a summary of your evaluation. Please include lots of emoji's that fit the genz culture. Please end every recommendation with "Your outfit would be evelated by ..."
"""

@app.post("/upload-image/")
async def upload_image(file: Optional[UploadFile] = File(None), text_input: Optional[str] = None):
    try:
        base64_encoded = None
        if file:
            content = await file.read()
            base64_encoded = base64.b64encode(content).decode('utf-8')
        
        # Create request object for analyze_image
        image_request = ImageAnalysisRequest(
            base64_image=base64_encoded,
            text_input=text_input
        )
        
        # Call analyze_image function
        result = await analyze_image(image_request)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def analyze_image(request: ImageAnalysisRequest):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert at structured data extraction. You will be given a photo and should convert it into the given structure."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ]

        # Add text input if provided
        if request.text_input:
            messages[1]["content"].append({
                "type": "text",
                "text": f"The user is asking this question. Your primary goal should be to help provide guidance for the following: {request.text_input}"
            })

        # Add image if provided
        if request.base64_image:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{request.base64_image}"
                },
            })

        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=Review,
            max_tokens=2000,
        )
        return response.choices[0].message.parsed

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image analysis: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
